"""Paged decode KV on an attention-offload device: page placement and lockstep pricing.

Used when an instance's ``decode_attention_offload`` sets ``"kv_allocator": "paged"``. The offload bundle must declare
``knobs.decode_attention.paged`` in its manifest (LLMCompass ``TokenPackedAttention.paged_record``): the page size P,
the time of one page step, the bytes of one partial output and one Q, and the device geometry.

Placement (no migration): each decode request's KV is ``n_kv_heads`` instances; an instance's full pages all sit on one
device (chosen least-loaded by pages when its first page is placed) and each new page goes to the least-loaded bank of
that device. The open page (``kv mod P`` tokens) stays in the instance's own memory (``gpu_open_page_tokens``), so the
device holds full pages only: the prompt's at the handoff, then one per P decode steps. A request's pages are freed when
it leaves the running set (finished or preempted; a preempted request's KV is rewritten when it decodes again).

Pricing (lockstep per device: every bank runs the same page step, and every full page has one shape): the attended
requests cost the most pages any bank holds of them x one page step, plus the pins (one partial per page, one Q per
instance with pages, the new tokens' K/V as amortized page writes) and the host merge of the partials.

V row blocks (manifest ``rowblock_ns`` / ``pages_per_rowblock``, LLMCompass token-packed ``sv_mapping="RMNCK"``): V is
stored tokens-on-lanes, ``pages_per_rowblock`` pages of one bank share a 1024-lane V row block, and SV runs once per row
block after the page steps. A page takes a free quarter of the fullest non-full row block of its bank (else an empty
or new one), the bank is the one whose page + row-block time grows least, and a device costs its busiest bank's pages x
page step + its busiest bank's attended row blocks x row-block step (a block with any attended page runs whole).

HBM-first page placement with spill (``gpu_resident_kv_tokens`` with ``gpu_resident_mode`` "serial", ``set_hbm_pool``):
the instance's memory holds up to that many decode KV tokens. Every decode request's open page lives there, counted at
the KV manager's block granularity; each full page goes there too while it fits (requests in batch order), and to the
device once it does not. No migration: a page stays where it was placed until its request leaves the running set, and
the pages freed then are refilled by later full pages. A request can be split between the two; the instance attends
its HBM pages with its open page, and only the device's pages count in ``cost_ns``.

Opt-in what-ifs (decode_attention_offload keys, all off by default; MultiPIMSim experiments/pim-duet-close-ideal-gap):
- ``rowblock_merge``: an instance's pages in one V row block share one softmax max (pass 1 runs over all of them before
  pass 2) and leave as one partial, the tree-output adders summing that instance's slices of the block. Partials are
  counted per (instance, row block) instead of per page; the page step and row-block step are unchanged.
- ``rowblock_affinity``: true -- a new page goes into its instance's own last row block while it has a free quarter, else
  into an empty row block of the bank whose page + row-block time is least; instances never share a row block. "soft" --
  the page goes into its instance's last row block only if that bank's page + row-block time with it is within
  ``rowblock_affinity_slack_pages`` (default 1) page steps of the default placement's choice, which it takes otherwise
  (instances may then share a row block; each instance's pages in one block still leave as one partial). "free" -- the
  page goes into its instance's last row block whenever that bank's page + row-block time with it stays within the
  busiest bank's time (of its device; of all devices with page_spread "global"), i.e. whenever merging does not lengthen
  the lockstep layer; otherwise the default placement. With page_spread "global", a page that cannot join its own row
  block first tries the devices already holding the instance (fewer Q copies) under the same test. "auto" -- merge only
  when the device side is loaded enough for the pins to matter: while the allocator holds at least
  ``rowblock_merge_min_pages`` pages (default: the count whose partials take four page steps on the pins), a page joins
  its instance's last row block whenever that block has a free quarter, else opens an empty row block on the best bank
  (instances do not share a block, so later pages can join); below it, the default placement (spread for latency).
- ``page_spread`` "global" (default "device"): a new page (with affinity: a new row block) goes to the device with the
  least page + row-block time instead of the instance's own device; Q crosses the pins once per (instance, device)
  holding its pages.
- ``pin_overlap`` (MultiPIMSim experiments/pim-duet-pin-overlap; bundles without V row blocks only): a drain path. Each
  page's partial is staged and leaves over the pins while the banks compute the next pages, and the Q of later pages
  arrives the same way, so the pins run alongside compute instead of after it. The device costs
  max(head + compute x ``pin_overlap_compute_factor`` (default 1.025: the staged readout's column slots) + tail, pins)
  instead of compute + pins. Head is the first lockstep round's Q (at most one per bank) and tail its last round's
  partials (at most one per bank), which have no compute to hide behind. The host merge stays serial.
- ``sub_batch_placement`` (MultiPIMSim experiments/pim-duet-default-audit; bundles without V row blocks only): stable
  sub-batch membership and sub-batch-aware placement, so that a lockstep device can run the two interleaved sub-batches
  the way ``sub_batch_barrier`` false prices them. A request joins the sub-batch with fewer labeled requests at its first
  decode and keeps it until it leaves the running set (``sub_batch_of``; the trace generator splits decodes by it). A new
  page goes to the device, then the bank, with the fewest pages of its request's sub-batch (ties: fewest pages; with
  page_spread "device" the instance's device as usual). Each bank then holds balanced A and B pages, so a sub-batch's
  own pass is about its share of the two together (the trace generator also floors each sub-batch's charge at its own
  pass, ``cost_ns`` of its requests).

HBM write-through page cache (``gpu_resident_cache`` "nofetch" / "fetch" with ``gpu_resident_mode`` "balanced",
``set_hbm_cache``): every full page lives on the device, as without it, and the instance's memory caches the most
recently filled pages. A page that fills was the open page (or, at the handoff, prefilled KV), so it enters the cache at
no cost. The cache holds up to ``gpu_resident_kv_tokens`` tokens including every decode request's open page (counted in
KV-manager blocks); when it is over, the oldest entries (FIFO by insertion) are dropped, for free. A page is all KV heads
of one request's P tokens. The instance may attend only cached pages ("nofetch"), or copy missing pages from the device
("fetch": the trace generator prices the copy and ``commit_plans`` inserts the fetched pages as the newest entries).
"""
from collections import OrderedDict
import json
import os

from .request import RequestStatus

_ALLOCATORS = {}


def paged_allocator(key, manifest_path, opts=None):
    """One allocator per (node, instance, hardware); its placement persists across iterations. ``opts``: the opt-in
    placement keys of the module docstring."""
    if key not in _ALLOCATORS:
        with open(manifest_path) as f:
            knobs = (json.load(f).get("knobs") or {}).get("decode_attention") or {}
        if not knobs.get("paged"):
            raise ValueError(f"kv_allocator 'paged' needs knobs.decode_attention.paged in {manifest_path}")
        _ALLOCATORS[key] = PagedKVAllocator(knobs["paged"], opts)
    return _ALLOCATORS[key]


class PagedKVAllocator:
    def __init__(self, rec, opts=None):
        self.P = int(rec["page_tokens"])
        self.page_ns = float(rec["page_ns"])
        self.partial_bytes = float(rec["partial_bytes"])
        self.q_bytes = float(rec["q_bytes"])
        self.kv_bytes_token_head = float(rec["kv_bytes_per_token_head"])
        self.pin_Bpns = float(rec["pin_bandwidth_Bps"]) / 1e9
        self.host_Bpns = float(rec["host_Bps"]) / 1e9
        self.slots = int(rec["slots"])
        self.banks = int(rec["banks_per_device"])
        self.devices = self.slots // self.banks
        self.kv_heads = int(rec["n_kv_heads"])
        self.block_ns = float(rec.get("rowblock_ns") or 0.0)
        self.ppb = int(rec.get("pages_per_rowblock") or 0)        # 0: no V row blocks (SV inside the page step)
        self.blocks = [[] for _ in range(self.slots)]             # per bank: pages in each V row block
        self.active = [0] * self.slots                            # per bank: row blocks holding >= 1 page
        self.load = [0] * self.slots           # full pages per bank
        self.dev_load = [0] * self.devices
        self.inst = {}                         # (request id, kv head) -> (device, [bank of each page])
        self.reqs = {}                         # request id -> Request, while it holds pages
        self.pages = 0
        self.hbm_pool = 0                      # decode KV tokens the instance's memory holds (0: every full page here)
        self.block = 1                         # KV-manager block (tokens): open pages count against the pool in blocks
        self.hbm = {}                          # request id -> its full pages in the instance's memory
        self.hbm_pages = 0
        self.hbm_peak = 0                      # most pool tokens in use (full + open pages) after any sync
        opts = opts or {}
        self.merge = bool(opts.get("rowblock_merge", False))
        self.affinity = opts.get("rowblock_affinity", False)
        if self.affinity not in (False, True, "soft", "free", "auto"):
            raise ValueError(f"rowblock_affinity must be true, false, 'soft', 'free' or 'auto', got {self.affinity!r}")
        self.slack_ns = float(opts.get("rowblock_affinity_slack_pages", 1)) * self.page_ns
        partial_ns = self.partial_bytes / self.pin_Bpns
        self.merge_min_pages = int(opts.get("rowblock_merge_min_pages", round(4 * self.page_ns / partial_ns)))
        self.spread = opts.get("page_spread", "device")
        if self.spread not in ("device", "global"):
            raise ValueError(f"page_spread must be 'device' or 'global', got {self.spread!r}")
        if (self.merge or self.affinity) and not self.ppb:
            raise ValueError("rowblock_merge / rowblock_affinity need a bundle with V row blocks (pages_per_rowblock)")
        self.pin_overlap = bool(opts.get("pin_overlap", False))
        self.overlap_factor = float(opts.get("pin_overlap_compute_factor", 1.025))
        if self.pin_overlap and self.ppb:
            raise ValueError("pin_overlap needs a bundle without V row blocks (pages_per_rowblock 0)")
        self.last_cost = {}                    # the last cost_ns call's terms (ns), for observers
        self.sub_aware = bool(opts.get("sub_batch_placement", False))
        if self.sub_aware and self.ppb:
            raise ValueError("sub_batch_placement needs a bundle without V row blocks (pages_per_rowblock 0)")
        self.sub_batch = {}                    # sub_batch_placement: request id -> its sub-batch (0 / 1), for its life
        self.sub_reqs = {}                     # ... request id -> Request, while labeled
        self.sub_count = [0, 0]                # ... labeled requests per sub-batch
        self.sub_bank = [[0] * self.slots for _ in range(2)]     # ... pages of each sub-batch per bank
        self.sub_dev = [[0] * self.devices for _ in range(2)]    # ... and per device
        self.opt_place = self.affinity or self.spread == "global" or self.sub_aware
        self.track = self.merge or self.spread == "global"
        self.meta = {}                         # (request id, kv head) -> (its row blocks {(bank, block)}, its devices)
        self.dev_active = [0] * self.devices   # page_spread "global": row blocks holding >= 1 page, per device
        self.cache_on = False                  # HBM write-through page cache (set_hbm_cache)

    def set_hbm_pool(self, tokens, block):
        """HBM-first page placement with spill (module docstring). Call before the first ``sync``."""
        if self.opt_place or self.track:
            raise ValueError("HBM-first spill is not combined with rowblock_merge / rowblock_affinity / page_spread / "
                             "sub_batch_placement")
        self.hbm_pool, self.block = int(tokens), max(1, int(block))

    def set_hbm_cache(self, tokens, block, fetch):
        """HBM write-through page cache (module docstring). Call before the first ``sync``."""
        if self.hbm_pool:
            raise ValueError("the HBM page cache is not combined with HBM-first spill")
        self.cache_on, self.cache_pool, self.block, self.cache_fetch = True, int(tokens), max(1, int(block)), bool(fetch)
        self.cache = OrderedDict()             # (request id, page index) -> None, oldest first
        self.cached = {}                       # request id -> set of its cached page indices
        self.cache_seen = {}                   # request id -> pages already filled (and inserted once)
        self.cache_cap = self.cache_pool       # this iteration's room for full pages (pool - open pages)
        self.plan_sink = None                  # the trace generator's per-sub-batch plans while it synthesizes a trace
        self.last_cache = {}                   # the kept trace's plans, summed (for observers)
        self.cache_totals = dict(iterations=0, short=0, fetched_pages=0, evicted_pages=0, inserted_pages=0)

    def _cache_insert(self, rid, i):
        self.cache[(rid, i)] = None
        self.cached.setdefault(rid, set()).add(i)

    def _cache_evict(self):
        while len(self.cache) * self.P > self.cache_cap and self.cache:
            (rid, i), _ = self.cache.popitem(last=False)
            self.cached[rid].discard(i)
            self.cache_totals["evicted_pages"] += 1

    def cached_tokens(self, rid):
        """Full-page tokens of request ``rid`` in the instance's memory (HBM page cache)."""
        return len(self.cached.get(rid, ())) * self.P

    def commit_plans(self, plans):
        """Apply the kept trace's per-sub-batch plans: fetch the pages they copy from the device (newest entries)."""
        tot = dict(short=0, fetch_tokens=0, full_tokens=0, cached_tokens=0, x_tokens=0.0, x_free_tokens=0.0)
        for ids, ks, fetch_tokens, xc, x, x_free in plans.values():
            full = sum((k // self.P) * self.P for k in ks)
            tot["full_tokens"] += full
            tot["cached_tokens"] += xc * full
            tot["x_tokens"] += x * full
            tot["x_free_tokens"] += x_free * full
            tot["short"] |= x_free > xc + 1e-9
            need = fetch_tokens
            for rid, k in zip(ids, ks):
                if need <= 0:
                    break
                have = self.cached.get(rid, set())
                for i in range(k // self.P - 1, -1, -1):
                    if need <= 0:
                        break
                    if i not in have:
                        self._cache_insert(rid, i)
                        self.cache_totals["fetched_pages"] += 1
                        tot["fetch_tokens"] += self.P
                        need -= self.P
        self._cache_evict()
        self.cache_totals["iterations"] += 1
        self.cache_totals["short"] += tot["short"]
        tot["cache_tokens"] = len(self.cache) * self.P
        self.last_cache = tot

    def hbm_tokens(self, rid):
        """Full-page tokens of request ``rid`` in the instance's memory."""
        return self.hbm.get(rid, 0) * self.P

    def sub_batch_of(self, rid):
        """sub_batch_placement: request ``rid``'s sub-batch (0 / 1), or None if it is not labeled."""
        return self.sub_batch.get(rid)

    def on_device(self, rid):
        """Whether request ``rid`` has any full page on the device."""
        return (rid, 0) in self.inst

    def device_tokens(self, rid):
        """Full-page tokens of request ``rid`` on the device."""
        ent = self.inst.get((rid, 0))
        return len(ent[1]) * self.P if ent else 0

    def _free(self, rid):
        for h in range(self.kv_heads):
            ent = self.inst.pop((rid, h), None)
            if ent is None:
                continue
            d, banks = ent
            for s, b in banks:
                self.load[s] -= 1
                if self.sub_aware:
                    self.sub_bank[self.sub_batch[rid]][s] -= 1
                    self.sub_dev[self.sub_batch[rid]][s // self.banks] -= 1
                if b is not None:
                    self.blocks[s][b] -= 1
                    self.active[s] -= self.blocks[s][b] == 0
                    if self.spread == "global" and self.blocks[s][b] == 0:
                        self.dev_active[s // self.banks] -= 1
            if self.spread == "global":
                for s, _ in banks:
                    self.dev_load[s // self.banks] -= 1
            else:
                self.dev_load[d] -= len(banks)
            self.pages -= len(banks)
            self.meta.pop((rid, h), None)
        self.hbm_pages -= self.hbm.pop(rid, 0)
        self.reqs.pop(rid, None)
        if self.cache_on:
            for i in self.cached.pop(rid, ()):
                del self.cache[(rid, i)]
            self.cache_seen.pop(rid, None)

    def sync(self, batch):
        """Free the pages of requests that left the running set; place the full pages this iteration's decode
        requests have reached. Idempotent for a given batch."""
        for rid in [rid for rid, req in self.reqs.items() if req.status != RequestStatus.RUNNING]:
            self._free(rid)
        if self.sub_aware:                     # labels of requests that left the running set (pages freed above)
            for rid in [rid for rid, req in self.sub_reqs.items() if req.status != RequestStatus.RUNNING]:
                self.sub_count[self.sub_batch.pop(rid)] -= 1
                del self.sub_reqs[rid]
        if self.affinity == "free":            # busiest bank per device (kept up to date as pages are placed)
            self._dev_max = [max(self._bank_ns(x) for x in range(dv * self.banks, (dv + 1) * self.banks))
                             for dv in range(self.devices)]
        free = None
        if self.hbm_pool:
            open_tokens = sum(-(-(k % self.P) // self.block) * self.block
                              for q, k in zip(batch.q_list, batch.k_list) if q == 1)
            free = self.hbm_pool - self.hbm_pages * self.P - open_tokens
        for req, q, k in zip(batch.requests, batch.q_list, batch.k_list):
            if q != 1:
                continue
            if self.sub_aware and req.id not in self.sub_batch:   # first decode: join the smaller sub-batch, for life
                sub = 0 if self.sub_count[0] <= self.sub_count[1] else 1
                self.sub_batch[req.id], self.sub_reqs[req.id] = sub, req
                self.sub_count[sub] += 1
            n = k // self.P
            if n == 0:
                continue
            self.reqs[req.id] = req
            if free is not None:
                in_hbm = self.hbm.get(req.id, 0)
                to_hbm = max(0, min(n - in_hbm - self.device_tokens(req.id) // self.P, free // self.P))
                if to_hbm:
                    self.hbm[req.id] = in_hbm = in_hbm + to_hbm
                    self.hbm_pages += to_hbm
                    free -= to_hbm * self.P
                n -= in_hbm                    # the rest are (or go) on the device
                if n == 0:
                    continue
            for h in range(self.kv_heads):
                ent = self.inst.get((req.id, h))
                if ent is None:
                    d = min(range(self.devices), key=self.dev_load.__getitem__)
                    ent = self.inst[(req.id, h)] = (d, [])
                    if self.track:
                        self.meta[(req.id, h)] = (set(), set())
                d, banks = ent
                while self.opt_place and len(banks) < n:
                    self._place_opt(req.id, h, d, banks)
                while len(banks) < n:
                    bank_range = range(d * self.banks, (d + 1) * self.banks)
                    if not self.ppb:
                        s, b = min(bank_range, key=self.load.__getitem__), None
                    else:
                        s = min(bank_range, key=self._placement_ns)
                        b = self._block_for(s)
                        if b == len(self.blocks[s]):
                            self.blocks[s].append(0)
                        self.active[s] += self.blocks[s][b] == 0
                        self.blocks[s][b] += 1
                    self.load[s] += 1
                    self.dev_load[d] += 1
                    self.pages += 1
                    banks.append((s, b))
                    if self.track:
                        self.meta[(req.id, h)][0].add((s, b))
                        self.meta[(req.id, h)][1].add(d)
        if free is not None:
            self.hbm_peak = max(self.hbm_peak, self.hbm_pool - free)
        if self.cache_on:                      # newly filled pages enter the cache; open pages take their blocks
            open_tokens = sum(-(-(k % self.P) // self.block) * self.block
                              for q, k in zip(batch.q_list, batch.k_list) if q == 1)
            self.cache_cap = max(0, self.cache_pool - open_tokens)
            for req, q, k in zip(batch.requests, batch.q_list, batch.k_list):
                if q != 1:
                    continue
                n = k // self.P
                for i in range(self.cache_seen.get(req.id, 0), n):
                    self._cache_insert(req.id, i)
                    self.cache_totals["inserted_pages"] += 1
                if n > self.cache_seen.get(req.id, 0):
                    self.cache_seen[req.id] = n
            self._cache_evict()

    def _bank_ns(self, s):
        return self.load[s] * self.page_ns + self.active[s] * self.block_ns

    def _dev_ns(self, d):
        """Device d's page + row-block time summed over its banks (page_spread "global" placement)."""
        return self.dev_load[d] * self.page_ns + self.dev_active[d] * self.block_ns

    def _place_opt(self, rid, h, d, banks):
        """Place one page of instance (rid, h) under rowblock_affinity / page_spread "global" / sub_batch_placement
        (module docstring)."""
        if self.sub_aware:                     # fewest pages of its sub-batch, then fewest pages (device, then bank)
            sub = self.sub_batch[rid]
            if self.spread == "global":
                d = min(range(self.devices), key=lambda x: (self.sub_dev[sub][x], self.dev_load[x]))
            s = min(range(d * self.banks, (d + 1) * self.banks), key=lambda x: (self.sub_bank[sub][x], self.load[x]))
            return self._commit(rid, h, d, banks, s, None)
        s = b = None
        if self.affinity == "auto":
            if self.pages < self.merge_min_pages:
                return self._place_default_opt(rid, h, d, banks)
            if banks:
                s, b = banks[-1]
                if self.blocks[s][b] >= self.ppb:
                    s = b = None
            if s is None:                      # open an empty row block on the best bank (of the best device)
                if self.spread == "global":
                    d = min(range(self.devices), key=self._dev_ns)
                bank_range = range(d * self.banks, (d + 1) * self.banks)
                s = min(bank_range, key=lambda x: self.load[x] * self.page_ns + self.active[x] * self.block_ns)
                b = next((i for i, c in enumerate(self.blocks[s]) if c == 0), len(self.blocks[s]))
            return self._commit(rid, h, d, banks, s, b)
        if self.affinity and banks:
            s, b = banks[-1]
            if self.blocks[s][b] >= self.ppb:
                s = b = None
        if s is not None and self.affinity == "free":
            t_own = (self.load[s] + 1) * self.page_ns + self.active[s] * self.block_ns
            if t_own > (max(self._dev_max) if self.spread == "global" else self._dev_max[s // self.banks]):
                s = b = None
        if s is None and self.affinity == "free" and self.spread == "global" and banks:
            held = self.meta[(rid, h)][1]       # devices already holding the instance: no new Q copy
            cand = min((x for dv in held for x in range(dv * self.banks, (dv + 1) * self.banks)), key=self._placement_ns)
            if self._placement_ns(cand) <= max(self._dev_max):
                s, b = cand, self._block_for(cand)
        if s is None or self.affinity == "soft":
            own = (s, b)
            if self.spread == "global":
                d = min(range(self.devices), key=self._dev_ns)
            bank_range = range(d * self.banks, (d + 1) * self.banks)
            if not self.ppb:
                s, b = min(bank_range, key=self.load.__getitem__), None
            elif self.affinity is True:
                s = min(bank_range, key=lambda x: self.load[x] * self.page_ns + self.active[x] * self.block_ns)
                b = next((i for i, c in enumerate(self.blocks[s]) if c == 0), len(self.blocks[s]))
            else:
                s = min(bank_range, key=self._placement_ns)
                b = self._block_for(s)
                if own[0] is not None and ((self.load[own[0]] + 1) * self.page_ns + self.active[own[0]] * self.block_ns
                                           <= self._placement_ns(s) + self.slack_ns):
                    s, b = own
        self._commit(rid, h, d, banks, s, b)

    def _place_default_opt(self, rid, h, d, banks):
        """The default placement rule (least page + row-block growth, fullest non-full block), on the instance's device
        or, with page_spread "global", on the least-loaded device."""
        if self.spread == "global":
            d = min(range(self.devices), key=self._dev_ns)
        bank_range = range(d * self.banks, (d + 1) * self.banks)
        s = min(bank_range, key=self._placement_ns)
        self._commit(rid, h, d, banks, s, self._block_for(s))

    def _commit(self, rid, h, d, banks, s, b):
        """Record one page of instance (rid, h) on bank s, row block b (d: the instance's device without global spread)."""
        dev = s // self.banks
        if b is not None:
            if b == len(self.blocks[s]):
                self.blocks[s].append(0)
            if self.blocks[s][b] == 0:
                self.active[s] += 1
                if self.spread == "global":
                    self.dev_active[dev] += 1
            self.blocks[s][b] += 1
        self.load[s] += 1
        self.dev_load[dev if self.spread == "global" else d] += 1
        self.pages += 1
        banks.append((s, b))
        if self.sub_aware:
            self.sub_bank[self.sub_batch[rid]][s] += 1
            self.sub_dev[self.sub_batch[rid]][dev] += 1
        if self.track:
            self.meta[(rid, h)][0].add((s, b))
            self.meta[(rid, h)][1].add(dev)
        if self.affinity == "free":
            self._dev_max[dev] = max(self._dev_max[dev], self._bank_ns(s))

    def _block_for(self, s):
        """Row block of bank s for a new page: the fullest one with a free quarter, else an empty one, else a new one."""
        best = None
        for b, n in enumerate(self.blocks[s]):
            if 0 < n < self.ppb and (best is None or n > self.blocks[s][best]):
                best = b
        if best is not None:
            return best
        return next((b for b, n in enumerate(self.blocks[s]) if n == 0), len(self.blocks[s]))

    def _placement_ns(self, s):
        """Bank s's page + row-block time with one more page."""
        b = self._block_for(s)
        opens = b == len(self.blocks[s]) or self.blocks[s][b] == 0
        return (self.load[s] + 1) * self.page_ns + (self.active[s] + opens) * self.block_ns

    def cost_ns(self, decode_ids):
        """Offloaded attention time (ns) of the decode requests ``decode_ids`` (one layer)."""
        ids = list(decode_ids)
        pages = inst = 0
        for rid in ids:
            for h in range(self.kv_heads):
                ent = self.inst.get((rid, h))
                if ent and ent[1]:
                    pages += len(ent[1])
                    inst += 1
        # New tokens' K/V reach the device as page writes; with an HBM pool only a request with pages there spills more.
        writers = len(ids) if not self.hbm_pool else sum(self.on_device(rid) for rid in ids)
        new_kv = writers * self.kv_heads * self.kv_bytes_token_head
        partials, qcopies = pages, inst
        if self.track:                         # rowblock_merge: one partial per (instance, row block); page_spread
            partials = qcopies = 0             # "global": one Q per (instance, device)
            for rid in ids:
                for h in range(self.kv_heads):
                    ent = self.inst.get((rid, h))
                    if ent and ent[1]:
                        m = self.meta[(rid, h)]
                        partials += len(m[0]) if self.merge else len(ent[1])
                        qcopies += len(m[1]) if self.spread == "global" else 1
        pins = (partials * self.partial_bytes + qcopies * self.q_bytes + new_kv) / self.pin_Bpns
        host = partials * self.partial_bytes / self.host_Bpns
        if self.ppb:
            compute = self._rowblock_compute_ns(ids, pages == self.pages)
        else:
            if pages == self.pages:            # every resident page is attended: the bank counts are the running ones
                busiest = max(self.load) if self.pages else 0
            else:
                count = {}
                for rid in ids:
                    for h in range(self.kv_heads):
                        ent = self.inst.get((rid, h))
                        for s, _ in (ent[1] if ent else ()):
                            count[s] = count.get(s, 0) + 1
                busiest = max(count.values(), default=0)
            compute = busiest * self.page_ns
        self.last_cost = dict(compute=compute, pins=pins, host=host, head=0.0, tail=0.0)
        if not self.pin_overlap:
            return compute + pins + host
        head = min(qcopies, pages, self.slots) * self.q_bytes / self.pin_Bpns
        tail = min(partials, self.slots) * self.partial_bytes / self.pin_Bpns
        self.last_cost.update(head=head, tail=tail)
        return max(head + compute * self.overlap_factor + tail, pins) + host

    def _rowblock_compute_ns(self, ids, all_attended):
        """Lockstep per device with V row blocks: page steps for the busiest bank's pages, then row-block steps for the
        busiest bank's attended row blocks; the slowest device sets the layer."""
        if all_attended:
            pages, blocks = self.load, self.active
        else:
            pages, seen = [0] * self.slots, set()
            for rid in ids:
                for h in range(self.kv_heads):
                    ent = self.inst.get((rid, h))
                    for s, b in (ent[1] if ent else ()):
                        pages[s] += 1
                        seen.add((s, b))
            blocks = [0] * self.slots
            for s, _ in seen:
                blocks[s] += 1
        worst = 0.0
        for d in range(self.devices):
            r = range(d * self.banks, (d + 1) * self.banks)
            worst = max(worst, max(pages[s] for s in r) * self.page_ns + max(blocks[s] for s in r) * self.block_ns)
        return worst

    def stats(self):
        mean = self.pages / self.slots
        out = {"pages": self.pages, "busiest_bank": max(self.load) if self.pages else 0, "mean_bank": mean}
        if self.ppb:
            out.update(busiest_rowblocks=max(self.active), rowblocks=sum(self.active))
        if self.hbm_pool:
            out.update(hbm_pages=self.hbm_pages, hbm_peak_tokens=self.hbm_peak)
        return out
