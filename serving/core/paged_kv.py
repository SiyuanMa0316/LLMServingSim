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
"""
import json
import os

from .request import RequestStatus

_ALLOCATORS = {}


def paged_allocator(key, manifest_path):
    """One allocator per (node, instance, hardware); its placement persists across iterations."""
    if key not in _ALLOCATORS:
        with open(manifest_path) as f:
            knobs = (json.load(f).get("knobs") or {}).get("decode_attention") or {}
        if not knobs.get("paged"):
            raise ValueError(f"kv_allocator 'paged' needs knobs.decode_attention.paged in {manifest_path}")
        _ALLOCATORS[key] = PagedKVAllocator(knobs["paged"])
    return _ALLOCATORS[key]


class PagedKVAllocator:
    def __init__(self, rec):
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

    def set_hbm_pool(self, tokens, block):
        """HBM-first page placement with spill (module docstring). Call before the first ``sync``."""
        self.hbm_pool, self.block = int(tokens), max(1, int(block))

    def hbm_tokens(self, rid):
        """Full-page tokens of request ``rid`` in the instance's memory."""
        return self.hbm.get(rid, 0) * self.P

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
                if b is not None:
                    self.blocks[s][b] -= 1
                    self.active[s] -= self.blocks[s][b] == 0
            self.dev_load[d] -= len(banks)
            self.pages -= len(banks)
        self.hbm_pages -= self.hbm.pop(rid, 0)
        self.reqs.pop(rid, None)

    def sync(self, batch):
        """Free the pages of requests that left the running set; place the full pages this iteration's decode
        requests have reached. Idempotent for a given batch."""
        for rid in [rid for rid, req in self.reqs.items() if req.status != RequestStatus.RUNNING]:
            self._free(rid)
        free = None
        if self.hbm_pool:
            open_tokens = sum(-(-(k % self.P) // self.block) * self.block
                              for q, k in zip(batch.q_list, batch.k_list) if q == 1)
            free = self.hbm_pool - self.hbm_pages * self.P - open_tokens
        for req, q, k in zip(batch.requests, batch.q_list, batch.k_list):
            if q != 1:
                continue
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
                d, banks = ent
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
        if free is not None:
            self.hbm_peak = max(self.hbm_peak, self.hbm_pool - free)

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
        pins = (pages * self.partial_bytes + inst * self.q_bytes + new_kv) / self.pin_Bpns
        host = pages * self.partial_bytes / self.host_Bpns
        if self.ppb:
            return self._rowblock_compute_ns(ids, pages == self.pages) + pins + host
        if pages == self.pages:                # every resident page is attended: the bank counts are the running ones
            busiest = max(self.load) if self.pages else 0
        else:
            count = {}
            for rid in ids:
                for h in range(self.kv_heads):
                    ent = self.inst.get((rid, h))
                    for s, _ in (ent[1] if ent else ()):
                        count[s] = count.get(s, 0) + 1
            busiest = max(count.values(), default=0)
        return busiest * self.page_ns + pins + host

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
