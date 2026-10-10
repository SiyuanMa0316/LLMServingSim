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
        self.load = [0] * self.slots           # full pages per bank
        self.dev_load = [0] * self.devices
        self.inst = {}                         # (request id, kv head) -> (device, [bank of each page])
        self.reqs = {}                         # request id -> Request, while it holds pages
        self.pages = 0

    def _free(self, rid):
        for h in range(self.kv_heads):
            ent = self.inst.pop((rid, h), None)
            if ent is None:
                continue
            d, banks = ent
            for s in banks:
                self.load[s] -= 1
            self.dev_load[d] -= len(banks)
            self.pages -= len(banks)
        self.reqs.pop(rid, None)

    def sync(self, batch):
        """Free the pages of requests that left the running set; place the full pages this iteration's decode
        requests have reached. Idempotent for a given batch."""
        for rid in [rid for rid, req in self.reqs.items() if req.status != RequestStatus.RUNNING]:
            self._free(rid)
        for req, q, k in zip(batch.requests, batch.q_list, batch.k_list):
            if q != 1:
                continue
            n = k // self.P
            if n == 0:
                continue
            self.reqs[req.id] = req
            for h in range(self.kv_heads):
                ent = self.inst.get((req.id, h))
                if ent is None:
                    d = min(range(self.devices), key=self.dev_load.__getitem__)
                    ent = self.inst[(req.id, h)] = (d, [])
                d, banks = ent
                while len(banks) < n:
                    s = min(range(d * self.banks, (d + 1) * self.banks), key=self.load.__getitem__)
                    self.load[s] += 1
                    self.dev_load[d] += 1
                    self.pages += 1
                    banks.append(s)

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
        if pages == self.pages:                # every resident page is attended: the bank counts are the running ones
            busiest = max(self.load) if self.pages else 0
        else:
            count = {}
            for rid in ids:
                for h in range(self.kv_heads):
                    ent = self.inst.get((rid, h))
                    for s in (ent[1] if ent else ()):
                        count[s] = count.get(s, 0) + 1
            busiest = max(count.values(), default=0)
        new_kv = len(ids) * self.kv_heads * self.kv_bytes_token_head
        pins = (pages * self.partial_bytes + inst * self.q_bytes + new_kv) / self.pin_Bpns
        host = pages * self.partial_bytes / self.host_Bpns
        return busiest * self.page_ns + pins + host

    def stats(self):
        mean = self.pages / self.slots
        return {"pages": self.pages, "busiest_bank": max(self.load) if self.pages else 0, "mean_bank": mean}
