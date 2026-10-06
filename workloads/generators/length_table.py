"""Length table (CSV) → LLMServingSim JSONL.

For traces that only record per-request token counts (Azure LLM inference
traces, Vidur's processed Arxiv / BWB tables). Every row becomes one flat
request with the source row's exact lengths::

    {"input_toks": <int>, "output_toks": <int>, "arrival_time_ns": <int>}

No token IDs are emitted, so run the simulator with
``--no-enable-prefix-caching``: the source has no prompt content, and
synthetic IDs would only fake prefix sharing.

Arrival times come from exactly one of:

* ``--timestamp-col`` — real timestamps (anything numpy parses as a
  datetime), shifted so the first emitted request arrives at 0.
* ``--sps`` — a Poisson process at that rate, seeded with ``--seed``. The
  first request arrives at 0. This is an experiment parameter, not a
  property of the source data.
* neither — every request arrives at 0 (static / offline).

Row order is preserved. ``--start-row`` / ``--num-reqs`` select a contiguous
window of the table.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def register_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source", required=True, help="Input CSV with a header row.")
    p.add_argument("--input-col", required=True, dest="input_col",
                   help="Column holding the input (prefill) token count.")
    p.add_argument("--output-col", required=True, dest="output_col",
                   help="Column holding the output (decode) token count.")
    p.add_argument("--output", required=True, help="Output JSONL path.")
    p.add_argument("--timestamp-col", default=None, dest="timestamp_col",
                   help="Column of real arrival timestamps. Exclusive with --sps.")
    p.add_argument("--sps", type=float, default=None,
                   help="Poisson arrival rate (requests / sec) for tables "
                        "without timestamps. Exclusive with --timestamp-col.")
    p.add_argument("--seed", type=int, default=42,
                   help="(--sps) RNG seed. Default 42.")
    p.add_argument("--start-row", type=int, default=0, dest="start_row",
                   help="Index of the first data row to emit. Default 0.")
    p.add_argument("--num-reqs", type=int, default=0, dest="num_reqs",
                   help="Number of rows to emit from --start-row (0 = all).")


def read_length_table(path, input_col, output_col, timestamp_col=None,
                      start_row=0, num_reqs=0):
    """Return ``(input_toks, output_toks, timestamps_ns_or_None)`` for the window."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for col in (input_col, output_col, timestamp_col):
            if col is not None and col not in (reader.fieldnames or []):
                raise ValueError(f"{path}: no column {col!r} (have {reader.fieldnames})")
        rows = list(reader)

    end = start_row + num_reqs if num_reqs > 0 else len(rows)
    rows = rows[start_row:end]
    if not rows:
        raise ValueError(f"{path}: no rows selected (start_row={start_row}, num_reqs={num_reqs})")

    def lengths(col):
        out = []
        for i, row in enumerate(rows, start=start_row):
            raw = row[col]
            try:
                value = int(raw)
            except (TypeError, ValueError):
                raise ValueError(f"{path}: data row {i}, column {col!r}: "
                                 f"expected a positive integer, got {raw!r}") from None
            if value <= 0:
                raise ValueError(f"{path}: data row {i}, column {col!r}: "
                                 f"expected a positive integer, got {value}")
            out.append(value)
        return out

    inputs, outputs = lengths(input_col), lengths(output_col)
    if timestamp_col is None:
        return inputs, outputs, None

    try:
        ts = np.array([row[timestamp_col] for row in rows], dtype="datetime64[ns]")
    except ValueError as exc:
        raise ValueError(f"{path}: column {timestamp_col!r} is not parseable "
                         f"as timestamps: {exc}") from None
    if np.isnat(ts).any():
        raise ValueError(f"{path}: column {timestamp_col!r} has empty timestamps")
    rel_ns = (ts - ts[0]).astype("int64")
    if (np.diff(rel_ns) < 0).any():
        raise ValueError(f"{path}: column {timestamp_col!r} goes backwards; "
                         f"sort the table first")
    return inputs, outputs, [int(t) for t in rel_ns]


def poisson_arrivals_ns(n, sps, seed):
    """``n`` nondecreasing arrival times (ns); the first is 0."""
    gaps = np.random.default_rng(seed).exponential(scale=1e9 / sps, size=n - 1)
    return [0] + [int(t) for t in np.cumsum(gaps.astype("int64"))]


def run(args: argparse.Namespace) -> int:
    if args.timestamp_col is not None and args.sps is not None:
        print("error: --timestamp-col and --sps are mutually exclusive", file=sys.stderr)
        return 2
    if args.sps is not None and args.sps <= 0:
        print("error: --sps must be positive", file=sys.stderr)
        return 2

    try:
        inputs, outputs, arrivals = read_length_table(
            args.source, args.input_col, args.output_col, args.timestamp_col,
            args.start_row, args.num_reqs)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if arrivals is None:
        arrivals = (poisson_arrivals_ns(len(inputs), args.sps, args.seed)
                    if args.sps is not None else [0] * len(inputs))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fout:
        for in_toks, out_toks, t_ns in zip(inputs, outputs, arrivals):
            fout.write(json.dumps({"input_toks": in_toks, "output_toks": out_toks,
                                   "arrival_time_ns": t_ns}) + "\n")

    print(f"Wrote {len(inputs)} requests -> {out_path}")
    return 0
