---
sidebar_position: 5
title: Length-table generator
---

# Length-table generator

Some public traces record only how many tokens each request had, not what
the prompt said: the Azure LLM inference traces, and length tables such as
the ones Vidur ships for Arxiv summarization. The `length-table` generator
turns any CSV with a per-request input length and output length into the
[flat JSONL format](/docs/workloads/jsonl-format) and keeps every row's
lengths exactly as they are.

## Quick run

From the repository root:

```bash
python -m workloads.generators length-table \
  --source AzureLLMInferenceTrace_conv.csv \
  --input-col ContextTokens --output-col GeneratedTokens \
  --timestamp-col TIMESTAMP \
  --output workloads/azure-conv.jsonl
```

`python -m workloads.generators length-table --help` lists every option.

## Arrival times

Exactly one source of arrival times is used:

| Option | Arrival times |
| --- | --- |
| `--timestamp-col` | The table's own timestamps, shifted so the first emitted request arrives at 0. Row order is kept; a table whose timestamps go backwards is rejected. |
| `--sps` | A seeded Poisson process. This is an experiment parameter, not a property of the data, so do not present it as a recorded arrival trace. |
| neither | Every request arrives at 0 (offline / static). |

`--start-row` and `--num-reqs` select a contiguous window of the table, which
keeps the local burstiness of a real trace.

## No token IDs

The generator writes `input_toks`, `output_toks` and `arrival_time_ns` only.
Without prompt content there is nothing to hash, and invented IDs would fake
prefix sharing, so run the simulator with `--no-enable-prefix-caching`.
Prefix-cache results from these traces would not mean anything.

## Validation

A row whose length is missing, non-numeric, zero or negative stops the run
with the row number and column name. Nothing is silently dropped.
