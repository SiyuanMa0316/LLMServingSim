import argparse
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from serving.core.router import Router
from workloads.generators import length_table as lt

AZURE = (
    "TIMESTAMP,ContextTokens,GeneratedTokens\n"
    "2023-11-16 18:15:46.6805900,374,44\n"
    "2023-11-16 18:15:50.9951690,396,109\n"
    "2023-11-16 18:15:50.9951690,20,7\n"
    "2023-11-16 18:16:01.0000000,1024,3\n"
)
LENGTHS = "num_prefill_tokens,num_decode_tokens\n3772,54\n2015,156\n10,5\n"


class LengthTableTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def write(self, name, text):
        path = self.dir / name
        path.write_text(text)
        return str(path)

    def convert(self, source, in_col, out_col, **kw):
        out = str(self.dir / "out.jsonl")
        args = argparse.Namespace(
            source=source, input_col=in_col, output_col=out_col, output=out,
            timestamp_col=kw.get("timestamp_col"), sps=kw.get("sps"),
            seed=kw.get("seed", 42), start_row=kw.get("start_row", 0),
            num_reqs=kw.get("num_reqs", 0))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            rc = lt.run(args)
        return rc, out

    def rows(self, out):
        return [json.loads(line) for line in Path(out).read_text().splitlines()]

    def test_real_timestamps_and_lengths_preserved(self):
        src = self.write("azure.csv", AZURE)
        rc, out = self.convert(src, "ContextTokens", "GeneratedTokens", timestamp_col="TIMESTAMP")
        self.assertEqual(rc, 0)
        rows = self.rows(out)
        self.assertEqual([r["input_toks"] for r in rows], [374, 396, 20, 1024])
        self.assertEqual([r["output_toks"] for r in rows], [44, 109, 7, 3])
        # Exact ns from the 7-digit fractional seconds; first request at 0.
        self.assertEqual([r["arrival_time_ns"] for r in rows],
                         [0, 4_314_579_000, 4_314_579_000, 14_319_410_000])
        self.assertTrue(all(set(r) == {"input_toks", "output_toks", "arrival_time_ns"} for r in rows))

    def test_window_renormalizes_first_arrival(self):
        src = self.write("azure.csv", AZURE)
        rc, out = self.convert(src, "ContextTokens", "GeneratedTokens",
                               timestamp_col="TIMESTAMP", start_row=1, num_reqs=2)
        self.assertEqual(rc, 0)
        rows = self.rows(out)
        self.assertEqual([r["input_toks"] for r in rows], [396, 20])
        self.assertEqual([r["arrival_time_ns"] for r in rows], [0, 0])

    def test_static_arrivals_are_all_zero(self):
        src = self.write("len.csv", LENGTHS)
        rc, out = self.convert(src, "num_prefill_tokens", "num_decode_tokens")
        self.assertEqual(rc, 0)
        self.assertEqual([r["arrival_time_ns"] for r in self.rows(out)], [0, 0, 0])

    def test_poisson_is_deterministic_sorted_and_seed_dependent(self):
        src = self.write("len.csv", LENGTHS + "10,5\n" * 20)
        _, a = self.convert(src, "num_prefill_tokens", "num_decode_tokens", sps=5.0, seed=7)
        first = Path(a).read_text()
        _, b = self.convert(src, "num_prefill_tokens", "num_decode_tokens", sps=5.0, seed=7)
        self.assertEqual(first, Path(b).read_text())
        times = [r["arrival_time_ns"] for r in self.rows(b)]
        self.assertEqual(times[0], 0)
        self.assertEqual(times, sorted(times))
        self.assertGreater(times[-1], 0)
        _, c = self.convert(src, "num_prefill_tokens", "num_decode_tokens", sps=5.0, seed=8)
        self.assertNotEqual(first, Path(c).read_text())

    def test_rejects_bad_lengths(self):
        for bad in ("0", "-3", "abc", ""):
            src = self.write("bad.csv", f"i,o\n5,7\n{bad},9\n")
            rc, _ = self.convert(src, "i", "o")
            self.assertEqual(rc, 1, msg=f"input value {bad!r}")

    def test_bad_length_error_names_row_and_column(self):
        src = self.write("bad.csv", "i,o\n5,7\n6,x\n")
        with self.assertRaisesRegex(ValueError, r"data row 1, column 'o'"):
            lt.read_length_table(src, "i", "o")

    def test_rejects_missing_column_and_backwards_timestamps(self):
        src = self.write("len.csv", LENGTHS)
        self.assertEqual(self.convert(src, "nope", "num_decode_tokens")[0], 1)
        back = self.write("back.csv", "t,i,o\n2023-01-01 00:00:02,5,5\n2023-01-01 00:00:01,5,5\n")
        self.assertEqual(self.convert(back, "i", "o", timestamp_col="t")[0], 1)

    def test_timestamp_col_and_sps_are_exclusive(self):
        src = self.write("azure.csv", AZURE)
        rc, _ = self.convert(src, "ContextTokens", "GeneratedTokens",
                             timestamp_col="TIMESTAMP", sps=1.0)
        self.assertEqual(rc, 2)

    def test_router_loads_generated_file(self):
        src = self.write("azure.csv", AZURE)
        _, out = self.convert(src, "ContextTokens", "GeneratedTokens", timestamp_col="TIMESTAMP")
        # Router.load_requests prefixes '../' to the path it is given.
        rel = os.path.relpath(out, os.path.join(os.getcwd(), ".."))
        router = Router(num_instances=0, schedulers=[], req_num=0)
        router.load_requests(rel, enable_prefix_caching=False)
        pending = router._pending_requests
        self.assertEqual([r["input_toks"] for r in pending], [374, 396, 20, 1024])
        # The router stores input + output as the request's total length.
        self.assertEqual([r["output_toks"] for r in pending], [418, 505, 27, 1027])
        self.assertEqual([r["arrival_time_ns"] for r in pending],
                         [0, 4_314_579_000, 4_314_579_000, 14_319_410_000])


if __name__ == "__main__":
    unittest.main()
