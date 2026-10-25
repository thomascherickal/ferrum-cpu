"""summarize.py: newest results file wins, the aggregate "all" entry is dropped."""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import summarize  # noqa: E402


def write_results(d, name, results, mtime):
    p = Path(d) / "run" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"results": results}))
    os.utime(p, (mtime, mtime))


class SummarizeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.res = Path(self.tmp.name) / "results"
        self.out = Path(self.tmp.name) / "deep" / "dir" / "out.json"

    def test_picks_newest_drops_all_and_records_metadata(self):
        write_results(self.res, "results_old.json", {"old|0": {"acc": 0.1}}, 1_000)
        write_results(self.res, "results_new.json", {"all": {"acc": 9}, "t|0": {"acc": 0.5}}, 2_000)
        tasks = summarize.summarize(self.res, self.out, "m.gguf", "20")
        self.assertEqual(tasks, {"t|0": {"acc": 0.5}})
        self.assertEqual(
            json.loads(self.out.read_text()),
            {"lighteval_version": "0.13.0", "model": "m.gguf", "max_samples": 20, "tasks": {"t|0": {"acc": 0.5}}},
        )

    def test_max_samples_absent_is_null(self):
        write_results(self.res, "results_1.json", {"t|0": {"acc": 1}}, 1)
        for absent in (None, ""):
            summarize.summarize(self.res, self.out, "m", absent)
            self.assertIsNone(json.loads(self.out.read_text())["max_samples"])

    def test_no_results_file_is_a_clear_error(self):
        self.res.mkdir()
        with self.assertRaises(SystemExit) as ctx:
            summarize.summarize(self.res, self.out, "m")
        self.assertIn("no results_*.json", str(ctx.exception))
        self.assertFalse(self.out.exists())

    def test_main_prints_tasks_and_validates_argc(self):
        write_results(self.res, "results_1.json", {"t|0": {"acc": 1}}, 1)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            summarize.main([str(self.res), str(self.out), "m"])
            summarize.main([str(self.res), str(self.out), "m", "5"])
        self.assertEqual(json.loads(buf.getvalue().split("\n}\n")[0] + "\n}"), {"t|0": {"acc": 1}})
        for bad in ([], ["a"], ["a", "b"], ["a", "b", "c", "d", "e"]):
            with self.assertRaises(SystemExit):
                summarize.main(bad)


if __name__ == "__main__":
    unittest.main()
