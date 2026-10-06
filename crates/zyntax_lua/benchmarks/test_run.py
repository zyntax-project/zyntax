import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run


class Results(unittest.TestCase):
    def parse(self, stdout):
        completed = subprocess.CompletedProcess([], 0, stdout, "")
        with patch.object(run.subprocess, "run", return_value=completed):
            return run.run_once(["lua"], "test.lua", {})

    def test_keeps_every_tree_check(self):
        _, result, _, error = self.parse(
            "stretch tree of depth 15 check: -1\n"
            "32768 trees of depth 4 check: -32768\n"
            "long lived tree of depth 14 check: -1\nelapsed: 0.25\n"
        )
        self.assertIsNone(error)
        self.assertEqual(len(result), 3)
        self.assertIn("-32768", result[1])

    def test_missing_result_is_a_failure(self):
        self.assertEqual(self.parse("elapsed: 0.01\n")[3], "no result output")

    def test_nonfinite_time_is_a_failure(self):
        self.assertEqual(self.parse("result: 1\nelapsed: 1e999\n")[3], "invalid elapsed time")

    def invoke(self, answers):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "results.json"
            argv = ["run.py", "--cranelift", "--runs", "2", "--only", "fib", "--out", str(report)]
            samples = [(0.1, [f"result: {answer}"], 0.2, None) for answer in answers]
            with (
                patch.object(run.sys, "argv", argv),
                patch.object(run, "find", return_value="/fake/lua"),
                patch.object(run, "run_once", side_effect=samples),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                status = run.main()
            return status, json.loads(report.read_text())["results"]["fib"]

    def test_a_correct_last_round_cannot_hide_an_earlier_wrong_answer(self):
        status, report = self.invoke([2, 1, 1, 1, 1, 1])
        self.assertEqual(status, 1)
        self.assertEqual(report["zylua"]["error"], "result changed between rounds")
        self.assertEqual(report["zylua"]["results"], [["result: 2"], ["result: 1"]])

    def test_consistent_wrong_answer_fails_against_lua(self):
        status, report = self.invoke([2, 1, 1, 2, 1, 1])
        self.assertEqual(status, 1)
        self.assertEqual(report["zylua"]["error"], "result differs from Lua")

    def test_matching_rounds_succeed(self):
        status, report = self.invoke([1] * 6)
        self.assertEqual(status, 0)
        self.assertTrue(all(r["error"] is None and len(r["results"]) == 2 for r in report.values()))


if __name__ == "__main__":
    unittest.main()
