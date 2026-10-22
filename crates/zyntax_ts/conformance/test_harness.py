import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("zynode_conformance", Path(__file__).with_name("run.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class HarnessTests(unittest.TestCase):
    def test_unsupported_is_not_a_type_error_pass(self):
        case = {"mode": "check"}
        reference = {"accepted": False, "kind": "Type"}
        native = {"code": 3, "stderr": '{"kind":"Unsupported","message":"union"}', "stdout": ""}
        ok, gap, _ = runner.compare(case, reference, native)
        self.assertFalse(ok)
        self.assertTrue(gap)
        self.assertEqual(runner.verdict(ok, gap, False), "UNSUPPORTED")

    def test_unexpected_pass_fails_known_failure(self):
        self.assertEqual(runner.verdict(True, False, True), "XPASS")

    def test_runtime_status_and_output_are_both_checked(self):
        case = {"mode": "run"}
        ref = {"accepted": True, "stdout": "42\n", "exit": 0}
        accepted = {"code": 0, "stderr": "", "stdout": ""}
        self.assertFalse(runner.compare(case, ref, accepted, {"code": 0, "stdout": "0\n"})[0])
        self.assertFalse(runner.compare(case, ref, accepted, {"code": 1, "stdout": "42\n"})[0])

    def test_checked_policy_is_reported_separately(self):
        native = {"code": 5, "stderr": '{"kind":"Soundness"}', "stdout": ""}
        self.assertTrue(runner.compare({"mode": "check", "policy_error": True}, {"accepted": True}, native)[0])
        self.assertFalse(runner.compare({"mode": "check"}, {"accepted": True}, native)[0])

    def test_upstream_virtual_files_and_option_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory) / "case.ts"
            case.write_text("// Test preamble\n\n// @strict: true\n// @target: es2022, esnext\n\n// @filename: a.ts\nexport const x = 1;\n// @filename: b.ts\nimport { x } from './a';\n")
            files, directives = runner.upstream_cases(case)
            self.assertEqual(set(files), {"a.ts", "b.ts"})
            self.assertEqual(runner.option_variants(directives), [{"strict": True, "target": "es2022"}, {"strict": True, "target": "esnext"}])
            with self.assertRaises(ValueError):
                runner.option_variants({"unknownDirective": "true"})
            with self.assertRaises(ValueError):
                runner.option_variants({"target": ",".join(["es2022"] * 100),
                                        "module": ",".join(["commonjs"] * 100)})

    def test_known_failures_require_full_issue_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "KNOWN_FAILURES"
            path.write_text("case short\n")
            with self.assertRaises(ValueError):
                runner.known_failures(path)

    def test_multifile_pins_hash_dependency_contents_and_relative_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "entry.ts"
            dependency = root / "sub" / "library.ts"
            dependency.parent.mkdir()
            entry.write_text("import {f} from './sub/library'; f();")
            dependency.write_text("export function f():number{return 1;}")
            stamp = runner.source_hash([entry, dependency])
            dependency.write_text("export function f():number{return 2;}")
            self.assertNotEqual(stamp, runner.source_hash([entry, dependency]))
            moved = root / "library.ts"
            moved.write_text("export function f():number{return 1;}")
            self.assertNotEqual(stamp, runner.source_hash([entry, moved]))

    def test_hung_process_is_bounded(self):
        result = runner.bounded([sys.executable, "-c", "import time; time.sleep(10)"], 0.05)
        self.assertIsNone(result["code"])
        self.assertIn("timed out", result["stderr"])


if __name__ == "__main__":
    unittest.main()
