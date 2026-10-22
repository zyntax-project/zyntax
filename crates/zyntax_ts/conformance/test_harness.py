import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
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

    @unittest.skipUnless(os.name == "posix", "process-group cleanup is POSIX-specific")
    def test_timeout_stops_descendants_that_inherit_output_pipes(self):
        script = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(10)']); time.sleep(10)"
        start = time.monotonic()
        result = runner.bounded([sys.executable, "-c", script], 0.5)
        self.assertIsNone(result["code"])
        self.assertLess(time.monotonic() - start, 5)

    def test_reference_compiler_is_an_exact_package_pin(self):
        self.assertRegex(runner.REFERENCE_COMPILER, r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$")


REFERENCE = Path(os.environ.get("ZYNTAX_TS_REFERENCE", runner.ROOT / "node_modules/typescript"))


@unittest.skipUnless(shutil.which("node") and (REFERENCE / "package.json").is_file(),
                     "install the pinned reference compiler to exercise the native API")
class ReferenceTests(unittest.TestCase):
    def invoke(self, root, source, action="check", options=None, compiler=REFERENCE):
        entry = root / "entry.ts"
        entry.write_text(source)
        request = root / "request.json"
        request.write_text(json.dumps({"files": [str(entry)], "options": options or {}}))
        return runner.bounded(["node", str(runner.ROOT / "reference.cjs"), action,
                               str(compiler), str(request)], 10)

    def test_mismatched_compiler_is_an_infrastructure_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compiler = root / "compiler"
            compiler.mkdir()
            (compiler / "package.json").write_text('{"version":"0.0.0"}')
            result = self.invoke(root, "console.log(1);", compiler=compiler)
            self.assertNotEqual(result["code"], 0)
            self.assertIn(f"Expected TypeScript {runner.REFERENCE_COMPILER}, found 0.0.0", result["stderr"])
            self.assertEqual(result["stdout"], "")

    def test_checker_categories_and_configuration_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for source, kind in [("const n:number=1;", "Accept"),
                                 ("const n:number='wrong';", "Type"),
                                 ("function broken( {", "Syntax")]:
                result = self.invoke(root, source)
                self.assertEqual(result["code"], 0, result["stderr"])
                diagnostic = json.loads(result["stdout"])
                self.assertEqual(diagnostic["compiler"], runner.REFERENCE_COMPILER)
                self.assertEqual(diagnostic["kind"], kind)
                self.assertEqual(diagnostic["accepted"], kind == "Accept")
            result = self.invoke(root, "const n=1;", options={"unknownCompilerOption": True})
            self.assertNotEqual(result["code"], 0)
            self.assertIn("unknownCompilerOption", result["stderr"])

    def test_chained_diagnostics_keep_their_hierarchy(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.invoke(Path(directory), "const f:(n:number)=>number=(n:boolean):number=>1;")
            self.assertEqual(result["code"], 0, result["stderr"])
            diagnostic = json.loads(result["stdout"])
            self.assertFalse(diagnostic["accepted"])
            self.assertIn("\n  Types of parameters", diagnostic["diagnostics"][0]["message"])
            self.assertIn("\n    Type 'number'", diagnostic["diagnostics"][0]["message"])

    def test_multifile_resolution_and_emission_do_not_write_javascript(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sub").mkdir()
            (root / "sub/index.ts").write_text("export {default} from '../library.ts';")
            (root / "library.ts").write_text("console.log('library');export default function(n:number):number{return n+2;}")
            source = "import step from './sub';import {default as again} from './library.js';console.log(step(3));console.log(again(5));"
            options = {"allowImportingTsExtensions": True}
            checked = self.invoke(root, source, options=options)
            self.assertEqual(checked["code"], 0, checked["stderr"])
            self.assertTrue(json.loads(checked["stdout"])["accepted"], checked["stdout"])
            result = self.invoke(root, source, action="run", options=options)
            self.assertEqual(result["code"], 0, result["stderr"])
            self.assertEqual(result["stdout"], "library\n5\n7\n")
            self.assertEqual(list(root.rglob("*.js")), [])

    def test_runtime_failure_preserves_output_and_exit_status(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.invoke(Path(directory), "console.log('before');throw new Error('boom');", action="run")
            self.assertEqual(result["code"], 1)
            self.assertEqual(result["stdout"], "before\n")
            self.assertIn("boom", result["stderr"])


if __name__ == "__main__":
    unittest.main()
