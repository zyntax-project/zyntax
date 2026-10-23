#!/usr/bin/env python3
"""Compare native TypeScript checking/output with a pinned TypeScript/Node oracle."""
import argparse
import collections
import concurrent.futures
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile

from fetch_upstream import inventory, verify
from upstream_baseline import regressions

ROOT = Path(__file__).resolve().parent
PROFILES = ("interpreter", "baseline", "llvm")
REFERENCE_COMPILER = json.loads((ROOT / "package.json").read_text())["devDependencies"]["typescript"]


def bounded(argv, timeout, env=None):
    try:
        with subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                              env=env, start_new_session=os.name == "posix") as run:
            try:
                stdout, stderr = run.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                # The reference API owns a native compiler child in the same session.
                if os.name == "posix":
                    try:
                        os.killpg(run.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    run.kill()
                run.communicate()
                return {"code": None, "stdout": "", "stderr": f"timed out after {timeout}s"}
            return {"code": run.returncode, "stdout": stdout.replace("\r\n", "\n"),
                    "stderr": stderr.replace("\r\n", "\n")}
    except OSError as error:
        return {"code": None, "stdout": "", "stderr": str(error)}


def known_failures(path):
    result = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2 or not re.fullmatch(r"[0-9a-f]{64}", fields[1]):
            raise ValueError(f"Invalid known failure: {line}")
        if fields[0] in result:
            raise ValueError(f"Duplicate known failure: {fields[0]}")
        result[fields[0]] = fields[1]
    return result


def verdict(ok, gap, known):
    if known:
        return "XPASS" if ok else "XFAIL"
    return "PASS" if ok else "UNSUPPORTED" if gap else "FAIL"


def diagnostic(outcome):
    for line in reversed(outcome["stderr"].splitlines()):
        try:
            value = json.loads(line)
            if isinstance(value, dict) and "kind" in value:
                return value
        except ValueError:
            pass
    return {}


def compare(case, reference, check, execution=None):
    native = diagnostic(check)
    gap = native.get("kind") == "Unsupported"
    if case.get("policy_error"):
        ok = reference["accepted"] and check["code"] == 5 and native.get("kind") == "Soundness"
        return ok, False, "additional checked-policy rejection" if ok else "wrong policy result"
    if reference["accepted"]:
        if check["code"] != 0:
            return False, gap, native.get("message", check["stderr"])
        if case["mode"] == "run":
            ok = execution is not None and execution["code"] == reference["exit"] and execution["stdout"] == reference["stdout"]
            return ok, False, "runtime output/status" if not ok else ""
        return True, False, ""
    ok = check["code"] == 1 and native.get("kind") == reference["kind"]
    return ok, gap, "diagnostic category differs" if not ok else ""


def oracle(args, case, files, options):
    with tempfile.TemporaryDirectory(prefix="zynode-reference-") as directory:
        request = Path(directory) / "request.json"
        request.write_text(json.dumps({"files": [str(p.resolve()) for p in files], "options": options}))
        checked = bounded([args.node, str(ROOT / "reference.cjs"), "check", str(args.typescript), str(request)], args.timeout)
        if checked["code"] != 0:
            raise RuntimeError(checked["stderr"])
        reference = json.loads(checked["stdout"])
        if reference["accepted"] and case["mode"] == "run":
            result = bounded([args.node, str(ROOT / "reference.cjs"), "run", str(args.typescript), str(request)], args.timeout)
            if result["code"] is None:
                raise RuntimeError(result["stderr"])
            reference.update(stdout=result["stdout"], exit=result["code"])
        return reference


def source_hash(files):
    digest = hashlib.sha256()
    base = Path(os.path.commonpath([p.parent for p in files]))
    for p in files:
        digest.update(p.relative_to(base).as_posix().encode())
        digest.update(b"\0")
        digest.update(p.read_bytes())
    return digest.hexdigest()


def native(args, case, file, profile):
    switches = ["--diagnostics=json", "--soundness", case.get("soundness", "typescript")]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ZYNTAX_", "ZYPY_", "ZYLUA_"))}
    if profile == "interpreter":
        switches += ["--baseline-threshold", str(2**32 - 1), "--no-osr"]
        env["ZYNTAX_DISABLE_WARM_UP"] = "1"
    else:
        switches += ["--baseline-threshold", "2"]
    if profile == "llvm":
        switches += ["--llvm"]
    checked = bounded([str(args.zynode), "check", *switches, str(file)], args.timeout, env)
    execution = None
    if checked["code"] == 0 and case["mode"] == "run" and not case.get("policy_error"):
        execution = bounded([str(args.zynode), "run", *switches, str(file)], args.timeout, env)
    return checked, execution


def upstream_cases(path):
    """Split official @filename files and expand compiler-option variations."""
    content = path.read_text(encoding="utf-8-sig")
    files = {}
    name = path.name
    lines = []
    directives = {}
    virtual_file = False
    for line in content.splitlines(keepends=True):
        match = re.match(r"\s*//\s*@([\w]+)\s*:\s*(.*?)\s*$", line)
        if match:
            key, value = match.groups()
            if key.lower() == "filename":
                if lines and (virtual_file or any(line.strip() and not line.lstrip().startswith("//") for line in lines)):
                    files[name] = "".join(lines)
                name, lines = value, []
                virtual_file = True
            else:
                directives[key.lower()] = value
        else:
            lines.append(line)
    if lines or virtual_file:
        files[name] = "".join(lines)
    return files, directives


def option_variants(directives):
    compiler_keys = {
        "strict", "strictNullChecks", "strictFunctionTypes", "noImplicitAny", "noImplicitReturns",
        "noUncheckedIndexedAccess", "exactOptionalPropertyTypes", "target", "module", "moduleResolution",
        "useDefineForClassFields", "noEmit", "allowJs", "checkJs", "alwaysStrict", "lib", "noLib",
        "strictPropertyInitialization", "noImplicitThis", "esModuleInterop", "resolveJsonModule",
        "noUnusedLocals", "noUnusedParameters", "noFallthroughCasesInSwitch", "useUnknownInCatchVariables",
    }
    canonical = {key.lower(): key for key in compiler_keys}
    ignored = {"declaration", "emitdeclarationonly", "sourcemap", "removecomments", "notypesandsymbols"}
    directives = {key.lower(): value for key, value in directives.items()}
    unknown = set(directives) - canonical.keys() - ignored
    if unknown:
        raise ValueError("unsupported upstream directives: " + ", ".join(sorted(unknown)))
    entries = []
    for key, value in directives.items():
        if key in ignored:
            continue
        values = [v.strip() for v in value.split(",")]
        if any(v == "*" or v.startswith(("-", "!")) for v in values):
            raise ValueError("upstream wildcard/exclusion option matrices are not implemented yet")
        if key == "lib":
            entries.append((canonical[key], [values]))
        else:
            entries.append((canonical[key], [True if v.lower() == "true" else False if v.lower() == "false" else v for v in values]))
    if not entries:
        return [{}]
    if math.prod(len(values) for _, values in entries) > 64:
        raise ValueError("too many compiler-option combinations")
    return [dict(zip((k for k, _ in entries), values)) for values in itertools.product(*(v for _, v in entries))]


def execute_case(args, case, files, options, pins, known):
    stamp = source_hash(files)
    if args.live or args.update or pins is None:
        reference = oracle(args, case, files, options)
        reference.update(source_sha256=stamp, options=options)
        if pins is None and any(not d.get("file") for d in reference["diagnostics"]):
            return [{"case": case["id"], "status": "SKIP", "options": options,
                     "reason": "reference configuration/global diagnostics prevent a source comparison",
                     "reference": reference}]
        if args.update and pins is not None:
            pins.write_text(json.dumps(reference, indent=2) + "\n")
    else:
        reference = json.loads(pins.read_text())
        if reference["source_sha256"] != stamp or reference["options"] != options or reference["compiler"] != REFERENCE_COMPILER:
            raise RuntimeError(f"stale reference for {case['id']}; use --update with the pinned compiler")
    rows = []
    for profile in args.profile:
        check, execution = native(args, case, files[0], profile)
        ok, gap, reason = compare(case, reference, check, execution)
        status = verdict(ok, gap, case["id"] in known)
        if check["code"] is None or check["code"] < 0 or (execution and (execution["code"] is None or execution["code"] < 0)):
            status = "FAIL"
        if case["id"] in known and not ok and not gap:
            status = "FAIL"
        row = {"case": case["id"], "profile": profile, "status": status, "reason": reason,
               "reference": reference, "native_check": check, "native_run": execution,
               "issue": known.get(case["id"]), "policy": case.get("soundness", "typescript")}
        rows.append(row)
        print(f"{status:11} {profile:11} {case['id']}" + (f": {reason}" if reason else ""), flush=True)
    return rows


class UpstreamSkip(ValueError):
    pass


def upstream_file(args, path):
    label = path.relative_to(args.upstream).as_posix()
    stamp = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = []

    def skip(reason, case=label, options=None):
        return {"case": case, "status": "SKIP", "reason": reason,
                "source_sha256": stamp, "options": options}

    try:
        files, directives = upstream_cases(path)
        variants = option_variants(directives)
    except (ValueError, UnicodeError) as error:
        return [skip(str(error))]
    if path.suffix != ".ts":
        return [skip("native JSX/JavaScript corpus checking is not implemented yet")]
    for i, options in enumerate(variants):
        options = {"skipLibCheck": False, **options}
        case = {"id": f"{label}#{i}", "mode": "check"}
        if any(options.get(key) is False for key in ("strict", "noImplicitAny", "strictNullChecks")):
            rows.append(skip("native relaxed-checking options are not implemented yet", case["id"], options))
            continue
        if any(Path(name).name.lower() in ("tsconfig.json", "jsconfig.json") for name in files):
            rows.append(skip("embedded tsconfig/jsconfig projects are not implemented yet", case["id"], options))
            continue
        try:
            with tempfile.TemporaryDirectory(prefix="zynode-upstream-") as directory:
                staged = []
                for name, content in files.items():
                    # Absolute POSIX virtual paths share the isolated fixture root.
                    relative = Path(name.lstrip("/"))
                    if ".." in relative.parts or "\\" in name or ":" in name or not relative.name:
                        raise UpstreamSkip(f"unsupported upstream virtual filename {name}")
                    target = Path(directory) / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(content)
                    if target.suffix == ".ts":
                        staged.append(target)
                if not staged:
                    rows.append(skip("no .ts source files", case["id"], options))
                elif len(staged) != 1:
                    rows.append(skip("upstream multi-root project checking is not implemented yet", case["id"], options))
                else:
                    rows += execute_case(args, case, staged, options, None, {})
        except UpstreamSkip as error:
            rows.append(skip(str(error), case["id"], options))
        except (ValueError, RuntimeError, OSError) as error:
            rows.append({"case": case["id"], "status": "ERROR", "reason": str(error),
                         "source_sha256": stamp, "options": options})
    for row in rows:
        row["corpus_source_sha256"] = stamp
        row["directives"] = directives
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zynode", type=Path, default=ROOT.parents[2] / "target/debug/zynode")
    parser.add_argument("--node", default="node")
    parser.add_argument("--typescript", type=Path, default=ROOT / "node_modules/typescript",
                        help="pinned TypeScript package directory")
    parser.add_argument("--profile", action="append", choices=PROFILES)
    parser.add_argument("--live", action="store_true", help="verify against the pinned compiler and Node")
    parser.add_argument("--update", action="store_true", help="refresh reference pins from the oracle")
    parser.add_argument("--only", default="")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--upstream", type=Path, help="TypeScript repository's tests/cases/conformance directory")
    parser.add_argument("--upstream-pin", type=Path, help="verify corpus provenance against a pin (upstream.json)")
    parser.add_argument("--regression-baseline", type=Path, help="gate a complete official run against reviewed per-case outcomes")
    parser.add_argument("--jobs", type=int, default=1, help="parallel upstream checker processes")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path, default=ROOT / "results.json")
    args = parser.parse_args()
    if args.timeout <= 0 or args.jobs <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("timeout, jobs and limit must be positive")
    if args.upstream_pin and not args.upstream:
        parser.error("--upstream-pin requires --upstream")
    if args.regression_baseline and (not args.upstream_pin or args.only or args.limit):
        parser.error("--regression-baseline requires the complete pinned upstream corpus")
    args.profile = args.profile or ["interpreter", "baseline"]
    args.zynode = args.zynode.resolve()
    if not args.zynode.is_file():
        raise ValueError(f"native binary not found: {args.zynode}; build zyntax_ts first")
    binary_hash = hashlib.sha256(args.zynode.read_bytes()).hexdigest()
    known = known_failures(ROOT / "KNOWN_FAILURES")
    cases = json.loads((ROOT / "cases.json").read_text())
    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)) or set(known) - set(ids):
        raise ValueError("duplicate cases or stale KNOWN_FAILURES entries")
    rows = []
    corpus = None
    if not args.upstream:
        selected = [c for c in cases if args.only in c["id"]]
        for case in selected[:args.limit]:
            file = ROOT / case["file"]
            files = [ROOT / p for p in case.get("files", [case["file"]])]
            if not files or files[0] != file:
                raise ValueError(f"entry must be the first file: {case['id']}")
            rows += execute_case(args, case, files, case.get("options", {}), file.with_suffix(".reference.json"), known)
    else:
        args.upstream = args.upstream.resolve()
        if not args.upstream.is_dir():
            raise ValueError(f"official corpus directory not found: {args.upstream}")
        if args.upstream_pin:
            pin = json.loads(args.upstream_pin.read_text())
            if pin["compiler"] != REFERENCE_COMPILER:
                raise ValueError("official corpus pin and reference compiler differ")
            package = json.loads((args.typescript / "package.json").read_text())
            if package.get("version") != REFERENCE_COMPILER or package.get("gitHead") != pin["revision"]:
                raise ValueError("installed reference compiler does not match the official corpus revision")
            corpus = verify(args.upstream, pin)
        else:
            corpus = {**inventory(args.upstream), "verified": False}
        paths = sorted(p for p in args.upstream.rglob("*") if p.suffix in (".ts", ".tsx", ".js"))
        selected = [p for p in paths if args.only in p.relative_to(args.upstream).as_posix()][:args.limit]
        corpus.update(source_files=len(paths), selected_files=len(selected),
                      extensions=dict(collections.Counter(p.suffix for p in paths)),
                      mode="check", defaults={"strict": True, "target": "es2022", "module": "commonjs",
                                             "noEmit": True, "skipLibCheck": False, "types": [], "lib": ["es2022", "dom"]})
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for count, result in enumerate(pool.map(lambda p: upstream_file(args, p), selected), 1):
                rows += result
                if count % 100 == 0 or count == len(selected):
                    print(f"Upstream files {count}/{len(selected)}: {dict(collections.Counter(r['status'] for r in rows))}", flush=True)
        corpus["completed_files"] = len(selected)
    if not rows:
        raise ValueError("no cases selected")
    counts = dict(collections.Counter(r["status"] for r in rows))
    report = {"reference_compiler": REFERENCE_COMPILER, "binary": str(args.zynode),
              "binary_sha256": binary_hash,
              "profiles": args.profile, "live_reference": bool(args.live or args.update or args.upstream),
              "counts": counts, "results": rows}
    if corpus is not None:
        report["corpus"] = corpus
    if args.regression_baseline:
        try:
            report["regressions"] = regressions(report, json.loads(args.regression_baseline.read_text()))
        except ValueError as error:
            report["regression_error"] = str(error)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    if hashlib.sha256(args.zynode.read_bytes()).hexdigest() != binary_hash:
        raise RuntimeError("native executable changed during the conformance run")
    print(json.dumps(counts, sort_keys=True))
    if report.get("regression_error"):
        print("regression gate error: " + report["regression_error"], file=sys.stderr)
        return 2
    if counts.get("ERROR") or all(row["status"] == "SKIP" for row in rows):
        return 2
    if args.regression_baseline:
        for regression in report["regressions"]:
            print("REGRESSION " + regression)
        return int(bool(report["regressions"]))
    return int(any(counts.get(name, 0) for name in ("FAIL", "UNSUPPORTED", "XPASS")))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print(f"conformance infrastructure error: {error}", file=sys.stderr)
        sys.exit(2)
