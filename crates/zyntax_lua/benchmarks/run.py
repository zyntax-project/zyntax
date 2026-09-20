#!/usr/bin/env python3
"""Run the benchmarks here on zylua, LuaJIT and PUC Lua, and report each
against PUC Lua.

Every benchmark prints `elapsed: <seconds>` for the work it timed and,
where it computes one, `result: <n>`; the three interpreters must
agree on the result or the row says so. The benchmark's own time and
the whole process's wall time (which a program run once pays in full,
start-up included) are both reported. The runs are interleaved, one
round of every interpreter after another, so a drift in the machine's
load falls on all three alike, and the median is reported with the
spread.

zylua is measured as a performance release is: an `llvm-backend`
build run with `ZYLUA_LLVM=1`, checked through `zylua backend`, which
must print `llvm`.

    run.py [--runs N] [--zylua PATH] [--luajit PATH] [--lua PATH]
           [--only NAME,...] [--out bench.json] [--cranelift]

An interpreter that fails a benchmark is a row that says so, with the
last line of its stderr.
"""

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# The five shared with the other language benches, then the kernels
# ZynML and Python measure.
BENCHES = [
    "fib",
    "binary_trees",
    "mandelbrot",
    "method_call",
    "nbody",
    "any_field",
    "branchy",
    "collatz",
    "inlined_call",
    "free_function_call",
    "lambda_call",
    "closure_call",
    "op_overload",
    "mandelbrot_strip",
    "records",
    "pipeline",
    "tokenize",
]


def find(name, candidates):
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return shutil.which(name)


def run_once(cmd, path, env):
    started = time.perf_counter()
    proc = subprocess.run(
        cmd + [path], capture_output=True, text=True, cwd=HERE, stdin=subprocess.DEVNULL, env=env
    )
    wall = time.perf_counter() - started
    if proc.returncode != 0:
        last = (proc.stderr.strip().split("\n") or [""])[-1]
        return None, None, wall, f"exit {proc.returncode}: {last}"
    m = re.search(r"^elapsed:\s*([0-9.eE+-]+)", proc.stdout, re.M)
    if not m:
        return None, None, wall, "no elapsed line"
    r = re.search(r"^(?:result|energy|checksum):\s*(-?\d+)", proc.stdout, re.M)
    return float(m.group(1)), (r.group(1) if r else None), wall, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--zylua")
    ap.add_argument("--luajit")
    ap.add_argument("--lua")
    ap.add_argument("--only")
    ap.add_argument("--out")
    ap.add_argument("--cranelift", action="store_true", help="measure zylua without the LLVM tier")
    args = ap.parse_args()

    root = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
    interpreters = {
        "zylua": find("zylua", [args.zylua, os.path.join(root, "target", "release", "zylua")]),
        "luajit": find("luajit", [args.luajit, "/opt/homebrew/bin/luajit"]),
        "lua": find("lua5.4", [args.lua, "/opt/homebrew/opt/lua@5.4/bin/lua5.4", "/opt/homebrew/bin/lua"]),
    }
    for name, path in interpreters.items():
        if not path:
            sys.exit(f"{name}: not found; pass --{name}")
    env = dict(os.environ)
    if not args.cranelift:
        env["ZYLUA_LLVM"] = "1"
        backend = subprocess.run(
            [interpreters["zylua"], "backend"], capture_output=True, text=True, env=env
        ).stdout.strip()
        if backend != "llvm":
            sys.exit(
                f"zylua reports the {backend!r} backend: build with "
                "LLVM_SYS_211_PREFIX=/opt/homebrew/opt/llvm cargo build --release -p zyntax_lua "
                "--features llvm-backend, or pass --cranelift to measure that tier alone"
            )
    commands = {
        "zylua": [interpreters["zylua"], "run"],
        "luajit": [interpreters["luajit"]],
        "lua": [interpreters["lua"]],
    }
    benches = args.only.split(",") if args.only else BENCHES

    results = {
        b: {i: {"elapsed": [], "wall": [], "result": None, "error": None} for i in commands}
        for b in benches
    }
    for round_ in range(args.runs):
        for bench in benches:
            path = os.path.join(HERE, bench + ".lua")
            for name, cmd in commands.items():
                r = results[bench][name]
                if r["error"]:
                    continue
                elapsed, result, wall, err = run_once(cmd, path, env)
                if err:
                    r["error"] = err
                    continue
                r["elapsed"].append(elapsed)
                r["wall"].append(wall)
                r["result"] = result
            print(f"round {round_ + 1}/{args.runs}: {bench}", file=sys.stderr)

    def stat(xs):
        if not xs:
            return None
        return {"median": statistics.median(xs), "min": min(xs), "max": max(xs), "runs": len(xs)}

    report = {}
    print(
        f"{'benchmark':<20} {'interp':<7} {'elapsed ms':>11} {'spread':>18} {'wall ms':>9} {'vs lua':>8}"
    )
    for bench in benches:
        base = stat(results[bench]["lua"]["elapsed"])
        answers = {results[bench][n]["result"] for n in commands if results[bench][n]["result"]}
        report[bench] = {}
        for name in commands:
            r = results[bench][name]
            e, w = stat(r["elapsed"]), stat(r["wall"])
            report[bench][name] = {"elapsed": e, "wall": w, "result": r["result"], "error": r["error"]}
            if r["error"]:
                print(f"{bench:<20} {name:<7} {r['error']}")
                continue
            ratio = f"{base['median'] / e['median']:.2f}x" if base and e["median"] else "-"
            flag = "  (result differs)" if len(answers) > 1 else ""
            print(
                f"{bench:<20} {name:<7} {e['median'] * 1000:11.1f} "
                f"{e['min'] * 1000:8.1f}..{e['max'] * 1000:<8.1f} {w['median'] * 1000:9.1f} {ratio:>8}{flag}"
            )
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"interpreters": interpreters, "llvm": not args.cranelift, "results": report}, f, indent=2)


if __name__ == "__main__":
    main()
