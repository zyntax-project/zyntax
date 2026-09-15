#!/usr/bin/env python3
"""Run the PyPy speed-center kernels on zypy, PyPy and CPython, the way
the speed center runs them, and report each interpreter against CPython.

Each kernel is the file the PyPy benchmarks repository carries (see
kernels/REVISION), unmodified, run as ``<interpreter> kernel.py -n N
[extra args]`` with N the speed center's iteration count; the kernel
prints one time per iteration, as its ``main`` measured it, and the
speed center's number for an interpreter is the mean of those, so that
is the ratio reported here, beside the whole process's wall time,
which a program that runs once pays in full.

    run.py [--fast] [--zypy PATH] [--pypy PATH] [--python PATH]
           [--only NAME,...] [--out speed.json]

A kernel an interpreter cannot run is a row that says so, with the
last line of its stderr; nothing is edited to fit.
"""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
KERNELS = os.path.join(HERE, "kernels")
SHIMS = os.path.join(HERE, "shims")
# A kernel prints times, not answers, so a kernel that runs on zypy is
# only a number once checks/<name>.py, which imports the kernel and
# prints what it computes for a small input, prints the same on zypy
# as on CPython. A kernel without a check is reported as unchecked.
CHECKS = os.path.join(HERE, "checks")

# Name on the speed center's chart -> (script, extra arguments,
# iteration scaling, files the script imports or reads). The trials
# and scalings are the speed center's: 50 iterations (5 with --fast)
# times the scaling.
SUITE = {
    "ai": ("bm_ai.py", [], 1.0, []),
    "chaos": ("chaos.py", [], 1.0, []),
    "crypto_pyaes": ("crypto_pyaes.py", [], 1.0, ["pyaes.py"]),
    "deltablue": ("deltablue.py", [], 1.0, []),
    "fannkuch": ("fannkuch.py", [], 1.0, []),
    "float": ("float.py", [], 1.0, []),
    "go": ("go.py", [], 1.0, []),
    "hexiom2": ("hexiom2.py", [], 0.1, []),
    "json_bench": ("json_bench.py", [], 1.0, []),
    "meteor-contest": ("meteor-contest.py", [], 1.0, []),
    "nbody_modified": ("nbody_modified.py", [], 1.0, []),
    "nqueens": ("nqueens.py", [], 0.1, []),
    "pidigits": ("pidigits.py", [], 0.1, []),  # needs an int wider than 64 bits
    "pyflate-fast": ("pyflate-fast.py", [], 1.0, ["interpreter.tar.bz2"]),
    "raytrace-simple": ("raytrace-simple.py", [], 1.0, []),
    "richards": ("bm_richards.py", [], 1.0, ["richards.py"]),
    "scimark_fft": ("scimark.py", ["--benchmark=FFT", "1024", "1000"], 0.1, []),
    "scimark_lu": ("scimark.py", ["--benchmark=LU", "100", "200"], 0.1, []),
    "scimark_montecarlo": ("scimark.py", ["--benchmark=MonteCarlo", "5000000"], 0.1, []),
    "scimark_sor": ("scimark.py", ["--benchmark=SOR", "100", "5000", "Array2D"], 0.1, []),
    "scimark_sparsematmult": (
        "scimark.py",
        ["--benchmark=SparseMatMult", "1000", "50000", "2000"],
        0.1,
        [],
    ),
    "spectral-norm": ("spectral-norm.py", [], 1.0, []),
    "telco": ("telco.py", [], 1.0, ["telco-bench.b"]),
}


# What a kernel computes with that zypy does not have: run on zypy it
# would finish with the wrong answer, which the kernel does not print,
# so the row says this instead of a time.
UNMET = {
    "pidigits": "int is 64 bits; the digits need arbitrary precision",
}


def interpreters(args):
    """(name, argv prefix, whether the optparse shim goes beside the
    kernel) for each interpreter found."""
    found = []
    zypy = args.zypy or os.path.join(HERE, "..", "..", "..", "..", "target", "release", "zypy")
    if os.path.exists(zypy):
        found.append(("zypy", [os.path.abspath(zypy), "run"], True))
    else:
        print(f"zypy not found at {zypy}; build it with cargo build --release -p zyntax_python", file=sys.stderr)
    pypy = args.pypy or shutil.which("pypy3")
    if pypy:
        found.append(("pypy", [pypy], False))
    python = args.python or shutil.which("python3") or sys.executable
    found.append(("cpython", [python], False))
    return found


def stage(kernel, extra_files, with_optparse):
    """A directory holding one kernel and what it imports, as the
    speed center's own/ directory would beside it."""
    d = tempfile.mkdtemp(prefix="speed-")
    for f in [kernel] + extra_files:
        shutil.copy(os.path.join(KERNELS, f), d)
    shutil.copy(os.path.join(SHIMS, "util.py"), d)
    if with_optparse:
        shutil.copy(os.path.join(SHIMS, "optparse.py"), d)
    return d


def check(prefix, python, kernel, extra_files, name, timeout):
    """Whether zypy computes what CPython computes: None when there is
    no check for the kernel, else the two outputs."""
    script = os.path.join(CHECKS, name + ".py")
    if not os.path.exists(script):
        return None
    outputs = []
    for argv, shim in ((prefix, True), ([python], False)):
        d = stage(kernel, extra_files, shim)
        try:
            shutil.copy(script, os.path.join(d, "check.py"))
            done = subprocess.run(argv + [os.path.join(d, "check.py")], cwd=d, capture_output=True, text=True, timeout=timeout)
            outputs.append(done.stdout if done.returncode == 0 else "exit " + str(done.returncode) + ": " + reason(done.stderr))
        except subprocess.TimeoutExpired:
            outputs.append("timeout")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    return outputs


def run_one(prefix, d, kernel, trials, extra, timeout):
    command = prefix + [os.path.join(d, kernel), "-n", str(trials)] + extra
    started = time.perf_counter()
    try:
        done = subprocess.run(command, cwd=d, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "wall_ms": timeout * 1000.0}
    wall = (time.perf_counter() - started) * 1000.0
    if done.returncode != 0:
        return {"status": "failed", "wall_ms": wall, "error": reason(done.stderr)}
    times = []
    for line in done.stdout.splitlines():
        try:
            times.append(float(line))
        except ValueError:
            pass
    if len(times) != trials:
        return {
            "status": "failed",
            "wall_ms": wall,
            "error": f"printed {len(times)} times for {trials} iterations",
        }
    return {
        "status": "ok",
        "wall_ms": wall,
        "iterations": times,
        "mean_ms": statistics.fmean(times) * 1000.0,
        "min_ms": min(times) * 1000.0,
    }


def reason(stderr):
    """The line of a failed run's stderr that says why: the diagnostic's
    own line, or the last line that is not part of a drawing."""
    lines = [l.strip() for l in stderr.splitlines() if l.strip()]
    for l in lines:
        if l.startswith(("Error:", "zypy:")) or "Error" in l and not l.startswith(("File", "Traceback")):
            return l[:200]
    plain = [l for l in lines if not set(l) <= set("│╭╰─┬┴╯├┤ ")]
    return (plain or [""])[-1][:200]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fast", action="store_true", help="5 iterations instead of 50, times the scaling")
    parser.add_argument("--zypy")
    parser.add_argument("--pypy")
    parser.add_argument("--python")
    parser.add_argument("--only", help="comma-separated kernel names")
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds per run")
    parser.add_argument("--out", default=os.path.join(HERE, "speed.json"))
    args = parser.parse_args()

    interps = interpreters(args)
    names = list(SUITE)
    if args.only:
        wanted = args.only.split(",")
        names = [n for n in names if n in wanted]
    base = 5 if args.fast else 50

    results = {}
    for name in names:
        kernel, extra, scaling, files = SUITE[name]
        trials = max(1, int(base * scaling))
        results[name] = {"trials": trials}
        print(f"==> {name} ({kernel} -n {trials} {' '.join(extra)})", file=sys.stderr)
        for interp, prefix, shim in interps:
            if interp == "zypy" and name in UNMET:
                results[name][interp] = {"status": "unsupported", "error": UNMET[name]}
                print(f"    {interp:<8} unsupported: {UNMET[name]}", file=sys.stderr)
                continue
            d = stage(kernel, files, shim)
            try:
                r = run_one(prefix, d, kernel, trials, extra, args.timeout)
            finally:
                shutil.rmtree(d, ignore_errors=True)
            if interp == "zypy" and r["status"] == "ok":
                python = [i for i in interps if i[0] == "cpython"][0][1][0]
                outputs = check(prefix, python, kernel, files, name, args.timeout)
                if outputs is None:
                    r["status"] = "unchecked"
                    r["error"] = "no checks/" + name + ".py to compare the answer with CPython's"
                elif outputs[0] != outputs[1]:
                    r["status"] = "wrong"
                    r["error"] = "check prints " + outputs[0].strip()[:60] + " where CPython prints " + outputs[1].strip()[:60]
            results[name][interp] = r
            if r["status"] == "ok":
                print(f"    {interp:<8} mean {r['mean_ms']:9.2f} ms  min {r['min_ms']:9.2f} ms  wall {r['wall_ms']:9.1f} ms", file=sys.stderr)
            else:
                print(f"    {interp:<8} {r['status']}: {r.get('error', '')}", file=sys.stderr)

    with open(args.out, "w") as f:
        json.dump({"kernels": results, "interpreters": [i[0] for i in interps]}, f, indent=1)
    print(table(results, [i[0] for i in interps]))
    print(f"\nwritten to {args.out}")


def table(results, interps):
    """Each interpreter's mean iteration time and wall time as a ratio
    to CPython's, the speed center's own chart being the first."""
    others = [i for i in interps if i != "cpython"]
    head = f"{'kernel':<24}" + "".join(f"{i + ' iter':>14}{i + ' wall':>14}" for i in others)
    lines = [head, "-" * len(head)]
    for name, r in results.items():
        c = r.get("cpython", {})
        row = f"{name:<24}"
        for i in others:
            x = r.get(i, {})
            if x.get("status") == "ok" and c.get("status") == "ok":
                row += f"{x['mean_ms'] / c['mean_ms']:>13.3f}x{x['wall_ms'] / c['wall_ms']:>13.3f}x"
            else:
                row += f"{x.get('status', 'missing'):>14}{'':>14}"
        lines.append(row)
    lines.append("")
    lines.append("iter: mean time of one iteration as the kernel measures it, relative to CPython (the speed center's number); wall: whole process, relative to CPython.")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
