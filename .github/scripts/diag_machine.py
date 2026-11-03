#!/usr/bin/env python3
"""Describe a CI machine and measure what zypy does on it.

    diag_machine.py --zypy PATH [--pypy PATH] --out FILE [--repeats N] [--kernels a,b,c]

GitHub documents neither the CPU model nor how the vCPUs of a runner map to
cores, and the kernels' times differ between machines of one model, so this
measures what the benchmarks depend on:

* the topology the kernel reports (thread siblings, caches, hypervisor);
* how much parallel work the vCPUs really do: k busy processes at once, and
  a pair on thread siblings against a pair on separate cores;
* time the hypervisor took from the VM (steal) during each run;
* each kernel under several settings of the tiered runtime, with the time
  each LLVM compile landed, so a slow run can be told from a late one.

The JSON it writes holds every run; a summary goes to stdout.
"""
import argparse
import glob
import json
import os
import re
import resource
import shutil
import statistics
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SPEED = os.path.join(HERE, "..", "..", "crates", "zyntax_python", "benchmarks", "speed")
sys.path.insert(0, SPEED)
import run as speed  # noqa: E402

BURN = (
    "import time\n"
    "t = time.perf_counter()\n"
    "x = 0\n"
    "for i in range(6000000):\n"
    "    x += (i * i) & 255\n"
    "print(time.perf_counter() - t)\n"
)


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def steal_ms():
    """Milliseconds the hypervisor has taken from this VM since boot."""
    for line in read("/proc/stat").splitlines():
        if line.startswith("cpu "):
            fields = line.split()
            return int(fields[8]) * 10 if len(fields) > 8 else 0
    return 0


def sibling_groups():
    groups = []
    for path in sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*/topology/thread_siblings_list")):
        text = read(path)
        if text and text not in groups:
            groups.append(text)
    out = []
    for g in groups:
        cpus = []
        for part in g.split(","):
            if "-" in part:
                a, b = part.split("-")
                cpus += list(range(int(a), int(b) + 1))
            else:
                cpus.append(int(part))
        out.append(cpus)
    return out


def describe():
    flags = read("/proc/cpuinfo")
    flag_line = next((l for l in flags.splitlines() if l.startswith("flags")), "")
    wanted = ["avx2", "avx512f", "avx512bw", "avx512vl", "bmi2", "fsrm", "erms", "amx_tile", "sha_ni", "vaes"]
    caches = {}
    for idx in sorted(glob.glob("/sys/devices/system/cpu/cpu0/cache/index*")):
        caches[f"L{read(idx + '/level')} {read(idx + '/type')}"] = {
            "size": read(idx + "/size"),
            "shared_with": read(idx + "/shared_cpu_list"),
        }
    perf = subprocess.run("perf stat -e instructions:u true", shell=True, capture_output=True, text=True)
    return {
        "cpu": next((l.split(":", 1)[1].strip() for l in flags.splitlines() if l.startswith("model name")), ""),
        "vcpus": os.cpu_count(),
        "affinity": len(os.sched_getaffinity(0)),
        "sibling_groups": sibling_groups(),
        "caches_cpu0": caches,
        "flags": [f for f in wanted if f" {f} " in flag_line + " "],
        "dmi": {"vendor": read("/sys/class/dmi/id/sys_vendor"), "product": read("/sys/class/dmi/id/product_name")},
        "kernel": sh("uname -r"),
        "hypervisor": sh("LC_ALL=C lscpu | awk -F: '/Hypervisor vendor/ {gsub(/^ +/,\"\",$2); print $2}'"),
        "governor": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "nominal_mhz": sh("LC_ALL=C lscpu | awk -F: '/CPU MHz/ {gsub(/^ +/,\"\",$2); print $2; exit}'"),
        "pmu_usable": perf.returncode == 0 and "not supported" not in perf.stderr,
        "perf_stderr": perf.stderr.strip()[:200],
    }


def burn_many(sets):
    """Run one busy loop per entry of `sets` (a taskset list, or None) at
    once; each process's own elapsed seconds."""
    procs = []
    for cpus in sets:
        argv = ["python3", "-c", BURN]
        if cpus is not None:
            argv = ["taskset", "-c", cpus] + argv
        procs.append(subprocess.Popen(argv, stdout=subprocess.PIPE, text=True))
    return [float(p.communicate()[0]) for p in procs]


def scaling(describe_):
    """How much slower each of k simultaneous busy processes gets than one
    alone: 1.0 is a core for each, 2.0 a core shared by two."""
    n = describe_["affinity"]
    solo = statistics.median(burn_many([None])[0] for _ in range(3))
    out = {"solo_s": solo, "slowdown": {}, "steal_ms": {}}
    for k in range(2, n + 1):
        before = steal_ms()
        times = [statistics.median(burn_many([None] * k)) for _ in range(3)]
        out["slowdown"][k] = statistics.median(times) / solo
        out["steal_ms"][k] = steal_ms() - before
    return out


def pairs(groups):
    """A pair of thread siblings and a pair on separate cores, as taskset lists."""
    sib = next(([str(g[0]), str(g[1])] for g in groups if len(g) > 1), None)
    cross = [str(groups[0][0]), str(groups[1][0])] if len(groups) > 1 else None
    return sib, cross


def pair_test(groups):
    sib, cross = pairs(groups)
    out = {}
    if sib is None:
        return {"smt": False}
    solo = statistics.median(burn_many([sib[0]])[0] for _ in range(3))
    out["smt"] = True
    out["solo_s"] = solo
    out["siblings_slowdown"] = statistics.median(statistics.median(burn_many(sib)) for _ in range(3)) / solo
    if cross:
        out["separate_cores_slowdown"] = statistics.median(statistics.median(burn_many(cross)) for _ in range(3)) / solo
    return out


LLVM = re.compile(r"\[llvm\] compiled (\S+) in ([\d.]+) ms on (\S+)")


def run_once(argv, cwd, env, trials):
    """One benchmark process: wall, CPU, steal, the iteration times it
    printed and, from the trace, when each LLVM compile finished."""
    steal0 = steal_ms()
    usage0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.perf_counter()
    p = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    events, out = [], []

    def read_err():
        for line in p.stderr:
            events.append((time.perf_counter() - t0, line.rstrip()))

    thread = threading.Thread(target=read_err)
    thread.start()
    for line in p.stdout:
        out.append(line)
    p.wait()
    thread.join()
    wall = (time.perf_counter() - t0) * 1000
    usage1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    times = []
    for line in out:
        try:
            times.append(float(line) * 1000)
        except ValueError:
            pass
    if p.returncode != 0 or len(times) != trials:
        return {"ok": False, "wall_ms": wall}
    llvm = []
    for at, line in events:
        m = LLVM.search(line)
        if m:
            llvm.append({"at_ms": at * 1000, "name": m.group(1), "took_ms": float(m.group(2))})
    return {
        "ok": True,
        "wall_ms": wall,
        "cpu_ms": (usage1.ru_utime + usage1.ru_stime - usage0.ru_utime - usage0.ru_stime) * 1000,
        "steal_ms": steal_ms() - steal0,
        "iterations_ms": times,
        "llvm": llvm,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zypy", required=True)
    ap.add_argument("--pypy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--kernels", default="crypto_pyaes,telco,spectral-norm,ai,richards,deltablue")
    args = ap.parse_args()
    zypy = os.path.abspath(args.zypy)

    d = describe()
    print(f"cpu: {d['cpu']}  vcpus {d['vcpus']}  sibling groups {d['sibling_groups']}")
    print(f"hypervisor: {d['hypervisor']!r}  dmi: {d['dmi']}  pmu usable: {d['pmu_usable']}")
    print(f"flags: {' '.join(d['flags'])}  caches(cpu0): { {k: v['size'] for k, v in d['caches_cpu0'].items()} }")

    scale = scaling(d)
    print("busy processes at once, slowdown of each against one alone (1.0 = a core each):",
          {k: round(v, 2) for k, v in scale["slowdown"].items()}, "steal ms", scale["steal_ms"])
    pair = pair_test(d["sibling_groups"])
    print("thread siblings vs separate cores:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in pair.items()})

    one_per_core = ",".join(str(g[0]) for g in d["sibling_groups"]) if any(len(g) > 1 for g in d["sibling_groups"]) else None
    variants = [
        ("default", {}, None),
        ("no-optimize-wait", {"ZYNTAX_OPTIMIZE_QUIET_MS": "0"}, None),
        ("two-compile-workers", {"ZYNTAX_COMPILE_WORKERS": "2"}, None),
        ("traced", {"ZYNTAX_TRACE_LAZY": "1"}, None),
    ]
    if one_per_core:
        variants.append(("one-vcpu-per-core", {}, one_per_core))
    if args.pypy:
        variants.append(("pypy", None, None))

    rows = []
    for name in args.kernels.split(","):
        kernel, extra, scaling_, files = speed.SUITE[name]
        trials = max(5, int(5 * scaling_))
        stages = {"zypy": speed.stage(kernel, files, True), "pypy": speed.stage(kernel, files, False) if args.pypy else None}
        try:
            for rep in range(args.repeats):
                order = variants[rep % len(variants):] + variants[:rep % len(variants)]
                for vname, env, cpus in order:
                    if vname == "pypy":
                        argv, cwd, e = [args.pypy, os.path.join(stages["pypy"], kernel), "-n", str(trials)] + extra, stages["pypy"], dict(os.environ)
                    else:
                        argv = [zypy, "run", os.path.join(stages["zypy"], kernel), "-n", str(trials)] + extra
                        if cpus:
                            argv = ["taskset", "-c", cpus] + argv
                        cwd, e = stages["zypy"], dict(os.environ, ZYPY_LLVM="1", **env)
                    r = run_once(argv, cwd, e, trials)
                    r.update(kernel=name, variant=vname, rep=rep)
                    rows.append(r)
        finally:
            for s in stages.values():
                if s:
                    shutil.rmtree(s, ignore_errors=True)

    with open(args.out, "w") as f:
        json.dump({"machine": d, "scaling": scale, "pairs": pair, "runs": rows}, f)

    print(f"\n{'kernel':14}{'variant':22}{'steady med':>11}{'min':>8}{'max':>8}{'wall':>7}{'steal':>7}{'llvm n':>7}{'first llvm ms':>14}{'last':>7}")
    for name in args.kernels.split(","):
        for vname, _, _ in variants:
            ok = [r for r in rows if r["kernel"] == name and r["variant"] == vname and r["ok"]]
            if not ok:
                continue
            steady = [statistics.median(r["iterations_ms"][1:]) for r in ok]
            traced = [r for r in ok if r["llvm"]]
            first = statistics.median(min(c["at_ms"] for c in r["llvm"]) for r in traced) if traced else float("nan")
            last = statistics.median(max(c["at_ms"] + 0 for c in r["llvm"]) for r in traced) if traced else float("nan")
            n = statistics.median(len(r["llvm"]) for r in traced) if traced else float("nan")
            print(f"{name:14}{vname:22}{statistics.median(steady):11.2f}{min(steady):8.2f}{max(steady):8.2f}"
                  f"{statistics.median(r['wall_ms'] for r in ok):7.0f}{statistics.median(r['steal_ms'] for r in ok):7.0f}"
                  f"{n:7.0f}{first:14.0f}{last:7.0f}")


if __name__ == "__main__":
    main()
