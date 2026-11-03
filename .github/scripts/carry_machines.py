#!/usr/bin/env python3
"""Keep one result file per CI machine on the published site.

Every run of the benchmark workflow lands on whichever machines the host
hands out, so a single run measures only some of them. This keeps the
last result of each machine from earlier deploys and adds this run's:

    carry_machines.py --base-url URL --suite zylua --site site/zylua RESULTS.json...

Under `<site>/machines/` it writes one `<key>.json` per machine, an
`index.json` that lists them, and a `history.json` with one line of
summary for every run each machine has ever had (how many rows, the
geometric-mean speedup over the baseline interpreter, and the rows won,
by benchmark time and by full process time), which the pages average
into an all-time figure per machine. Earlier machines are read from
`<base-url>/<suite>/machines/`,
where the previous deploy put them. A missing index (the first deploy)
starts the history; any other failure to read it stops the script, so a
network error cannot publish a page that has lost every machine but this
run's.
"""
import argparse
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request


def fetch(url):
    """The body at `url`, or None when nothing is published there.

    A published site is cached for minutes, and a deploy that follows
    another closely would read the index from before it, so a query the
    cache has not seen goes with every request. A failure other than a
    missing file is tried again, then raised.
    """
    if url.startswith(("http://", "https://")):
        url += ("&" if "?" in url else "?") + f"fresh={time.time_ns()}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt == 3:
                raise
        except urllib.error.URLError as e:
            # A `file://` base, as the tests use, reports a missing file so.
            if isinstance(e.reason, FileNotFoundError):
                return None
            if attempt == 3:
                raise
        time.sleep(2 * (attempt + 1))


def machine_key(meta):
    """A file-name-safe identity for the machine a result came from."""
    cpu = meta.get("cpu") or meta.get("arch") or "unknown"
    cpu = re.sub(r"\((?:R|TM|tm|r)\)", "", cpu)
    cpu = re.sub(r"\b\d+-Core\b|\bProcessor\b|\bCPU\b|@.*$", "", cpu, flags=re.I)
    key = f"{meta.get('os') or 'unknown'} {cpu}".lower()
    return re.sub(r"[^a-z0-9]+", "-", key).strip("-")


# What a suite's result file holds: the rows, our interpreter, the baseline
# it is judged against, and the warm and cold time of each.
SUITES = {
    "zylua": {"rows": "results", "ours": "zylua", "base": "luajit", "warm": "elapsed", "cold": "wall"},
    "zypy": {"rows": "kernels", "ours": "zypy", "base": "pypy", "warm": "mean_ms", "cold": "wall_ms"},
}


def time_of(suite, row, field):
    """Milliseconds for one interpreter's row, or None when it has none."""
    if not row or row.get("error") or (suite == "zypy" and row.get("status") != "ok"):
        return None
    value = row.get(field)
    if isinstance(value, dict):
        value = value.get("median")
        value = value * 1000 if isinstance(value, (int, float)) else None
    return value if isinstance(value, (int, float)) and value > 0 else None


def summarize(suite, data, key, meta):
    """One run's standing against the baseline: the geometric mean of
    baseline time over our time across the rows both finished (above 1
    is faster), and how many rows we won, by warm and by cold time."""
    spec = SUITES[suite]
    out = {
        "key": key,
        "os": meta.get("os"),
        "cpu": meta.get("cpu"),
        "date": meta.get("date"),
        "commit": meta.get("commit"),
    }
    for name in ("warm", "cold"):
        ratios = []
        for row in (data.get(spec["rows"]) or {}).values():
            ours = time_of(suite, row.get(spec["ours"]), spec[name])
            base = time_of(suite, row.get(spec["base"]), spec[name])
            if ours and base:
                ratios.append(base / ours)
        out[name] = math.exp(sum(map(math.log, ratios)) / len(ratios)) if ratios else None
        out["wins_" + name] = sum(1 for r in ratios if r >= 1)
        out["n"] = len(ratios)
    return out


def entry_for(key, meta):
    return {
        "key": key,
        "file": f"{key}.json",
        "os": meta.get("os"),
        "arch": meta.get("arch"),
        "cpu": meta.get("cpu"),
        "cores": meta.get("cpu_cores"),
        "ram_gb": meta.get("ram_gb"),
        "commit": meta.get("commit"),
        "date": meta.get("date"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="where the site is published")
    ap.add_argument("--suite", required=True, help="the language page's directory")
    ap.add_argument("--site", required=True, help="the suite's directory in the site being built")
    ap.add_argument("--seed", help="history to start from when none is published yet")
    ap.add_argument("results", nargs="*", help="this run's result files")
    args = ap.parse_args()

    import os

    base = f"{args.base_url.rstrip('/')}/{args.suite}/machines"
    out = os.path.join(args.site, "machines")
    os.makedirs(out, exist_ok=True)

    kept = {}
    body = fetch(f"{base}/index.json")
    if body is None:
        print(f"{args.suite}: no earlier machines published")
    else:
        for entry in json.loads(body).get("machines", []):
            data = fetch(f"{base}/{entry['file']}")
            if data is None:
                sys.exit(f"{args.suite}: the index lists {entry['file']} but it is not published")
            with open(os.path.join(out, entry["file"]), "wb") as f:
                f.write(data)
            kept[entry["key"]] = entry
        print(f"{args.suite}: kept {len(kept)} earlier machines")

    body = fetch(f"{base}/history.json")
    if body is None and args.seed:
        with open(args.seed, "rb") as f:
            body = f.read()
        print(f"{args.suite}: no history published; starting from {args.seed}")
    runs = json.loads(body).get("runs", []) if body else []
    seen = {(r["key"], r["date"]) for r in runs}

    for path in args.results:
        with open(path, "rb") as f:
            raw = f.read()
        data = json.loads(raw)
        meta = data.get("meta") or {}
        key = machine_key(meta)
        with open(os.path.join(out, f"{key}.json"), "wb") as f:
            f.write(raw)
        kept[key] = entry_for(key, meta)
        if (key, meta.get("date")) not in seen:
            runs.append(summarize(args.suite, data, key, meta))
        print(f"{args.suite}: {key} from {meta.get('commit')} on {meta.get('date')}")

    machines = sorted(kept.values(), key=lambda e: (e["os"] != "linux", e["os"] or "", e["cpu"] or ""))
    with open(os.path.join(out, "index.json"), "w") as f:
        json.dump({"machines": machines}, f, indent=1)
        f.write("\n")
    runs.sort(key=lambda r: (r["date"] or "", r["key"]))
    with open(os.path.join(out, "history.json"), "w") as f:
        json.dump({"runs": runs}, f, separators=(",", ":"))
        f.write("\n")


if __name__ == "__main__":
    main()
