"""Per-case regression gate for a complete, verified official corpus report."""
import hashlib
import json


def state(row):
    status = row["status"]
    check = row.get("native_check")
    if check and (check["code"] is None or check["code"] < 0):
        raise ValueError(f"native crash/timeout: {row['case']}")
    if status in ("PASS", "UNSUPPORTED", "SKIP"):
        return status
    if status != "FAIL" or not check:
        raise ValueError(f"unexpected result {status}: {row['case']}")
    if check["code"] == 0:
        return "FAIL:accept"
    if check["code"] != 1:
        raise ValueError(f"native process failure: {row['case']}")
    for line in reversed(check["stderr"].splitlines()):
        try:
            diagnostic = json.loads(line)
        except ValueError:
            continue
        if isinstance(diagnostic, dict) and diagnostic.get("kind") in ("Syntax", "Type"):
            if row["reference"]["accepted"]:
                return "FAIL:reject"
            return "FAIL:" + diagnostic["kind"]
    raise ValueError(f"missing native diagnostic: {row['case']}")


def snapshot(report):
    corpus = report.get("corpus", {})
    size = corpus.get("files", 0)
    if not corpus.get("verified") or not size or any(
        corpus.get(key) != size for key in ("source_files", "selected_files", "completed_files")
    ):
        raise ValueError("regression gate requires a complete verified corpus")
    if report.get("profiles") != ["baseline"] or not report.get("live_reference"):
        raise ValueError("regression gate requires one baseline checker profile and a live reference")
    cases, specifications = {}, []
    for row in report["results"]:
        name = row["case"]
        if name in cases:
            raise ValueError(f"duplicate official case: {name}")
        cases[name] = state(row)
        reference = row.get("reference", {})
        specifications.append([name, row.get("corpus_source_sha256", row.get("source_sha256")),
                               reference.get("options", row.get("options")),
                               reference.get("accepted"), reference.get("kind")])
    if len({name.rsplit("#", 1)[0] for name in cases}) != size:
        raise ValueError("official source coverage is incomplete")
    if not any(value == "PASS" for value in cases.values()):
        raise ValueError("official corpus has no passing comparisons")
    provenance = {key: corpus[key] for key in (
        "repository", "revision", "compiler", "inventory_sha256", "files", "defaults", "mode")}
    provenance["reference_compiler"] = report["reference_compiler"]
    digest = hashlib.sha256(json.dumps(sorted(specifications), sort_keys=True).encode()).hexdigest()
    return {"version": 1, "corpus": provenance, "comparisons_sha256": digest,
            "cases": dict(sorted(cases.items()))}


def regressions(report, baseline):
    current = snapshot(report)
    for key in ("version", "corpus", "comparisons_sha256"):
        if current[key] != baseline.get(key):
            raise ValueError(f"official regression baseline {key} differs; review corpus/options coverage")
    if current["cases"].keys() != baseline["cases"].keys():
        raise ValueError("official regression baseline case inventory differs")
    return [f"{case}: {previous} -> {current['cases'][case]}"
            for case, previous in baseline["cases"].items()
            if current["cases"][case] != previous and current["cases"][case] != "PASS"]


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--write-baseline", required=True, type=Path,
                        help="save reviewed outcomes; never invoked by CI")
    args = parser.parse_args()
    args.write_baseline.write_text(json.dumps(snapshot(json.loads(args.report.read_text())), indent=2) + "\n")
