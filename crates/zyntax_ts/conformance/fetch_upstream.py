#!/usr/bin/env python3
"""Fetch and verify the pinned official TypeScript conformance sources."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import tempfile
import urllib.request

PIN = Path(__file__).with_name("upstream.json")


def inventory(root):
    digest = hashlib.sha256()
    paths = sorted(p for p in root.rglob("*") if p.is_file())
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {"files": len(paths), "inventory_sha256": digest.hexdigest()}


def verify(root, pin):
    found = inventory(root)
    for key, value in found.items():
        if value != pin[key]:
            raise ValueError(f"official corpus {key}: expected {pin[key]}, found {value}")
    return {**pin, "verified": True}


def extract(archive, destination, pin):
    prefix = PurePosixPath(f"TypeScript-{pin['revision']}") / pin["directory"]
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            path = PurePosixPath(member.name)
            if not path.is_relative_to(prefix) or not member.isfile():
                continue
            relative = path.relative_to(prefix)
            if ".." in relative.parts:
                raise ValueError(f"unsafe archive path: {path}")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source:
                target.write_bytes(source.read())
    verify(destination, pin)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="empty directory for corpus sources")
    parser.add_argument("--archive", type=Path, help="reuse a downloaded archive")
    args = parser.parse_args()
    pin = json.loads(PIN.read_text())
    if args.destination.exists() and any(args.destination.iterdir()):
        verify(args.destination, pin)
    else:
        args.destination.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="zynode-corpus-") as directory:
            archive = args.archive or Path(directory) / "typescript.tar.gz"
            if not args.archive:
                url = f"https://codeload.github.com/microsoft/TypeScript/tar.gz/{pin['revision']}"
                urllib.request.urlretrieve(url, archive)
            if hashlib.sha256(archive.read_bytes()).hexdigest() != pin["archive_sha256"]:
                raise ValueError("official corpus archive checksum mismatch")
            extract(archive, args.destination, pin)
    print(json.dumps(verify(args.destination, pin), indent=2))


if __name__ == "__main__":
    main()
