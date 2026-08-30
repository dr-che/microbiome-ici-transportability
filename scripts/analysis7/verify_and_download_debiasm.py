#!/usr/bin/env python
from __future__ import annotations
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

VERSION = "0.0.2"
EXPECTED_SHA256 = "51b194d88f265af06e3c36d780dbe6958d51f6a5367c1a0f555fa4ec3c354d1a"

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "packages"
    out_dir.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(f"https://pypi.org/pypi/DEBIAS-M/{VERSION}/json", timeout=60) as response:
        meta = json.load(response)

    files = [u for u in meta["urls"] if u.get("packagetype") == "sdist"]
    if len(files) != 1:
        raise RuntimeError(f"Expected one sdist, found {len(files)}")
    item = files[0]

    if item["digests"]["sha256"] != EXPECTED_SHA256:
        raise RuntimeError("PyPI JSON SHA-256 differs from the frozen expected value")

    target = out_dir / item["filename"]
    urllib.request.urlretrieve(item["url"], target)

    observed = hashlib.sha256(target.read_bytes()).hexdigest()
    if observed != EXPECTED_SHA256:
        raise RuntimeError(f"Downloaded hash mismatch: {observed}")

    status = {
        "status": "PASS",
        "version": VERSION,
        "filename": target.name,
        "url": item["url"],
        "sha256": observed,
        "size_bytes": target.stat().st_size,
    }
    (out_dir / "DEBIASM_download_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, indent=2))

if __name__ == "__main__":
    main()
