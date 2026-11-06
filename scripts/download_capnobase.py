"""Download the CapnoBase IEEE TBME Respiratory Rate Benchmark (42 cases, 8 min) from Borealis.

doi:10.5683/SP2/NLB8IT. Downloading means accepting the dataset's terms of use (no re-identification
of subjects; cite Karlen et al., IEEE TBME 2013). The benchmark is meant for testing, not training.
Fetches the signal, labels, reference, parameter and meta tables in their original CSV form.
Usage:  python scripts/download_capnobase.py [--out data/raw/capnobase]
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

API = "https://borealisdata.ca/api"
DOI = "doi:10.5683/SP2/NLB8IT"
WANTED = ("_signal.tab", "_labels.tab", "_reference.tab", "_param.tab", "_meta.tab")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/raw/capnobase"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(f"{API}/datasets/:persistentId/?persistentId={DOI}") as r:
        version = json.load(r)["data"]["latestVersion"]
    (args.out / "TERMS_OF_USE.txt").write_text(version.get("termsOfUse", ""), encoding="utf-8")

    files = [f["dataFile"] for f in version["files"]
             if f["dataFile"]["filename"].endswith(WANTED) or f["dataFile"]["filename"].endswith(".txt")]
    for i, f in enumerate(sorted(files, key=lambda x: x["filename"]), 1):
        name = f["filename"].replace(".tab", ".csv")
        dest = args.out / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        with urllib.request.urlopen(f"{API}/access/datafile/{f['id']}?format=original") as r:
            dest.write_bytes(r.read())
        if i % 20 == 0:
            print(f"{i}/{len(files)} files")
    print(f"done: {len(list(args.out.glob('*_signal.csv')))} recordings in {args.out}")


if __name__ == "__main__":
    main()
