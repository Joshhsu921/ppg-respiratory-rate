"""Generate the QR code that opens the live demo (web/qr.svg).

Usage:  python scripts/make_qr.py [--url https://...]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import segno

DEMO_URL = "https://joshhsu921.github.io/ppg-respiratory-rate/"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEMO_URL)
    ap.add_argument("--out", type=Path, default=Path("web/qr.svg"))
    args = ap.parse_args()
    qr = segno.make(args.url, error="m")
    qr.save(args.out, kind="svg", scale=8, border=2, dark="#13211f", light="#ffffff",
            xmldecl=False, svgclass=None, lineclass=None)
    print(f"{args.url} -> {args.out}")


if __name__ == "__main__":
    main()
