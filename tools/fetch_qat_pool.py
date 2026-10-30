"""Fetch a fine-tuning pool for QAT, and record exactly what it contains.

QAT does not need the pretraining corpus: it needs a few hundred representative natural images.
This fetches a public pool that is disjoint from every evaluation set used in this project
(Kodak, CLIC Professional, Set14), so no test image can leak into fine-tuning.

    python tools/fetch_qat_pool.py --source div2k_valid
    python tools/fetch_qat_pool.py --source div2k_train --limit 200

Sources are recorded with URL, byte size and per-file SHA-256 in `manifest.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import ssl
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120"}

SOURCES = {
    "div2k_valid": {
        "url": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
        "note": "DIV2K validation split, 100 images, 2040x1356 class. Used here only as a QAT pool; "
                "it is never used for evaluation in this project.",
        "expect": 100,
    },
    "div2k_train": {
        "url": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
        "note": "DIV2K training split, 800 images. Large: the host serves it slowly.",
        "expect": 800,
    },
}

EVAL_SETS = {"kodak", "clic_professional", "set14", "kodak_thumb"}


def download(url: str, target: Path) -> None:
    context = ssl._create_unverified_context()
    request = urllib.request.Request(url, headers=UA)
    target.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=120, context=context) as response:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        with target.open("wb") as stream:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
                done += len(chunk)
                if total and done % (25 * 1024 * 1024) < 1024 * 1024:
                    elapsed = time.perf_counter() - start
                    print(f"  {done/2**20:.0f}/{total/2**20:.0f} MB "
                          f"({done/2**20/max(elapsed,1e-9):.2f} MB/s)", flush=True)
    print(f"  downloaded {target.stat().st_size/2**20:.1f} MB in {time.perf_counter()-start:.0f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=sorted(SOURCES), default="div2k_valid")
    parser.add_argument("--dest", default="datasets/qat_pool")
    parser.add_argument("--limit", type=int, default=0, help="keep at most N images")
    parser.add_argument("--keep-archive", action="store_true")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    spec = SOURCES[args.source]
    dest = Path(args.dest) / args.source
    if args.source in EVAL_SETS:
        raise SystemExit("refusing to build a QAT pool from an evaluation set")
    dest.mkdir(parents=True, exist_ok=True)
    archive = Path(args.dest) / f"{args.source}.zip"

    if not archive.is_file():
        print(f"fetching {spec['url']}")
        download(spec["url"], archive)
    else:
        print(f"archive present: {archive}")

    extracted = 0
    with zipfile.ZipFile(archive) as bundle:
        names = [n for n in bundle.namelist() if n.lower().endswith((".png", ".jpg", ".jpeg"))]
        names.sort()
        if args.limit:
            names = names[: args.limit]
        for name in names:
            target = dest / Path(name).name
            if target.is_file():
                extracted += 1
                continue
            with bundle.open(name) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)
            extracted += 1
    print(f"images in pool: {extracted}")

    files = []
    for path in sorted(dest.glob("*")):
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append({"file": path.name, "bytes": path.stat().st_size, "sha256": digest})
    manifest = {
        "source": args.source,
        "url": spec["url"],
        "note": spec["note"],
        "used_for": "QAT fine-tuning pool. Disjoint from kodak, clic_professional and set14.",
        "eval_sets_excluded": sorted(EVAL_SETS),
        "count": len(files),
        "total_bytes": sum(f["bytes"] for f in files),
        "files": files,
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                        encoding="utf-8")
    if not args.keep_archive:
        archive.unlink(missing_ok=True)
    print(json.dumps({"pool": str(dest), "images": len(files),
                      "total_mb": round(manifest["total_bytes"] / 2**20, 1)}))


if __name__ == "__main__":
    main()
