"""Fetch a self-contained TeX engine (Tectonic) so the report can be built from LaTeX.

Tectonic is a single executable that downloads the packages a document needs on first run, so it
does not require a multi-gigabyte TeX distribution.

    python tools/fetch_tectonic.py
"""

from __future__ import annotations

import json
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "tools" / "tectonic"
API = "https://api.github.com/repos/tectonic-typesetting/tectonic/releases/latest"
UA = {"User-Agent": "Mozilla/5.0", "Accept": "application/vnd.github+json"}


def fetch(url: str, destination: Path) -> int:
    context = ssl._create_unverified_context()
    request = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(request, timeout=180, context=context) as response, \
            destination.open("wb") as stream:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            stream.write(chunk)
            done += len(chunk)
            if total and done % (5 << 20) < (1 << 20):
                print(f"  {done/2**20:.0f}/{total/2**20:.0f} MB", flush=True)
    return destination.stat().st_size


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    TARGET.mkdir(parents=True, exist_ok=True)
    existing = list(TARGET.glob("tectonic*.exe"))
    if existing:
        print(json.dumps({"status": "present", "binary": str(existing[0])}))
        return

    context = ssl._create_unverified_context()
    request = urllib.request.Request(API, headers=UA)
    with urllib.request.urlopen(request, timeout=60, context=context) as response:
        release = json.loads(response.read().decode("utf-8"))

    asset = None
    for candidate in release.get("assets", []):
        name = candidate["name"].lower()
        if "x86_64-pc-windows" in name and name.endswith(".zip"):
            asset = candidate
            break
    if asset is None:
        names = [a["name"] for a in release.get("assets", [])]
        raise SystemExit(f"no Windows x86_64 zip in release assets: {names}")

    print(f"downloading {asset['name']} ({asset['size']/2**20:.0f} MB)")
    archive = TARGET / asset["name"]
    size = fetch(asset["browser_download_url"], archive)
    print(f"  {size/2**20:.1f} MB downloaded")

    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            if member.lower().endswith(".exe"):
                bundle.extract(member, TARGET)
                extracted = TARGET / member
                binary = TARGET / "tectonic.exe"
                if extracted != binary:
                    extracted.replace(binary)
                print(json.dumps({"status": "installed", "binary": str(binary),
                                  "version": release.get("tag_name")}))
                break
    archive.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
