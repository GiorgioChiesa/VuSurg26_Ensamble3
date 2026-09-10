#!/usr/bin/env python3
"""Download and unpack SAR-RARP50 from its figshare record.

WHY FIGSHARE AND NOT SYNAPSE
    The challenge originally pointed at Synapse, which needs an account and a
    signed terms-of-use. The published dataset lives on UCL's figshare record
    and is openly downloadable: article 24932529 is the train set (45 files,
    26.95 GB), 24932499 the test set (11 files, 6.61 GB).

LICENCE
    CC BY-NC-SA 4.0 — non-commercial, share-alike. That is a constraint on what
    a model trained on it may be used for; it is recorded in the manifest this
    script writes so the provenance travels with the data.

WHAT IT GIVES US
    Per-frame segmentation where one class is the instrument **wrist** — the same
    anatomy SurgVU26 cat1 calls the clevis. The class index is NOT assumed here:
    `scripts/convert_sar_rarp50_wrist_to_yolo.py` verifies the palette against
    the data before converting anything.

Usage:
    python scripts/download_sar_rarp50.py --output-dir data/sar_rarp50
    python scripts/download_sar_rarp50.py --output-dir data/sar_rarp50 --list
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

ARTICLES = {"train": 24932529, "test": 24932499}
API = "https://api.figshare.com/v2/articles/{article_id}"
UA = {"User-Agent": "Mozilla/5.0"}


def _article(article_id: int) -> dict:
    request = urllib.request.Request(API.format(article_id=article_id), headers=UA)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode())


def _download(url: str, destination: Path, expected_md5: str | None) -> bool:
    """Fetch one file, skipping it when a correct copy is already on disk."""
    if destination.exists() and expected_md5:
        digest = hashlib.md5()
        with destination.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() == expected_md5:
            print(f"  = {destination.name} (già presente, md5 ok)", flush=True)
            return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(request, timeout=300) as response, temporary.open("wb") as handle:
        while chunk := response.read(1 << 20):
            handle.write(chunk)
    temporary.replace(destination)
    print(f"  + {destination.name} ({destination.stat().st_size / 1e9:.2f} GB)", flush=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", choices=sorted(ARTICLES) + ["all"], default="train")
    parser.add_argument("--list", action="store_true", help="Print the file list and exit.")
    parser.add_argument("--no-unzip", action="store_true")
    args = parser.parse_args()

    splits = sorted(ARTICLES) if args.split == "all" else [args.split]
    manifest: dict[str, object] = {
        "format": "clevis.sar-rarp50-download",
        "version": 1,
        "source": "figshare",
        "licence": "CC BY-NC-SA 4.0",
        "splits": {},
    }

    for split in splits:
        article = _article(ARTICLES[split])
        files = article.get("files") or []
        total = sum(f.get("size", 0) for f in files)
        print(f"== {split}: {article.get('title')} — {len(files)} file, {total / 1e9:.2f} GB")
        manifest["splits"][split] = {
            "article_id": ARTICLES[split],
            "title": article.get("title"),
            "files": [{"name": f["name"], "size": f["size"], "md5": f.get("computed_md5")} for f in files],
        }
        if args.list:
            for entry in files:
                print(f"  {entry['name']:24} {entry['size'] / 1e9:6.2f} GB")
            continue

        zip_dir = args.output_dir / "_zips" / split
        unpacked = args.output_dir / split
        for entry in files:
            destination = zip_dir / entry["name"]
            _download(entry["download_url"], destination, entry.get("computed_md5"))
            if args.no_unzip or not zipfile.is_zipfile(destination):
                continue
            # Named after the archive so two videos never merge into one tree.
            target = unpacked / Path(entry["name"]).stem
            if target.exists():
                continue
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(destination) as archive:
                archive.extractall(target)
            print(f"    -> scompattato in {target}", flush=True)

    if not args.list:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("fatto")


if __name__ == "__main__":
    main()
