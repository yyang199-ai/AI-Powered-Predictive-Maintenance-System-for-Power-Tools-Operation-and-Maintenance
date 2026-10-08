#!/usr/bin/env python
"""Download a pinned complete IMS run 2 mirror, validate and prepare features."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from predictive.ims_data import (IMS_ARCHIVE_URL, IMS_MIRROR, IMS_MIRROR_COMMIT,
                                IMS_RUN2_COUNT, snapshot_timestamp, prepare_ims_run2)


def download_run2(raw_dir: Path):
    """Select only expected timestamp files; never extract arbitrary ZIP paths."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ims-run2-") as temporary:
        archive = Path(temporary) / "run2.zip"
        request = urllib.request.Request(IMS_ARCHIVE_URL, headers={"User-Agent": "predictive-maintenance-research/1.0"})
        print(f"Downloading pinned IMS mirror ({IMS_MIRROR_COMMIT}); approximately 133 MB compressed...", flush=True)
        with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
        count = 0
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist():
                parts = member.filename.split("/")
                if len(parts) != 3 or parts[1] != "data" or member.is_dir():
                    continue
                try:
                    snapshot_timestamp(parts[-1])
                except ValueError:
                    continue
                if member.file_size > 2_000_000:
                    raise ValueError("unexpectedly large IMS snapshot in archive")
                (raw_dir / parts[-1]).write_bytes(zipped.read(member))
                count += 1
        if count != IMS_RUN2_COUNT:
            raise ValueError(f"pinned mirror archive has {count} snapshots, expected {IMS_RUN2_COUNT}")
        provenance = {"repository": IMS_MIRROR, "commit": IMS_MIRROR_COMMIT,
                      "download_url": IMS_ARCHIVE_URL,
                      "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                      "archive_bytes": archive.stat().st_size,
                      "download_method": "pinned_third_party_github_archive",
                      "official_source_verified_this_session": False,
                      "license_declared_by_mirror": None}
        (raw_dir / "download_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Downloaded {count} snapshots; approximately 524 MB uncompressed.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="download a pinned mirror of complete run 2")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/public/NASA_IMS/raw/run2")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/public/NASA_IMS/prepared")
    parser.add_argument("--allow-partial", action="store_true", help="exploratory partial import; never assigns endpoint proxy RUL")
    args = parser.parse_args()
    if args.download:
        download_run2(args.raw_dir)
    manifest = prepare_ims_run2(args.raw_dir, args.output_dir, require_complete=not args.allow_partial)
    print(json.dumps({key: manifest[key] for key in ["download_status", "snapshot_count", "feature_rows", "unique_failure_events"]}, ensure_ascii=False, indent=2))
    print(f"Prepared dataset: {args.output_dir}")
    print("Endpoint countdown is a retrospective proxy; no RUL model was trained or validated.")


if __name__ == "__main__":
    main()
