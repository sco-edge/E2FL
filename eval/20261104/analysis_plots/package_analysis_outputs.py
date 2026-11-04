#!/usr/bin/env python3
"""Package revised paper figures and reproducibility records without old archives."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "loracast_www_figures_5-8_revised.zip"
INVENTORY = ROOT / "revision_archive_inventory.json"
PREFIX = Path("loracast_www_figures_5-8_revised")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return (path.is_file() and path.suffix not in (".zip", ".pyc")
            and "__pycache__" not in relative.parts
            and relative != INVENTORY.relative_to(ROOT))


def main() -> None:
    for area, count in (("main", 9), ("supplementary", 24)):
        for extension in ("pdf", "svg", "png"):
            files = list((ROOT / area).glob(f"*.{extension}"))
            if len(files) != count:
                raise AssertionError(f"{area}: expected {count} {extension}, found {len(files)}")
    paths = sorted((path for path in ROOT.rglob("*") if included(path)),
                   key=lambda path: str(path.relative_to(ROOT)))
    INVENTORY.write_text(json.dumps({
        "scope": "revised nine main figure sets, unchanged 24 supplementary sets, scripts, values and audits",
        "files": [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in paths],
    }, indent=2) + "\n")
    paths.append(INVENTORY)
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as zipped:
        for path in paths:
            zipped.write(path, str(PREFIX / path.relative_to(ROOT)))
    with zipfile.ZipFile(ARCHIVE) as zipped:
        if zipped.testzip() is not None:
            raise AssertionError("Archive CRC check failed")
        if len(zipped.namelist()) != len(paths):
            raise AssertionError("Archive file count differs from source inventory")
    print(f"Packaged {len(paths)} files: {ARCHIVE} ({ARCHIVE.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
