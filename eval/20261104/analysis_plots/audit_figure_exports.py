#!/usr/bin/env python3
"""Audit physical sizes, fonts, vector text, and PNG DPI of paper figures.

Run with the bundled Codex Python, which includes pypdf, pdfplumber, Pillow.
This checks the exported files themselves, not only their source figsize.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pdfplumber
from PIL import Image
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
WIDTH_HEIGHT_CM = ((8.6, 5.2),)


def close(actual: float, expected: float, tolerance: float = 0.04) -> bool:
    return abs(actual - expected) <= tolerance


def audit(pdf: Path) -> dict:
    base = pdf.with_suffix("")
    svg = base.with_suffix(".svg")
    png = base.with_suffix(".png")
    reader = PdfReader(str(pdf))
    assert len(reader.pages) == 1
    page = reader.pages[0]
    width_cm = float(page.mediabox.width) * 2.54 / 72
    height_cm = float(page.mediabox.height) * 2.54 / 72
    match = next(((w, h) for w, h in WIDTH_HEIGHT_CM
                  if close(width_cm, w) and close(height_cm, h)), None)
    with pdfplumber.open(pdf) as document:
        chars = document.pages[0].chars
        actual_fonts = sorted({str(char["fontname"]) for char in chars})
        sizes = sorted({round(float(char["size"]), 1) for char in chars})
        # pdfplumber reports glyph advance as `size` for rotated y labels;
        # their physical font size is the character box width instead.
        axis_10pt = 10.0 in sizes or any(
            not char["upright"] and abs(float(char["width"]) - 10.0) < 0.05
            for char in chars
        )
        text_inside_page = all(
            float(char["x0"]) >= -0.3 and float(char["x1"]) <= float(page.mediabox.width) + 0.3
            and float(char["top"]) >= -0.3 and float(char["bottom"]) <= float(page.mediabox.height) + 0.3
            for char in chars
        )
    content = page.get_contents().get_data().decode("latin1")
    if png.is_file():
        with Image.open(png) as image:
            dpi = image.info.get("dpi")
            png_dpi = [float(dpi[0]), float(dpi[1])] if dpi else None
            pixels = [image.width, image.height]
    else:
        png_dpi, pixels = None, None
    svg_text = svg.is_file() and "<text" in svg.read_text(errors="ignore")
    checks = {
        "physical_size": match is not None,
        "pdf_times_new_roman": bool(actual_fonts) and all("TimesNewRoman" in font for font in actual_fonts),
        "font_10_pt_present": axis_10pt,
        "font_9_pt_present": 9.0 in sizes,
        "axis_width_0_8_pt_present": "0.8 w" in content,
        "text_not_clipped": text_inside_page,
        "svg_present_editable_text": svg_text,
        "png_600_dpi": png_dpi is not None and all(abs(value - 600) < 1 for value in png_dpi),
    }
    return {
        "Figure": base.name,
        "Intended width/height (cm)": "" if match is None else f"{match[0]:.1f} x {match[1]:.1f}",
        "Actual PDF width/height (cm)": f"{width_cm:.3f} x {height_cm:.3f}",
        "Axis label font": "10 pt" if axis_10pt else "CHECK",
        "Tick font": "9 pt" if 9.0 in sizes else "CHECK",
        "Legend font": "8.5 pt present" if 8.5 in sizes else "N/A or CHECK",
        "Line width": "1.4 pt present" if "1.4 w" in content else "N/A (bar-only) or CHECK",
        "Output formats": "/".join(fmt for fmt, present in (("PDF", True), ("SVG", svg.is_file()),
                                                       ("PNG", png.is_file())) if present),
        "PASS/FAIL": "PASS" if all(checks.values()) else "FAIL",
        "Notes": "; ".join(key for key, value in checks.items() if not value),
        "PDF path": str(pdf.resolve()),
        "PNG pixels": pixels,
        "PNG DPI": png_dpi,
        "PDF fonts": actual_fonts,
        "Observed text sizes": sizes,
        "Checks": checks,
    }


def main() -> None:
    paths = sorted((ROOT / "main").glob("*.pdf")) + sorted((ROOT / "supplementary").glob("*.pdf"))
    if len(paths) != 33 or len(list((ROOT / "main").glob("*.pdf"))) != 9:
        raise RuntimeError("Expected 9 main and 24 supplementary single-graph PDFs")
    rows = [audit(path) for path in paths]
    out = ROOT / "figure_size_font_audit.csv"
    fields = ("Figure", "Intended width/height (cm)", "Actual PDF width/height (cm)",
              "Axis label font", "Tick font", "Legend font", "Line width",
              "Output formats", "PASS/FAIL", "Notes")
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (ROOT / "verification/figure_export_audit.json").write_text(json.dumps(rows, indent=2) + "\n")
    failures = [row["Figure"] + ": " + row["Notes"] for row in rows if row["PASS/FAIL"] != "PASS"]
    print(f"Audited {len(rows)} PDF/SVG/PNG figure sets; failures: {len(failures)}")
    for failure in failures:
        print(failure)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
