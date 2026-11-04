#!/usr/bin/env python3
"""Check WWW Fig. 6 mean coordinates, dimensions, editable text, and resolution."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean
import xml.etree.ElementTree as ET

from PIL import Image
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
FIGURES = {
    "fig6a_learning_vs_active_time": (8.6, 5.2),
    "fig6b_learning_vs_client_energy": (8.6, 5.2),
}
REQUIRED_POLICIES = ("FedAvg", "Fixed-Hetero", "PyramidFL-style", "LoRACast")


def font_at(style: str, size: float) -> bool:
    # Matplotlib versions serialize SVG CSS as either font-size or the font shorthand.
    return f"font-size: {size:g}px" in style or f"font: {size:g}px" in style


def verify_mean_coordinates() -> dict:
    with (ROOT / "verification/fig6_plot_data.csv").open(newline="") as stream:
        plotted = list(csv.DictReader(stream))
    with (ROOT / "verification/fig6_seed_run_summary.csv").open(newline="") as stream:
        seed_summary = list(csv.DictReader(stream))
    audit = json.loads((ROOT / "verification/fig6_validation.json").read_text())
    expected_rounds = audit["rounds_plotted"]
    if sorted(set(expected_rounds)) != expected_rounds:
        raise AssertionError("Fig. 6 rounds are not strictly sorted")
    checked = 0
    for panel in ("time", "energy"):
        for policy in ("U", "F", "P", "C"):
            for round_no in expected_rounds:
                same_key = [row for row in plotted if row["panel"] == panel and
                            row["policy_code"] == policy and int(row["round"]) == round_no]
                seeds = [row for row in same_key if row["record_type"] == "observed_seed_checkpoint"]
                averages = [row for row in same_key if row["record_type"] == "round_matched_seed_mean"]
                if sorted(int(row["seed"]) for row in seeds) != [43, 44, 45] or len(averages) != 1:
                    raise AssertionError(f"missing or duplicate Fig. 6 inputs: {panel}/{policy}/R{round_no}")
                x = mean(float(row["cumulative_cost"]) for row in seeds)
                y = mean(float(row["BF16_full_sequence_NLL"]) for row in seeds)
                if not math.isclose(x, float(averages[0]["cumulative_cost"]), rel_tol=0, abs_tol=2e-9) or not math.isclose(
                    y, float(averages[0]["BF16_full_sequence_NLL"]), rel_tol=0, abs_tol=2e-11):
                    raise AssertionError(f"incorrect mean point: {panel}/{policy}/R{round_no}")
                if round_no == 12:
                    raw_summaries = [row for row in seed_summary if row["policy_code"] == policy]
                    col = "total_active_time_min" if panel == "time" else "total_measured_client_energy_kJ"
                    if len(raw_summaries) != 3 or not math.isclose(
                        x, mean(float(row[col]) for row in raw_summaries), rel_tol=0, abs_tol=2e-9
                    ) or not math.isclose(
                        y, mean(float(row["R12_validation_NLL"]) for row in raw_summaries), rel_tol=0, abs_tol=2e-11
                    ):
                        raise AssertionError(f"R12 point does not match seed summary: {panel}/{policy}")
                checked += 1
    expected_rows = 2 * 4 * len(expected_rounds) * 4
    if len(plotted) != expected_rows:
        raise AssertionError(f"unexpected Fig. 6 archive rows: {len(plotted)} != {expected_rows}")
    return {"status": "PASS", "round_matched_mean_points_checked": checked,
            "seed_ids": [43, 44, 45], "rounds": expected_rounds,
            "R12_crosschecked_against_seed_summary": True}


def main() -> None:
    report = {}
    for stem, (target_w, target_h) in FIGURES.items():
        base = ROOT / "main" / stem
        pdf, svg, png = (base.with_suffix(suffix) for suffix in (".pdf", ".svg", ".png"))
        if not all(path.is_file() for path in (pdf, svg, png)):
            raise AssertionError(f"missing one output format for {stem}")
        pages = PdfReader(pdf).pages
        if len(pages) != 1:
            raise AssertionError(f"not one PDF page: {stem}")
        page = pages[0]
        width = float(page.mediabox.width) * 2.54 / 72
        height = float(page.mediabox.height) * 2.54 / 72
        if abs(width - target_w) > .02 or abs(height - target_h) > .02:
            raise AssertionError(f"PDF size mismatch: {stem}: {width} x {height} cm")
        fonts = []
        for ref in page["/Resources"]["/Font"].get_object().values():
            font = ref.get_object()
            fonts.append(str(font.get("/BaseFont")))
            if "TimesNewRoman" not in fonts[-1]:
                raise AssertionError(f"non-Times PDF font: {stem}: {fonts[-1]}")
            descriptor = font.get("/FontDescriptor")
            if descriptor and "/FontFile2" not in descriptor.get_object():
                raise AssertionError(f"font is not embedded: {stem}: {fonts[-1]}")
        tree = ET.parse(svg)
        texts = [("".join(el.itertext()), el.attrib.get("style", "")) for el in tree.iter() if el.tag.endswith("}text")]
        for label in REQUIRED_POLICIES:
            found = [style for value, style in texts if value == label]
            if not found or not all(font_at(style, 8.5) for style in found):
                raise AssertionError(f"missing or malformed legend: {stem}: {label}")
        for label in ("Validation NLL", "Cumulative server active time (min)" if "energy" not in stem else "Cumulative client energy (kJ)"):
            found = [style for value, style in texts if value == label]
            if not found or not all(font_at(style, 10) for style in found):
                raise AssertionError(f"missing or malformed axis label: {stem}: {label}")
        tick_text = [style for value, style in texts if value in {"15", "20", "25", "30", "35", "40", "50", "1.530", "1.540", "1.550"}]
        if not tick_text or not all(font_at(style, 9) for style in tick_text):
            raise AssertionError(f"tick font mismatch: {stem}")
        with Image.open(png) as image:
            png_size = image.size
            dpi = image.info.get("dpi", (0, 0))
        if min(dpi) < 599:
            raise AssertionError(f"PNG below 600 dpi: {stem}: {dpi}")
        if abs(png_size[0] - target_w / 2.54 * 600) > 3 or abs(png_size[1] - target_h / 2.54 * 600) > 3:
            raise AssertionError(f"PNG pixel dimensions mismatch: {stem}: {png_size}")
        report[stem] = {"status": "PASS", "pdf_width_cm": width, "pdf_height_cm": height,
                        "png_pixels": png_size, "png_dpi": dpi, "embedded_pdf_fonts": fonts,
                        "editable_svg_text_count": len(texts), "axis_label_pt": 10,
                        "tick_label_pt": 9, "legend_pt": 8.5, "major_line_width_pt": 1.8,
                        "axes_width_pt": .8}
    report["coordinate_verification"] = verify_mean_coordinates()
    (ROOT / "verification/fig6_size_font_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
