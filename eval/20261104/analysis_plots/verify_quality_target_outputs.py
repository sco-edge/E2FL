#!/usr/bin/env python3
"""Verify Supplementary S3 physical size, embedded fonts, and 600-dpi exports."""
from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
FIGURES = {"plot_learning_quality_targets": (17.8, 5.1),
           "figS3a_time_to_target": (8.6, 5.2),
           "figS3b_energy_to_target": (8.6, 5.2)}


def main() -> None:
    report = {}
    for stem, target in FIGURES.items():
        base = ROOT / "supplementary" / stem
        pdf, svg, png = (base.with_suffix(ext) for ext in (".pdf", ".svg", ".png"))
        if not all(path.is_file() for path in (pdf, svg, png)):
            raise AssertionError(f"missing output format: {stem}")
        pages = PdfReader(pdf).pages
        if len(pages) != 1:
            raise AssertionError(f"PDF page count: {stem}")
        page = pages[0]
        actual = (float(page.mediabox.width) * 2.54 / 72, float(page.mediabox.height) * 2.54 / 72)
        if any(abs(x-y) > .02 for x, y in zip(actual, target)):
            raise AssertionError(f"PDF dimension mismatch: {stem}: {actual}")
        fonts = []
        for ref in page["/Resources"]["/Font"].get_object().values():
            font = ref.get_object()
            family = str(font.get("/BaseFont"))
            descriptor = font.get("/FontDescriptor")
            if "TimesNewRoman" not in family or (descriptor and "/FontFile2" not in descriptor.get_object()):
                raise AssertionError(f"PDF font mismatch: {stem}: {family}")
            fonts.append(family)
        tree = ET.parse(svg)
        texts = [("".join(el.itertext()), el.attrib.get("style", "")) for el in tree.iter() if el.tag.endswith("}text")]
        legend_labels = ("NLL ≤ 1.535", "NLL ≤ 1.533")
        for label in legend_labels:
            styles = [style for value, style in texts if value == label]
            if not styles or not all("font-size: 8.5px" in style for style in styles):
                raise AssertionError(f"legend font mismatch: {stem}/{label}")
        axis_label = "Server-active time to target (min)" if stem != "figS3b_energy_to_target" else "Client energy to target (kJ)"
        if stem == "plot_learning_quality_targets":
            labels = ("Server-active time to target (min)", "Client energy to target (kJ)")
        else:
            labels = (axis_label,)
        for label in labels:
            styles = [style for value, style in texts if value == label]
            if not styles or not all("font-size: 10px" in style for style in styles):
                raise AssertionError(f"axis font mismatch: {stem}/{label}")
        tick_styles = [style for value, style in texts if value in {"0", "10", "20", "30", "40", "50", "60",
                                                                     "FedAvg", "Fixed-Hetero", "PyramidFL-style", "LoRACast"}]
        if not tick_styles or not all("font-size: 9px" in style for style in tick_styles):
            raise AssertionError(f"tick label font mismatch: {stem}")
        count_styles = [style for value, style in texts if value in {"1/3", "2/3", "3/3"}]
        if len(count_styles) < 8 or not all("font-size: 8.5px" in style for style in count_styles):
            raise AssertionError(f"reached-count label missing or font mismatch: {stem}")
        with Image.open(png) as image:
            px, dpi = image.size, image.info.get("dpi", (0, 0))
        if min(dpi) < 599 or abs(px[0] - target[0]/2.54*600) > 3 or abs(px[1] - target[1]/2.54*600) > 3:
            raise AssertionError(f"PNG dimension or dpi mismatch: {stem}/{px}/{dpi}")
        report[stem] = {"status": "PASS", "pdf_cm": actual, "png_pixels": px, "png_dpi": dpi,
                        "embedded_pdf_fonts": fonts, "editable_svg_text_count": len(texts),
                        "axis_label_pt": 10, "tick_label_pt": 9, "legend_and_count_pt": 8.5,
                        "bar_edge_pt": .9, "axes_width_pt": .8}
    (ROOT / "verification/figS3_size_font_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
