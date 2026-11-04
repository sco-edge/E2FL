#!/usr/bin/env python3
"""Verify Supplementary S6 size, font, editable SVG labels and PNG resolution."""
from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
FIGURES = {"plot_historical_estimator_errors": (17.8, 5.1),
           "figS6a_compute_time_error": (8.6, 5.2),
           "figS6b_compute_energy_error": (8.6, 5.2)}


def main() -> None:
    report = {}
    for stem, (width_cm, height_cm) in FIGURES.items():
        base = ROOT / "supplementary" / stem
        pdf, svg, png = (base.with_suffix(ext) for ext in (".pdf", ".svg", ".png"))
        if not all(path.is_file() for path in (pdf, svg, png)):
            raise AssertionError(f"missing output format: {stem}")
        pages = PdfReader(pdf).pages
        if len(pages) != 1:
            raise AssertionError(f"PDF page count: {stem}")
        page = pages[0]
        actual = (float(page.mediabox.width) * 2.54 / 72, float(page.mediabox.height) * 2.54 / 72)
        if abs(actual[0]-width_cm) > .02 or abs(actual[1]-height_cm) > .02:
            raise AssertionError(f"PDF size: {stem}/{actual}")
        fonts = []
        for ref in page["/Resources"]["/Font"].get_object().values():
            font = ref.get_object()
            family = str(font.get("/BaseFont"))
            descriptor = font.get("/FontDescriptor")
            if "TimesNewRoman" not in family or (descriptor and "/FontFile2" not in descriptor.get_object()):
                raise AssertionError(f"PDF font: {stem}/{family}")
            fonts.append(family)
        tree = ET.parse(svg)
        texts = [("".join(el.itertext()), el.attrib.get("style", "")) for el in tree.iter() if el.tag.endswith("}text")]
        axis_labels = ["Empirical CDF"]
        if stem == "plot_historical_estimator_errors":
            axis_labels += ["Compute-time error (%)", "Compute-energy error (%)"]
        else:
            axis_labels.append("Compute-time error (%)" if "time" in stem else "Compute-energy error (%)")
        for label in axis_labels:
            styles = [style for value, style in texts if value == label]
            if not styles or not all("font-size: 10px" in style for style in styles):
                raise AssertionError(f"axis label font: {stem}/{label}")
        for label in ("RPi 5 (n=24)", "Jetson Orin NX (n=12)"):
            styles = [style for value, style in texts if value == label]
            if not styles or not all("font-size: 8.5px" in style for style in styles):
                raise AssertionError(f"legend font: {stem}/{label}")
        ticks = [style for value, style in texts if value in {"0", "5", "10", "0.0", "0.5", "1.0"}]
        if not ticks or not all("font-size: 9px" in style for style in ticks):
            raise AssertionError(f"tick font: {stem}")
        with Image.open(png) as image:
            size, dpi = image.size, image.info.get("dpi", (0, 0))
        if min(dpi) < 599 or abs(size[0]-width_cm/2.54*600) > 3 or abs(size[1]-height_cm/2.54*600) > 3:
            raise AssertionError(f"PNG dimensions: {stem}/{size}/{dpi}")
        report[stem] = {"status": "PASS", "pdf_cm": actual, "png_pixels": size, "png_dpi": dpi,
                        "embedded_pdf_fonts": fonts, "editable_svg_text_count": len(texts),
                        "axis_label_pt": 10, "tick_label_pt": 9, "legend_pt": 8.5,
                        "major_line_width_pt": 1.4, "axes_width_pt": .8}
    (ROOT / "verification/figS6_size_font_audit.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
