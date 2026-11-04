#!/usr/bin/env python3
"""Check S2 publication exports against the inherited INFOCOM plot style."""
from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
FIGURES = {
    "figS2_workload_response": ("Pi compute (s)", "NX steps"),
    "figS2_efficiency_and_reply_barrier": ("FL input tokens / s", "Reply barrier (s)"),
}
EXPECTED_CM = (17.8, 5.1)
LEGEND = ("Fixed", "Frozen cost", "Last rate", "LoRACast")


def main() -> None:
    report = {}
    for stem, labels in FIGURES.items():
        base = ROOT / "supplementary" / stem
        pdf, svg, png = (base.with_suffix(ext) for ext in (".pdf", ".svg", ".png"))
        if not all(path.is_file() for path in (pdf, svg, png)):
            raise AssertionError(f"missing S2 export format: {stem}")
        pages = PdfReader(pdf).pages
        if len(pages) != 1:
            raise AssertionError(f"PDF page count: {stem}")
        page = pages[0]
        size_cm = (float(page.mediabox.width)*2.54/72,
                   float(page.mediabox.height)*2.54/72)
        if any(abs(actual-want)>.02 for actual,want in zip(size_cm,EXPECTED_CM)):
            raise AssertionError(f"PDF physical size: {stem}/{size_cm}")
        fonts = []
        for ref in page["/Resources"]["/Font"].get_object().values():
            font = ref.get_object()
            family = str(font.get("/BaseFont"))
            descriptor = font.get("/FontDescriptor")
            if "TimesNewRoman" not in family or (descriptor and "/FontFile2" not in descriptor.get_object()):
                raise AssertionError(f"PDF font not embedded Times New Roman: {stem}/{family}")
            fonts.append(family)
        texts = [("".join(el.itertext()),el.attrib.get("style",""))
                 for el in ET.parse(svg).iter() if el.tag.endswith("}text")]
        for label in (*labels,"FL round"):
            styles = [style for value,style in texts if value==label]
            if not styles or not all("font-size: 10px" in style for style in styles):
                raise AssertionError(f"axis label missing or wrong size: {stem}/{label}")
        for label in LEGEND:
            styles = [style for value,style in texts if value==label]
            if not styles or not all("font-size: 8.5px" in style for style in styles):
                raise AssertionError(f"legend missing or wrong size: {stem}/{label}")
        with Image.open(png) as raster:
            pixels,dpi = raster.size,raster.info.get("dpi",(0,0))
        if min(dpi)<599 or any(abs(pixel-cm/2.54*600)>3
                              for pixel,cm in zip(pixels,EXPECTED_CM)):
            raise AssertionError(f"PNG export resolution: {stem}/{pixels}/{dpi}")
        report[stem] = {"status":"PASS","pdf_cm":size_cm,"png_pixels":pixels,
                        "png_dpi":dpi,"embedded_pdf_fonts":fonts,
                        "editable_svg_text_count":len(texts),"axis_label_pt":10,
                        "tick_label_pt":9,"legend_pt":8.5,
                        "major_line_width_pt":1.4,"axes_width_pt":.8}
    output=ROOT/"verification"/"figS2_size_font_audit.json"
    output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
