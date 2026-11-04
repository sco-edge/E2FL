#!/usr/bin/env python3
"""Rebuild the archived before/after sheet from the earlier eight-panel revision.

The former Fig. 5 exports live under comparison/superseded_fig5 after the
three-panel estimator/contention suite replaced them in the main directory.
"""

from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
STEMS = (
    "fig5a_historical_compute_prediction_error",
    "fig5b_runtime_workload_adaptation",
    "fig6a_learning_vs_active_time",
    "fig6b_learning_vs_client_energy",
    "fig7a_equal_work_local_epochs",
    "fig7b_window_filling_local_epochs",
    "fig8a_d128_training_budget_management",
    "fig8b_d512_training_budget_management",
)
TITLES = (
    "Fig. 5(a)  Compute prediction error",
    "Fig. 5(b)  Runtime workload adaptation",
    "Fig. 6(a)  NLL versus active time",
    "Fig. 6(b)  NLL versus client energy",
    "Fig. 7(a)  Equal-work local epochs",
    "Fig. 7(b)  Window-filling local epochs",
    "Fig. 8(a)  D128 budget management",
    "Fig. 8(b)  D512 budget management",
)


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/System/Library/Fonts/Supplemental/Times New Roman.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def make_sheet(indices: range, destination: Path) -> None:
    cell_w, cell_h, label_h = 850, 520, 48
    margin, gap = 24, 22
    width = margin * 2 + cell_w * 2 + gap
    height = margin * 2 + 64 + len(indices) * (cell_h + label_h + gap)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin), "Before", fill="#222222", font=font(30))
    draw.text((margin + cell_w + gap, margin), "Revised", fill="#222222", font=font(30))
    y_start = margin + 64
    for row_pos, index in enumerate(indices):
        y = y_start + row_pos * (cell_h + label_h + gap)
        stem = STEMS[index]
        revised = ((ROOT / "comparison/superseded_fig5") if index < 2 else (ROOT / "main")) / f"{stem}.png"
        for column, image_path in enumerate((ROOT / "comparison/before" / f"{stem}.png", revised)):
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            with Image.open(image_path) as original:
                panel = original.convert("RGB")
                panel.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
            x = margin + column * (cell_w + gap) + (cell_w - panel.width) // 2
            canvas.paste(panel, (x, y + (cell_h - panel.height) // 2))
        draw.text((margin, y + cell_h + 4), TITLES[index], fill="#222222", font=font(23))
        draw.line((margin, y + cell_h + label_h + gap // 2, width - margin,
                   y + cell_h + label_h + gap // 2), fill="#d0d0d0", width=1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


if __name__ == "__main__":
    make_sheet(range(8), ROOT / "comparison/before_after_overview.png")
    make_sheet(range(4), ROOT / "comparison/fig5_6_before_after.png")
    make_sheet(range(4, 8), ROOT / "comparison/fig7_8_before_after.png")
    print("Wrote full and split before/after overviews")
