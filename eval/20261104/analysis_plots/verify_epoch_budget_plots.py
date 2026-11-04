#!/usr/bin/env python3
"""Independently check revised Fig. 7/8 points against frozen raw RTX logs."""

from __future__ import annotations

import csv
import json
import math
import subprocess
from pathlib import Path

from PIL import Image


OUT = Path(__file__).resolve().parents[1]
WORKSPACE = OUT.parents[1]
ROUNDS = (0, 10, 20, 30, 40, 41, 42, 50, 60, 70, 80, 90, 100)
STEMS = (
    "fig7a_equal_work_local_epochs",
    "fig7b_window_filling_local_epochs",
    "fig8a_d128_training_budget_management",
    "fig8b_d512_training_budget_management",
)


def csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def same(observed: float, expected: float, context: str) -> None:
    if not math.isclose(observed, expected, rel_tol=0, abs_tol=1e-12):
        raise AssertionError(f"{context}: {observed} != {expected}")


def verify_fig7() -> dict:
    plotted = csv_rows(OUT / "verification/fig7_single_panel_plotted_values.csv")
    preserved = csv_rows(OUT / "verification/fig7a_equal_work_source_checkpoints.csv")
    assert len(plotted) == 42 and len(preserved) == 39
    selected = [row for row in plotted if row["output_stem"].startswith("fig7a")]
    assert len(selected) == 3 and {int(row["round"]) for row in selected} == {100}
    checked = 0
    endpoints = {}
    for rows in (plotted, preserved):
        for row in rows:
            assert row["budget"] == "D128" and int(row["training_seed"]) == 42
            assert row["loss_mode"] == "full_sequence_64_token"
            path = WORKSPACE / "outputs/loracast_long_horizon_100r/runs" / row["run_id"] / "validation.jsonl"
            raw = next(item for item in jsonl(path) if item["round"] == int(row["round"]))
            assert raw["measurement"] == "RTX_actual_model_validation"
            assert raw["checkpoint_sha256"] == row["checkpoint_sha256"]
            same(float(row["balanced_validation_nll"]), raw["balanced_client_loss"], row["run_id"])
            checked += 1
            if rows is plotted and row["output_stem"].startswith("fig7a"):
                endpoints[f"E{row['epoch']}"] = raw["balanced_client_loss"]
    previous = csv_rows(OUT / "verification/fig7_r100_endpoints.csv")
    previous_checks = 0
    for regime, prefix in (("equal_work", "fig7a"), ("heterogeneous", "fig7b")):
        for epoch in (1, 2, 4):
            old = next(row for row in previous if row["budget"] == "D128"
                       and row["regime"] == regime and int(row["epoch"]) == epoch)
            new = next(row for row in plotted if row["output_stem"].startswith(prefix)
                       and int(row["epoch"]) == epoch and int(row["round"]) == 100)
            same(float(new["balanced_validation_nll"]),
                 float(old["r100_balanced_validation_nll"]), f"old/new Fig7 {regime}/E{epoch}")
            previous_checks += 1
    return {"plotted_points": len(plotted), "preserved_equal_work_checkpoints": len(preserved),
            "raw_point_comparisons": checked, "previous_r100_comparisons": previous_checks,
            "seed": 42, "r100_nll": endpoints}


def verify_fig8() -> dict:
    plotted = csv_rows(OUT / "verification/fig8_single_panel_plotted_values.csv")
    assert len(plotted) == 78
    endpoints = {}
    comparisons = 0
    previous = csv_rows(OUT / "tables/fig8_mean_values.csv")
    previous_checks = 0
    for budget in ("d128", "d512"):
        endpoints[budget.upper()] = {}
        for policy in ("nr", "full", "exposurecap2"):
            series = {}
            for seed in (43, 44, 45):
                path = (WORKSPACE / "exposure_policy/raw/confirmation_runs" /
                        f"confirm_s{seed}_{budget}_{policy}/validation.jsonl")
                rows = jsonl(path)
                assert [item["round"] for item in rows] == list(ROUNDS)
                assert all(item["measurement"] == "RTX_actual_model_validation" and
                           item["loss_mode"] == "full_sequence_64_token" for item in rows)
                series[seed] = {item["round"]: item["balanced_client_loss"] for item in rows}
            for round_number in ROUNDS:
                matches = [row for row in plotted if row["budget"] == budget.upper()
                           and row["source_policy"] == policy and int(row["round"]) == round_number]
                assert len(matches) == 1
                row = matches[0]
                assert row["seeds"] == "43,44,45" and int(row["seed_count"]) == 3
                expected = sum(series[seed][round_number] for seed in (43, 44, 45)) / 3
                same(float(row["balanced_nll_mean"]), expected, f"{budget}/{policy}/R{round_number}")
                comparisons += 1
                if round_number == 100:
                    endpoints[budget.upper()][policy] = expected
                    old = next(item for item in previous if item["budget"] == budget.upper()
                               and item["source_policy"] == policy and int(item["round"]) == 100)
                    same(expected, float(old["mean"]), f"old/new Fig8 {budget}/{policy}")
                    previous_checks += 1
    return {"mean_points": len(plotted), "raw_mean_comparisons": comparisons,
            "previous_r100_comparisons": previous_checks,
            "seeds": [43, 44, 45], "r100_mean_nll": endpoints}


def verify_exports() -> dict:
    result = {}
    for stem in STEMS:
        base = OUT / "main" / stem
        pdf = base.with_suffix(".pdf")
        svg = base.with_suffix(".svg")
        png = base.with_suffix(".png")
        info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
        page = next(line for line in info.splitlines() if line.startswith("Page size:"))
        parts = page.split()
        size = [float(parts[2]) / 72 * 2.54, float(parts[4]) / 72 * 2.54]
        assert all(abs(actual - target) < 0.01 for actual, target in zip(size, (8.6, 5.2)))
        assert "<text" in svg.read_text()
        with Image.open(png) as image:
            assert image.info.get("dpi") and min(image.info["dpi"]) >= 599
            pixels = list(image.size)
        result[stem] = {"size_cm": size, "png_pixels": pixels,
                        "pdf_svg_png": True, "status": "PASS"}
    return result


def main() -> None:
    record = {"status": "PASS", "fig7": verify_fig7(), "fig8": verify_fig8(),
              "exports": verify_exports(),
              "measurement_scope": "Actual RTX training and full-sequence 64-token balanced-client NLL; no edge timing inferred"}
    path = OUT / "verification/fig7_8_revision_audit.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"PASS: {path}")


if __name__ == "__main__":
    main()
