#!/usr/bin/env python3
"""Historical Pi5/Orin NX estimator-error ECDF, verified back to July raw logs."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_device_learning_efficiency import load_style, ROOT


DATA_DIR = Path("/Users/jang-won-u/Documents/GitHub/EEFL/plot/revision/loracast_plot_scripts_v2/figures/fig5_estimator_devicewise")
SAMPLES = DATA_DIR / "fig5_estimator_samples_used.csv"
SUMMARY = DATA_DIR / "fig5_summary_after_warmup.csv"
SCRIPT = Path("/Users/jang-won-u/Documents/GitHub/EEFL/plot/revision/loracast_plot_scripts_v2/plot_estimator_devicewise.py")
EXPECTED_RUNS = {"MAXN", "15W", "10W"}
DEVICES = ("RPi 5", "Jetson Orin NX")
DEV_STYLE = {
    "RPi 5": {"color": "#4d4d4d", "linestyle": "-", "marker": "o", "raw": "RPi5"},
    "Jetson Orin NX": {"color": "#000000", "linestyle": "--", "marker": "s", "raw": "jetson_orin_nx"},
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual: float, expected: float, what: str, atol: float = 1e-6) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=atol):
        raise AssertionError(f"{what}: {actual} != {expected}")


def raw_path(recorded: str) -> Path:
    original = Path(recorded)
    if original.is_file():
        return original
    # Historical MAXN source was moved into bkp after the derived table was
    # created. Resolve this exact known relocation, never a most-recent CSV.
    if original.parent.name == "20. power (LoRACast-Online_MAXN)":
        moved = original.parent / "bkp" / original.name
        if moved.is_file():
            return moved
    raise FileNotFoundError(f"missing original predictor log: {recorded}")


def validated_samples() -> tuple[list[dict], dict[str, str], dict]:
    records = read_csv(SAMPLES)
    expected = read_csv(SUMMARY)
    if len(records) != 108 or len(expected) != 4:
        raise AssertionError("historical estimator sample/summary count changed")
    if {row["run"] for row in records} != EXPECTED_RUNS:
        raise AssertionError("unexpected historical estimator runs")
    raw_sources = {recorded: raw_path(recorded) for recorded in {r["source_csv"] for r in records}}
    source_hashes = {str(path): sha(path) for path in raw_sources.values()}
    raw_by_source = {}
    for recorded, path in raw_sources.items():
        indexed = {}
        for row in read_csv(path):
            if row["phase"] != "train":
                continue
            key = (int(row["round"]), row["device_name"])
            if key in indexed:
                raise AssertionError(f"duplicate raw training row: {path}/{key}")
            indexed[key] = row
        raw_by_source[recorded] = indexed

    plotted = []
    for row in records:
        if not row["compute_time_error_pct"]:
            continue  # The derived CSV includes empty evaluation-phase entries.
        run, device, round_no = row["run"], row["device"], int(row["round"])
        raw = raw_by_source[row["source_csv"]][(round_no, DEV_STYLE[device]["raw"])]
        actual_time = float(raw["loop_time"])
        predicted_time = float(raw["pred_train_time_self"])
        measured_power_w = float(raw["energy"])
        actual_energy = measured_power_w * actual_time
        calibrated_power = float(row["calibrated_power_w"])
        predicted_energy = calibrated_power * predicted_time
        close(float(row["actual_compute_time_s"]), actual_time, f"actual time {run}/{device}/{round_no}")
        close(float(row["predicted_compute_time_s"]), predicted_time, f"predicted time {run}/{device}/{round_no}")
        close(float(row["actual_compute_energy_j"]), actual_energy, f"actual energy {run}/{device}/{round_no}")
        close(float(row["predicted_compute_energy_j"]), predicted_energy, f"predicted energy {run}/{device}/{round_no}")
        if actual_time <= 0 or actual_energy <= 0:
            raise AssertionError("nonpositive actual compute cost")
        close(float(row["compute_time_error_pct"]), abs(predicted_time - actual_time)/actual_time*100,
              f"compute time APE {run}/{device}/{round_no}")
        close(float(row["compute_energy_error_pct"]), abs(predicted_energy - actual_energy)/actual_energy*100,
              f"compute energy APE {run}/{device}/{round_no}")
        if round_no > 2:
            plotted.append({"run": run, "device": device, "round": round_no,
                            "compute_time_error_pct": float(row["compute_time_error_pct"]),
                            "compute_energy_error_pct": float(row["compute_energy_error_pct"]),
                            "original_logged_csv": str(raw_sources[row["source_csv"]])})

    if len(plotted) != 36:
        raise AssertionError("post-warm-up sample count should be 36")
    verification = {}
    for metric in ("compute_time_error_pct", "compute_energy_error_pct"):
        for device in DEVICES:
            values = [r[metric] for r in plotted if r["device"] == device]
            expected_row = next((r for r in expected if r["metric"] == metric and r["device"] == device), None)
            if expected_row is None or len(values) != int(expected_row["n"]):
                raise AssertionError(f"historical summary count mismatch: {metric}/{device}")
            close(float(expected_row["mean_pct"]), float(np.mean(values)), f"historical mean {metric}/{device}")
            close(float(expected_row["median_pct"]), median(values), f"historical median {metric}/{device}")
            close(float(expected_row["p90_pct"]), float(np.percentile(values, 90)), f"historical p90 {metric}/{device}")
            verification[f"{device}/{metric}"] = {"n": len(values), "mean_pct": float(np.mean(values)),
                                                 "median_pct": median(values), "p90_pct": float(np.percentile(values,90))}
    return plotted, source_hashes, verification


def ecdf_panel(ax, records: list[dict], *, metric: str) -> None:
    for device in DEVICES:
        errors = sorted(float(row[metric]) for row in records if row["device"] == device)
        n = len(errors)
        x = [0.0, *errors]
        y = [0.0, *(index/n for index in range(1, n+1))]
        style = DEV_STYLE[device]
        ax.step(x, y, where="post", label=f"{device} (n={n})", color=style["color"],
                linestyle=style["linestyle"], linewidth=1.4, zorder=3)
        ax.plot(errors, y[1:], linestyle="none", marker=style["marker"], markersize=2.7,
                color=style["color"], markerfacecolor="white", markeredgewidth=.7, zorder=4)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 1.02)
    ax.set_xticks((0, 5, 10))
    ax.set_yticks((0, .5, 1))
    ax.set_xlabel("Compute-time error (%)" if metric == "compute_time_error_pct" else "Compute-energy error (%)")
    ax.grid(axis="y", color="black", linestyle="--", linewidth=.5, alpha=.25)
    ax.tick_params(width=.8, length=3)
    ax.set_axisbelow(True)


def save(fig, stem: str, style_module) -> None:
    base = ROOT / "supplementary" / stem
    fig.savefig(str(base)+".svg", bbox_inches=None)
    style_module.savefig(fig, str(base), png_dpi=600)


def main() -> None:
    style = load_style()
    records, source_hashes, summary = validated_samples()
    fig, axes = plt.subplots(1, 2, figsize=style.cm_to_inch(17.8, 5.1), sharey=True)
    fig.subplots_adjust(left=.075, right=.985, bottom=.26, top=.77, wspace=.15)
    ecdf_panel(axes[0], records, metric="compute_time_error_pct")
    ecdf_panel(axes[1], records, metric="compute_energy_error_pct")
    axes[0].set_ylabel("Empirical CDF")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(.54,.98), frameon=False)
    save(fig, "plot_historical_estimator_errors", style)
    for metric, stem in (("compute_time_error_pct", "figS6a_compute_time_error"),
                         ("compute_energy_error_pct", "figS6b_compute_energy_error")):
        single, ax = plt.subplots(figsize=style.cm_to_inch(8.6, 5.2))
        single.subplots_adjust(left=.18, right=.97, bottom=.25, top=.75)
        ecdf_panel(ax, records, metric=metric)
        ax.set_ylabel("Empirical CDF")
        h, lab = ax.get_legend_handles_labels()
        single.legend(h, lab, ncol=1, loc="upper center", bbox_to_anchor=(.54,.99), frameon=False)
        save(single, stem, style)
    out = ROOT / "verification/figS6_plot_data.csv"
    with out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    audit = {"status": "PASS", "version_scope": "Historical July Pi5/Orin NX online-calibrated cost estimator; MAXN, 15W, 10W",
             "warmup_rounds_excluded": [1,2], "plotted_post_warmup_samples": 36,
             "metric": "Absolute relative error of logged pre-execution local compute-time prediction and warm-up-power compute-energy estimate",
             "energy_boundary": "Logged average compute power times measured local compute duration; excludes communication, waiting, server and AP",
             "missing_components": ["No full server decision overhead inferred", "No current Nano+NX predictor error inferred"],
             "derived_sample_source": str(SAMPLES), "derived_sample_sha256": sha(SAMPLES),
             "summary_source": str(SUMMARY), "summary_sha256": sha(SUMMARY),
             "historical_parser_source": str(SCRIPT), "historical_parser_sha256": sha(SCRIPT),
             "raw_source_sha256": source_hashes, "summary_recalculated": summary,
             "note": "MAXN original raw CSV moved into bkp after derived table creation; exact basename and row values were matched."}
    (ROOT / "verification/figS6_validation.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps({"status": "PASS", "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
