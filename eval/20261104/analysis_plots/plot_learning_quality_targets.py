#!/usr/bin/env python3
"""Supplementary S3: actual-checkpoint time and energy to frozen NLL targets."""
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
from matplotlib.patches import Patch

from plot_device_learning_efficiency import load_style, NAMES, POLICIES, SEEDS, SOURCE, ROOT


TIME_SOURCE = SOURCE / "time_to_target.csv"
ENERGY_SOURCE = SOURCE / "energy_to_target.csv"
ALIGNMENT_SOURCE = SOURCE / "checkpoint_time_energy_alignment.csv"
TARGETS = (1.535, 1.533)
TARGET_STYLE = {
    1.535: {"facecolor": "white", "hatch": "///", "label": "NLL ≤ 1.535"},
    1.533: {"facecolor": "#d9d9d9", "hatch": "\\\\\\", "label": "NLL ≤ 1.533"},
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validated_targets() -> tuple[dict, dict]:
    time_records = read_csv(TIME_SOURCE)
    energy_records = read_csv(ENERGY_SOURCE)
    alignment = read_csv(ALIGNMENT_SOURCE)
    if len(time_records) != 24 or len(energy_records) != 24:
        raise AssertionError("expected 3 seeds x 4 methods x 2 frozen targets")
    # Both historical outputs have the same complete cost fields; verify their
    # byte identity to avoid accidentally combining different revisions.
    if source_hash(TIME_SOURCE) != source_hash(ENERGY_SOURCE):
        raise AssertionError("time/energy target source revisions differ")
    key = lambda row: (int(row["seed"]), row["policy"], float(row["target_nll"]))
    target = {key(row): row for row in time_records}
    alignment_by_key = {(int(row["seed"]), row["policy"], int(row["round"])): row for row in alignment}
    if len(target) != 24 or len(alignment_by_key) != 156:
        raise AssertionError("duplicate or missing target/checkpoint records")
    for seed in SEEDS:
        for policy in POLICIES:
            for threshold in TARGETS:
                row = target[(seed, policy, threshold)]
                first = next((round_no for round_no in range(13)
                              if float(alignment_by_key[(seed, policy, round_no)]["full_sequence_nll_bf16"]) <= threshold), None)
                if row["fixed_before_confirmation"] != "True":
                    raise AssertionError("target was not frozen before independent confirmation")
                if first is None:
                    if row["attainment"] != "not_reached" or any(row[field] for field in
                        ("first_reached_round", "active_time_s", "client_energy_j", "checkpoint_sha256")):
                        raise AssertionError(f"censored run was given a value: {seed}/{policy}/{threshold}")
                else:
                    expected = alignment_by_key[(seed, policy, first)]
                    if row["attainment"] != "reached" or int(row["first_reached_round"]) != first:
                        raise AssertionError(f"incorrect first attained checkpoint: {seed}/{policy}/{threshold}")
                    if row["run_id"] != expected["run_id"] or row["checkpoint_sha256"] != expected["checkpoint_sha256"]:
                        raise AssertionError(f"checkpoint provenance mismatch: {seed}/{policy}/{threshold}")
                    for field, source in (("active_time_s", "cumulative_active_time_s"),
                                          ("client_energy_j", "cumulative_client_energy_j")):
                        if not math.isclose(float(row[field]), float(expected[source]), rel_tol=1e-10, abs_tol=1e-5):
                            raise AssertionError(f"checkpoint cost mismatch: {seed}/{policy}/{threshold}/{field}")
    return target, alignment_by_key


def panel(ax, target: dict, *, cost: str) -> list[dict]:
    field, scale, axis_label = (("active_time_s", 1/60, "Server-active time to target (min)") if cost == "time"
                                else ("client_energy_j", 1/1000, "Client energy to target (kJ)"))
    summary = []
    all_values = [float(row[field]) * scale for row in target.values() if row["attainment"] == "reached"]
    maximum = max(all_values)
    for index, policy in enumerate(POLICIES):
        base_y = len(POLICIES) - 1 - index
        for threshold, offset in ((1.535, +.19), (1.533, -.19)):
            attained = [(seed, target[(seed, policy, threshold)]) for seed in SEEDS
                        if target[(seed, policy, threshold)]["attainment"] == "reached"]
            values = [float(row[field]) * scale for _, row in attained]
            y = base_y + offset
            if values:
                midpoint = median(values)
                ax.barh(y, midpoint, height=.32, facecolor=TARGET_STYLE[threshold]["facecolor"],
                        edgecolor="black", linewidth=.9, hatch=TARGET_STYLE[threshold]["hatch"], zorder=2)
                for seed, row in attained:
                    yy = y + (SEEDS.index(seed) - 1) * .075
                    ax.plot(float(row[field]) * scale, yy, marker="o", markersize=3.0,
                            markerfacecolor="black", markeredgecolor="white", markeredgewidth=.35,
                            linestyle="none", zorder=5)
                ax.text(maximum * 1.075, y, f"{len(values)}/3", va="center", ha="left", fontsize=8.5)
            else:
                ax.text(maximum * .035, y, "0/3", va="center", ha="left", fontsize=8.5)
            summary.append({"panel": cost, "policy_code": policy, "method": NAMES[policy],
                            "target_nll": f"{threshold:.3f}", "reached_count": len(values), "seed_count": 3,
                            "median_among_reached": "" if not values else f"{median(values):.10g}",
                            "unit": "min" if cost == "time" else "kJ",
                            "reached_seed_ids": "|".join(str(seed) for seed, _ in attained)})
    ax.set_yticks([3, 2, 1, 0], [NAMES[policy] for policy in POLICIES])
    ax.set_xlim(0, maximum * 1.22)
    ax.set_ylim(-.58, 3.58)
    ax.set_xlabel(axis_label)
    ax.grid(axis="x", linestyle="--", linewidth=.5, color="black", alpha=.25)
    ax.set_axisbelow(True)
    ax.tick_params(width=.8, length=3)
    return summary


def save(fig, stem: str, style_module) -> None:
    base = ROOT / "supplementary" / stem
    fig.savefig(str(base) + ".svg", bbox_inches=None)
    style_module.savefig(fig, str(base), png_dpi=600)


def main() -> None:
    style = load_style()
    target, _ = validated_targets()
    fig, axes = plt.subplots(1, 2, figsize=style.cm_to_inch(17.8, 5.1), sharey=True)
    fig.subplots_adjust(left=.18, right=.985, bottom=.245, top=.78, wspace=.13)
    all_summary = panel(axes[0], target, cost="time")
    all_summary.extend(panel(axes[1], target, cost="energy"))
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    legend = [Patch(facecolor=TARGET_STYLE[value]["facecolor"], edgecolor="black", linewidth=.9,
                    hatch=TARGET_STYLE[value]["hatch"], label=TARGET_STYLE[value]["label"]) for value in TARGETS]
    fig.legend(handles=legend, ncol=2, loc="upper center", bbox_to_anchor=(.56, .98), frameon=False)
    save(fig, "plot_learning_quality_targets", style)
    for cost, stem in (("time", "figS3a_time_to_target"), ("energy", "figS3b_energy_to_target")):
        one, ax = plt.subplots(figsize=style.cm_to_inch(8.6, 5.2))
        one.subplots_adjust(left=.32, right=.98, bottom=.25, top=.75)
        panel(ax, target, cost=cost)
        one.legend(handles=legend, ncol=2, loc="upper center", bbox_to_anchor=(.58, .98), frameon=False,
                   handlelength=1.4, columnspacing=.8)
        save(one, stem, style)
    write_csv(ROOT / "verification/figS3_target_summary.csv", all_summary)
    audit = {"status": "PASS", "seed_ids": list(SEEDS), "frozen_targets": list(TARGETS),
             "first_reached_checkpoint_only": True, "censored_runs_imputed": False,
             "time_source": str(TIME_SOURCE), "time_source_sha256": source_hash(TIME_SOURCE),
             "energy_source": str(ENERGY_SOURCE), "energy_source_sha256": source_hash(ENERGY_SOURCE),
             "checkpoint_alignment": str(ALIGNMENT_SOURCE), "checkpoint_alignment_sha256": source_hash(ALIGNMENT_SOURCE),
             "summary": all_summary}
    (ROOT / "verification/figS3_validation.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "reached_counts": [r for r in all_summary if r["panel"] == "time"]}, indent=2))


if __name__ == "__main__":
    main()
