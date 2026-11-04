#!/usr/bin/env python3
"""Alternative Figure 5(a): logged Pi5/NX pre-dispatch workload search.

This is a compact single-column alternative to the historical predictor MARE
panel. It reuses an *actual* round-3 decision record, not reconstructed costs.
It overlaps in evidence with the existing Section III Figure 3 and should not
be described as an independent prediction-accuracy experiment.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_cost_prediction_adaptation import MAIN, VERIFY, load_style, sha256


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "exposure_policy/system_design_figures/inputs/pi5_nx_s43_r03_preallocation_decision.json"
SOURCE_CHECK = ROOT / "exposure_policy/system_design_figures/figures/figure3_validation.json"


def main() -> None:
    MAIN.mkdir(parents=True, exist_ok=True)
    VERIFY.mkdir(parents=True, exist_ok=True)
    style = load_style()
    source = json.loads(SOURCE.read_text())
    check = json.loads(SOURCE_CHECK.read_text())
    if sha256(SOURCE) != check["source_sha256"]:
        raise ValueError("logged decision differs from verified Section III source")
    if (source["round"] != 3 or source["event"] != "allocation_decided_before_train_dispatch"
            or source["policy"] != "C_original_online_cost"):
        raise ValueError("unexpected Pi5/NX decision record")
    if not math.isclose(source["safety_margin"], 0.95):
        raise ValueError("unexpected compute-window safety margin")
    selected_s, selected_h, selected_time = source["selected_by_original_search"]
    assert (selected_s, selected_h) == (52, 51)
    reference = float(source["pi_reference_compute_s"])
    allowed = reference * float(source["safety_margin"])
    records: list[dict[str, object]] = []
    plotted: set[tuple[int, int, float]] = set()
    for idx, candidate in enumerate(source["tested_candidates_in_original_search"], start=1):
        s = int(candidate["samples"])
        h = int(candidate["step_cap"])
        # The logged run fixes B=E=1; H=0 denotes an uncapped step count.
        effective_steps = h if h > 0 else s
        predicted = float(candidate["predicted_compute_s"])
        key = (s, effective_steps, predicted)
        is_selected = s == selected_s and h == selected_h and math.isclose(predicted, selected_time)
        valid = candidate["decision"] == "accepted_by_time"
        if valid != (predicted <= allowed + 1e-9):
            raise ValueError(f"logged feasibility mismatch at candidate {idx}")
        records.append({
            "candidate_order": idx,
            "samples_S": s,
            "step_cap_H": h,
            "effective_optimizer_steps": effective_steps,
            "workload_size_S_times_H_eff": s * effective_steps,
            "predicted_compute_s": predicted,
            "pi_reference_compute_s": reference,
            "allowed_compute_s": allowed,
            "feasible": int(valid),
            "selected": int(is_selected),
            "plotted": int(key not in plotted),
            "source_json": str(SOURCE),
            "source_sha256": sha256(SOURCE),
        })
        plotted.add(key)
    if (len(records), sum(r["plotted"] for r in records), sum(r["selected"] for r in records)) != (17, 16, 1):
        raise ValueError("candidate count or selection differs from verified record")
    values = VERIFY / "fig5a_alternative_reference_window_values.csv"
    with values.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    fig, ax = plt.subplots(figsize=style.cm_to_inch(8.6, 5.2))
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.24, top=0.92)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", color="black", linewidth=0.5, alpha=0.25)
    ax.tick_params(direction="out", width=0.8, top=False, right=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    for r in records:
        if not r["plotted"] or r["selected"]:
            continue
        if r["feasible"]:
            ax.plot(r["predicted_compute_s"], r["workload_size_S_times_H_eff"],
                    marker="o", linestyle="none", markerfacecolor="white",
                    markeredgecolor="black", markersize=4.2, zorder=3)
        else:
            ax.plot(r["predicted_compute_s"], r["workload_size_S_times_H_eff"],
                    marker="x", linestyle="none", color="black",
                    markersize=5, markeredgewidth=1.1, zorder=3)
    chosen = next(r for r in records if r["selected"])
    ax.plot(chosen["predicted_compute_s"], chosen["workload_size_S_times_H_eff"],
            marker="s", linestyle="none", markerfacecolor="black",
            markeredgecolor="black", markersize=6.0, zorder=4)
    ax.axvline(allowed, linestyle=":", color="black", linewidth=1.1, zorder=1)
    ax.axvline(reference, linestyle="--", color="black", linewidth=1.1, zorder=1)
    ax.text(allowed - 0.15, 2910, "Allowed", ha="right", va="top", fontsize=8.5)
    ax.text(reference + 0.17, 2910, "Reference", ha="left", va="top", fontsize=8.5)
    handles = [
        Line2D([], [], color="black", marker="o", linestyle="none",
               markerfacecolor="white", markersize=4.2, label="Within limit"),
        Line2D([], [], color="black", marker="x", linestyle="none",
               markersize=5, label="Exceeds"),
        Line2D([], [], color="black", marker="s", linestyle="none",
               markerfacecolor="black", markersize=6, label="Selected"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.03, 0.92), frameon=False,
              fontsize=8.5, handletextpad=0.3, borderpad=0.2)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 3050)
    ax.set_xticks((0, 5, 10, 15, 20))
    ax.set_yticks((0, 1000, 2000, 3000))
    ax.set_xlabel("Predicted compute time (s)")
    ax.set_ylabel("Workload size")
    stem = "fig5a_alternative_reference_window"
    fig.savefig(MAIN / f"{stem}.svg", bbox_inches=None)
    style.savefig(fig, str(MAIN / stem), png_dpi=600)
    result = {
        "status": "CONDITIONAL_ALTERNATIVE; same Pi5/NX R3 source as Section III Figure 3",
        "source_json": str(SOURCE), "source_sha256": sha256(SOURCE),
        "run_id": "final-dynamic-s43-f-r16-20261019", "seed": 43,
        "round": 3, "version": "October Pi5/NX original online-cost allocator",
        "prediction_timing": "logged before train dispatch",
        "metric_boundary": "predicted local compute only; no post-execution time or energy plotted",
        "candidate_count": 17, "unique_plotted_workloads": 16,
        "selected_S_H": [selected_s, selected_h],
        "reference_compute_s": reference, "allowed_compute_s": allowed,
        "selected_predicted_compute_s": selected_time,
        "selection": "actual original allocator decision; no reconstruction",
        "condition": "B=E=1; H=0 means uncapped local steps",
        "formats": ["PDF", "SVG", "PNG_600dpi"],
        "intended_size_cm": [8.6, 5.2],
    }
    (VERIFY / "fig5a_alternative_reference_window_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
