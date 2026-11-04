#!/usr/bin/env python3
"""Supplementary S1: verified independent Pi5/NX F/A/B/C comparison.

No FL or model evaluation is run; this reads the admitted, completed seed-43/44
16-round physical campaign. Bar heights are means of two seed-level runs and
overlaid dots are the two actual run values.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_cost_prediction_adaptation import OUT, VERIFY, load_style, rows, sha256


CAMPAIGN = Path(
    "/Users/jang-won-u/Documents/Codex/2026-10-10/new-chat/outputs/"
    "final_dynamic_policy_comparison_20261019"
)
SUMMARY = CAMPAIGN / "policy_summary.csv"
MATRIX = CAMPAIGN / "current_effective_run_matrix.csv"
QC = CAMPAIGN / "final_qc.json"
SOURCE_MANIFEST = CAMPAIGN / "source_manifest_final.json"
SUPP = OUT / "supplementary"
LABEL = {"F": "Fixed", "A": "Frozen", "B": "Last rate", "C": "LoRACast"}
HATCH = {"F": "///", "A": "\\\\\\", "B": "xx", "C": "..."}


def plot_panel(ax, data: list[dict[str, object]], field: str, label: str) -> None:
    ax.grid(axis="y", linestyle="--", color="black", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.tick_params(direction="out", width=0.8, top=False, right=False)
    for x, policy in enumerate(("F", "A", "B", "C")):
        values = [float(r[field]) for r in data if r["policy"] == policy]
        if len(values) != 2:
            raise ValueError(f"expected seed 43/44 for {policy}: {values}")
        ax.bar(x, mean(values), width=0.6, facecolor="white", edgecolor="black",
               linewidth=0.9, hatch=HATCH[policy], zorder=2)
        for seed, marker, dx in ((43, "o", -0.11), (44, "s", 0.11)):
            val = next(float(r[field]) for r in data if r["policy"] == policy and r["seed"] == seed)
            ax.plot(x + dx, val, marker=marker, linestyle="none", color="black",
                    markerfacecolor="white", markersize=3.8, markeredgewidth=0.9, zorder=4)
    ax.set_xticks(range(4), [LABEL[p] for p in ("F", "A", "B", "C")])
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylabel(label)
    ax.set_ylim(bottom=0)


def main() -> None:
    style = load_style()
    SUPP.mkdir(parents=True, exist_ok=True)
    VERIFY.mkdir(parents=True, exist_ok=True)
    quality_gate = json.loads(QC.read_text())
    if quality_gate["status"] != "passed" or quality_gate["verified_runs"] != 8 or quality_gate["training_rounds"] != 128:
        raise ValueError("Pi5/NX campaign failed required integrity gate")
    source_manifest = json.loads(SOURCE_MANIFEST.read_text())
    if source_manifest["result_files_sha256"]["policy_summary.csv"] != sha256(SUMMARY):
        raise ValueError("summary changed since source manifest was sealed")
    if source_manifest["effective_matrix_sha256"] != sha256(MATRIX):
        raise ValueError("effective run matrix changed since source manifest was sealed")
    manifest_runs = {(int(r["seed"]), r["policy"]): r for r in source_manifest["runs"]}
    admitted = {(int(r["seed"]), r["policy"]): r for r in rows(MATRIX)}
    if len(manifest_runs) != 8 or set(manifest_runs) != set(admitted):
        raise ValueError("source manifest does not match admitted F/A/B/C run set")
    for key, matrix_run in admitted.items():
        original = manifest_runs[key]
        if original["run_id"] != matrix_run["experiment_id"] or original["config_sha256"] != matrix_run["config_sha256"]:
            raise ValueError(f"run/config mismatch between matrix and source manifest: {key}")
        verification_file = Path(original["verification_file"])
        if sha256(verification_file) != original["verification_sha256"]:
            raise ValueError(f"per-run verification changed since source manifest was sealed: {key}")
    records: list[dict[str, object]] = []
    for source in rows(SUMMARY):
        seed, policy, run_id = int(source["seed"]), source["policy"], source["run_id"]
        original = manifest_runs.get((seed, policy))
        if admitted.get((seed, policy), {}).get("experiment_id") != run_id or original is None:
            raise ValueError(f"non-admitted or mismatched Pi5/NX run: {run_id}")
        if seed not in (43, 44) or policy not in LABEL:
            raise ValueError(f"unexpected seed/policy in admitted summary: {seed}/{policy}")
        total_tokens = int(source["fl_input_tokens"])
        active_s = float(source["active_server_time_s"])
        active_j = float(source["active_client_device_energy_j"])
        throughput = float(source["fl_input_tokens_per_active_s"])
        energy_rate = float(source["active_j_per_1k_fl_input_tokens"])
        if not math.isclose(throughput, total_tokens / active_s, rel_tol=1e-10):
            raise ValueError(f"throughput ratio mismatch for {run_id}")
        if not math.isclose(energy_rate, 1000 * active_j / total_tokens, rel_tol=1e-10):
            raise ValueError(f"energy ratio mismatch for {run_id}")
        records.append({
            "seed": seed, "policy": policy, "display_name": LABEL[policy], "run_id": run_id,
            "input_tokens": total_tokens,
            "measured_server_active_s": active_s,
            "measured_client_device_active_j": active_j,
            "tokens_per_active_s": throughput,
            "j_per_1k_input_tokens": energy_rate,
            "source_policy_summary_csv": str(SUMMARY),
            "source_policy_summary_sha256": sha256(SUMMARY),
            "source_run_config_sha256": original["config_sha256"],
            "source_run_verification_sha256": original["verification_sha256"],
        })
    if len(records) != 8 or {(r["seed"], r["policy"]) for r in records} != {(s, p) for s in (43, 44) for p in LABEL}:
        raise ValueError("incomplete F/A/B/C independent seed coverage")
    values = VERIFY / "fig5_s1_pi5_nx_ablation_values.csv"
    with values.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    fig, axes = plt.subplots(1, 2, figsize=style.cm_to_inch(17.8, 5.1))
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.24, top=0.93, wspace=0.31)
    plot_panel(axes[0], records, "tokens_per_active_s", "Input tokens / active s")
    plot_panel(axes[1], records, "j_per_1k_input_tokens", "Energy (J / 1K input tokens)")
    # The caption identifies circle/square as seeds 43/44. An in-panel legend
    # would cover the top of these short bars at the publication height.
    stem = "s1_pi5_nx_cost_ablation"
    fig.savefig(SUPP / f"{stem}.svg", bbox_inches=None)
    style.savefig(fig, str(SUPP / stem), png_dpi=600)
    report = {
        "status": "READY_SUPPLEMENTARY; two independent paired seeds only",
        "source_summary_csv": str(SUMMARY), "source_summary_sha256": sha256(SUMMARY),
        "source_admitted_matrix_csv": str(MATRIX), "source_admitted_matrix_sha256": sha256(MATRIX),
        "source_final_qc_json": str(QC), "source_final_qc_sha256": sha256(QC),
        "source_manifest_json": str(SOURCE_MANIFEST), "source_manifest_sha256": sha256(SOURCE_MANIFEST),
        "run_ids": [r["run_id"] for r in records], "seeds": [43, 44],
        "same_seed_round0_adapter_sha256": quality_gate["same_seed_round0_adapter_sha256"],
        "excluded_seed43_B_run": quality_gate["original_excluded_seed43_B_run"],
        "admitted_seed43_B_run": quality_gate["effective_seed43_B_run"],
        "admitted_runs": [
            {"seed": r["seed"], "policy": r["policy"], "run_id": r["run_id"],
             "config_sha256": r["source_run_config_sha256"],
             "verification_sha256": r["source_run_verification_sha256"]}
            for r in records
        ],
        "rounds_per_run": 16,
        "cost_boundary": "measured Pi5+Orin NX active client-device energy; server/AP excluded; intentional cooldown excluded",
        "metrics": {
            "throughput": "total FL input tokens / measured active server seconds",
            "energy_rate": "1000 * measured active client-device joules / total FL input tokens",
        },
        "policy_definitions": {
            "F": "fixed observed round-4 quota",
            "A": "frozen NX cost estimate; changing Pi reference and window search",
            "B": "latest observed NX step rate",
            "C": "original LoRACast-NR online-cost policy",
        },
        "run_order_and_overlap_limitation": "same finite 1 Hz inference request trace, but actual overlap and ambient conditions varied; no causal online-alpha claim",
        "negative_result_by_seed": {
            str(seed): {
                "fixed_vs_online_tokens_per_active_s": next(float(r["tokens_per_active_s"]) for r in records if r["seed"] == seed and r["policy"] == "F") - next(float(r["tokens_per_active_s"]) for r in records if r["seed"] == seed and r["policy"] == "C"),
                "frozen_vs_online_tokens_per_active_s": next(float(r["tokens_per_active_s"]) for r in records if r["seed"] == seed and r["policy"] == "A") - next(float(r["tokens_per_active_s"]) for r in records if r["seed"] == seed and r["policy"] == "C"),
                "online_minus_fixed_j_per_1k_tokens": next(float(r["j_per_1k_input_tokens"]) for r in records if r["seed"] == seed and r["policy"] == "C") - next(float(r["j_per_1k_input_tokens"]) for r in records if r["seed"] == seed and r["policy"] == "F"),
                "online_minus_frozen_j_per_1k_tokens": next(float(r["j_per_1k_input_tokens"]) for r in records if r["seed"] == seed and r["policy"] == "C") - next(float(r["j_per_1k_input_tokens"]) for r in records if r["seed"] == seed and r["policy"] == "A"),
            }
            for seed in (43, 44)
        },
        "display_note": "bars are two-seed means; circle and square are original seed-43/44 run values",
        "intended_size_cm": [17.8, 5.1], "formats": ["PDF", "SVG", "PNG_600dpi"],
    }
    (VERIFY / "fig5_s1_pi5_nx_ablation_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "runs": len(records), "output": str(SUPP / f"{stem}.pdf")}, indent=2))


if __name__ == "__main__":
    main()
