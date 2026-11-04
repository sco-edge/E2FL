#!/usr/bin/env python3
"""Supplementary quota/omission trace from actual D512 Exposure-2 RTX FL.

This visualizes executed samples and cumulative exposure, never requested caps.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_training_budget_policies import INPUT, OUT, load_style, read_jsonl, save, sha256


RUN_ID = "confirm_s43_d512_exposurecap2"
SOURCE = INPUT / "raw/confirmation_runs" / RUN_ID / "rounds.jsonl"
DATASET_SIZE = 512
CAP = 2.0


def main() -> None:
    style_sha = load_style()
    rows = read_jsonl(SOURCE)
    assert len(rows) == 100 and [row["round"] for row in rows] == list(range(1, 101))
    assert all(row["run_id"] == RUN_ID and row["policy"] == "exposurecap2" for row in rows)
    cumulative = 0
    plotted = []
    for row in rows:
        client = row["clients"]["nx"]
        executed = int(client["samples"])
        assert executed == len(client["visited_sample_ids"])
        assert client["optimizer_steps"] == executed
        before = cumulative / DATASET_SIZE
        cumulative += executed
        after = cumulative / DATASET_SIZE
        assert abs(after - row["nx_exposure"]) < 1e-12
        assert after <= CAP + 1e-12
        plotted.append({"round": row["round"], "nx_executed_samples": executed,
                        "nx_optimizer_steps": client["optimizer_steps"],
                        "cumulative_executed_samples": cumulative,
                        "exposure_before": before, "exposure_after": after,
                        "remaining_exposure": CAP - after, "action": row["action"],
                        "run_id": RUN_ID, "measurement": "RTX_actual_completed_work"})
    assert cumulative == 1024
    partial = [row for row in plotted if row["action"] == "cumulative_exposure_cap_2_partial"]
    omitted = [row for row in plotted if row["action"] == "cumulative_exposure_cap_2_below_minimum"]
    assert len(partial) == 1 and partial[0]["round"] == 33 and partial[0]["nx_executed_samples"] == 24
    assert len(omitted) == 67 and omitted[0]["round"] == 34
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    with (OUT / "tables/figS4_quota_omission_trace.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(plotted[0]))
        writer.writeheader()
        writer.writerows(plotted)

    fig, axes = plt.subplots(1, 2, figsize=(17.8 / 2.54, 5.1 / 2.54))
    fig._target_width_cm, fig._target_height_cm = 17.8, 5.1
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.23, top=0.90, wspace=0.23)
    rounds = [row["round"] for row in plotted]
    samples = [row["nx_executed_samples"] for row in plotted]
    axes[0].bar(rounds, samples, width=0.85, facecolor="white", edgecolor="black",
                linewidth=0.9, zorder=3)
    axes[0].bar([partial[0]["round"]], [partial[0]["nx_executed_samples"]],
                width=0.85, facecolor="white", edgecolor="black", hatch="///",
                linewidth=0.9, zorder=4)
    axes[0].axvline(33.5, color="black", linestyle=":", linewidth=1.0)
    axes[0].annotate("partial", xy=(33, 24), xytext=(43, 26),
                     arrowprops={"arrowstyle": "->", "linewidth": 0.8}, ha="left")
    axes[0].set_ylabel("Executed NX samples")
    axes[0].set_ylim(0, 39)

    axes[1].plot(rounds, [row["exposure_after"] for row in plotted],
                 color="black", linestyle="-", linewidth=1.4, label="Used")
    axes[1].plot(rounds, [row["remaining_exposure"] for row in plotted],
                 color="#666666", linestyle="--", linewidth=1.4, label="Remaining")
    axes[1].axhline(CAP, color="black", linestyle=":", linewidth=0.9)
    axes[1].axvline(33.5, color="black", linestyle=":", linewidth=1.0)
    axes[1].set_ylabel("Cumulative exposure")
    axes[1].set_ylim(0, 2.12)
    axes[1].legend(loc="center right", frameon=False)
    for ax in axes:
        ax.set_xlim(0, 100)
        ax.set_xticks((0, 20, 40, 60, 80, 100))
        ax.set_xlabel("FL round")
        ax.grid(axis="y", color="black", linestyle="--", linewidth=0.5, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    target = OUT / "supplementary/figS4_d512_exposure2_quota_omission"
    save(fig, target)
    (OUT / "verification").mkdir(parents=True, exist_ok=True)
    support = {"figure": "S4", "run_id": RUN_ID, "seed": 43,
               "policy_version": "October RTX NR allocation with pretraining Exposure-2 cap",
               "source": str(SOURCE.resolve()), "source_sha256": sha256(SOURCE),
               "style_source_sha256": style_sha, "model": "Qwen2.5-0.5B-Instruct",
               "metric": "actual executed NX samples and cumulative executed samples / 512",
               "partial_round": 33, "omitted_rounds": 67, "first_omission_round": 34,
               "final_exposure": CAP, "source_rows": 100,
               "output_base": str(target.resolve())}
    (OUT / "verification/figS4_source_evidence.json").write_text(json.dumps(support, indent=2) + "\n")


if __name__ == "__main__":
    main()
