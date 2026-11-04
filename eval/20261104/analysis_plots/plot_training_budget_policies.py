#!/usr/bin/env python3
"""WWW Fig. 8 from verified October RTX independent FL trajectories.

No training, scoring, extrapolation, or checkpoint interpolation is performed.
This script selects explicit run IDs from the frozen experiment manifest and
cross-checks each plotted row against its original validation.jsonl record.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[3]
INPUT = PROJECT / "exposure_policy"
OUT = PROJECT / "analysis_plots"
STYLE_SOURCE = Path("/Users/jang-won-u/Documents/GitHub/EEFL/plot/revision/plot_loracast_revision.py")
SEEDS = (43, 44, 45)
ROUNDS = (0, 10, 20, 30, 40, 41, 42, 50, 60, 70, 80, 90, 100)
POLICIES = ("nr", "full", "exposurecap2")
DISPLAY = {"nr": "Unbounded", "full": "Rebalancing", "exposurecap2": "Exposure-2"}
LINE = {
    "nr": {"color": "#111111", "linestyle": "-", "marker": "o"},
    "full": {"color": "#666666", "linestyle": "--", "marker": "s"},
    "exposurecap2": {"color": "#333333", "linestyle": "-.", "marker": "^"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_style() -> str:
    if not STYLE_SOURCE.is_file():
        raise FileNotFoundError(f"INFOCOM reference style unavailable: {STYLE_SOURCE}")
    spec = importlib.util.spec_from_file_location("loracast_revision_style", STYLE_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.paper_style()
    matplotlib.rcParams["svg.fonttype"] = "none"
    # The historical style requests tight cropping; fixed-size paper panels
    # instead retain their specified PDF MediaBox while keeping its typography.
    matplotlib.rcParams["savefig.bbox"] = None
    return sha256(STYLE_SOURCE)


def verify_sources() -> tuple[pd.DataFrame, list[dict], dict]:
    derived_path = INPUT / "long_horizon_learning.csv"
    manifest_path = INPUT / "experiment_manifest.json"
    endpoint_path = INPUT / "independent_seed_results.csv"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["primary_seeds"] == list(SEEDS)
    assert manifest["policy_version"]["full"].startswith("October 2026 RTX W3/theta0.3")
    assert manifest["learning"]["sequence_cap"] == 64
    assert "full-sequence" in manifest["evaluation"]
    registered = {run["run_id"]: run for run in manifest["runs"]}
    derived = pd.read_csv(derived_path)
    endpoints = pd.read_csv(endpoint_path)
    plotted: list[dict] = []
    raw_files: list[dict] = []
    for budget in ("d128", "d512", "dfull"):
        for seed in SEEDS:
            for policy in POLICIES if budget != "dfull" else ("nr", "full"):
                run_id = f"confirm_s{seed}_{budget}_{policy}"
                entry = registered[run_id]
                assert entry["seed"] == seed and entry["policy"] == policy
                assert entry["rounds"] == 100 and entry["scenario_id"] == f"{budget}_stationary_e1_heterogeneous"
                raw_dir = INPUT / "raw/confirmation_runs" / run_id
                validation_path = raw_dir / "validation.jsonl"
                rounds_path = raw_dir / "rounds.jsonl"
                state_path = raw_dir / "state.json"
                raw_validation = read_jsonl(validation_path)
                raw_rounds = read_jsonl(rounds_path)
                state = json.loads(state_path.read_text())
                assert [int(row["round"]) for row in raw_validation] == list(ROUNDS)
                assert [int(row["round"]) for row in raw_rounds] == list(range(1, 101))
                assert state["run_id"] == run_id and state["round"] == 100
                assert all(row["run_id"] == run_id for row in raw_validation + raw_rounds)
                assert all(row["measurement"] == "RTX_actual_model_validation" for row in raw_validation)
                assert all(row["loss_mode"] == "full_sequence_64_token" for row in raw_validation)
                assert all(row["cap_per_view"] == 512 for row in raw_validation)
                assert all(row["experiment_seed"] == seed for row in raw_rounds)
                if policy == "full":
                    omissions = [row for row in raw_rounds if row["action"] == "hard_omission"]
                    assert len(omissions) == 49
                    assert all(row["triggered"] and row["clients"]["nx"]["samples"] == 0 for row in omissions)
                if policy == "exposurecap2":
                    assert raw_rounds[-1]["nx_exposure"] <= 2 + 1e-12
                selected = derived.loc[derived.run_id.eq(run_id)].sort_values("round")
                assert selected.shape[0] == len(ROUNDS)
                assert selected["round"].tolist() == list(ROUNDS)
                assert selected["seed"].tolist() == [seed] * len(ROUNDS)
                assert selected["policy"].tolist() == [policy] * len(ROUNDS)
                assert selected["trajectory_status"].eq("actual_independent_RTX_FL").all()
                assert selected["provenance"].eq("actual_RTX_training_and_scoring").all()
                for original, (_, row) in zip(raw_validation, selected.iterrows()):
                    assert original["checkpoint_sha256"] == row.checkpoint_sha256
                    assert abs(original["balanced_client_loss"] - row.balanced_nll) < 1e-12
                    assert abs(original["global_loss"] - row.global_nll) < 1e-12
                    assert abs(original["pi_loss"] - row.pi_nll) < 1e-12
                    assert abs(original["nx_loss"] - row.nx_nll) < 1e-12
                    local_checkpoint = raw_dir / Path(original["checkpoint"]).name
                    assert local_checkpoint.is_file(), local_checkpoint
                    assert sha256(local_checkpoint) == original["checkpoint_sha256"]
                    plotted.append({
                        "figure": "fig8" if budget != "dfull" else "supplementary_dfull",
                        "budget": budget.upper(), "policy": DISPLAY[policy], "source_policy": policy,
                        "seed": seed, "round": int(row["round"]),
                        "balanced_nll": float(row.balanced_nll),
                        "run_id": run_id, "checkpoint_sha256": row.checkpoint_sha256,
                        "metric": "heldout_full_sequence_64_token_balanced_client_nll",
                        "provenance": "actual_RTX_training_and_scoring",
                    })
                endpoint = endpoints.loc[endpoints.run_id.eq(run_id)]
                assert endpoint.shape[0] == 1
                assert abs(float(endpoint.iloc[0].r100_balanced_nll) - raw_validation[-1]["balanced_client_loss"]) < 1e-12
                raw_files.append({
                    "run_id": run_id, "seed": seed, "budget": budget.upper(), "policy": policy,
                    "validation_jsonl": str(validation_path.resolve()),
                    "validation_sha256": sha256(validation_path),
                    "rounds_jsonl": str(rounds_path.resolve()),
                    "rounds_sha256": sha256(rounds_path),
                    "state_json": str(state_path.resolve()),
                    "state_sha256": sha256(state_path),
                    "checkpoint_count_verified": len(ROUNDS),
                    "checkpoint_sha256_verified": True,
                })
    table = pd.DataFrame(plotted)
    expected = {"D128": {"nr": 3.3744, "full": 2.4006, "exposurecap2": 1.5762},
                "D512": {"nr": 1.8142, "full": 1.5977, "exposurecap2": 1.5669},
                "DFULL": {"nr": 1.5450, "full": 1.5555}}
    for budget, policies in expected.items():
        for policy, rounded_mean in policies.items():
            mean = table.loc[(table.budget == budget) & (table.source_policy == policy) &
                             (table["round"] == 100), "balanced_nll"].mean()
            assert round(mean, 4) == rounded_mean, (budget, policy, mean)
    support = {
        "derived_file": str(derived_path.resolve()), "derived_sha256": sha256(derived_path),
        "endpoint_file": str(endpoint_path.resolve()), "endpoint_sha256": sha256(endpoint_path),
        "experiment_manifest": str(manifest_path.resolve()),
        "experiment_manifest_sha256": sha256(manifest_path),
        "raw_files": raw_files, "selected_raw_checkpoint_count": len(plotted),
        "metric": "64-token full-sequence balanced client validation NLL",
        "model": manifest["learning"]["model"],
        "optimizer": manifest["learning"]["optimizer"],
        "policy_version": manifest["policy_version"],
        "seed_count": 3,
        "dfull_exposure2": "Inactive cap; exact NR alias in the frozen manifest, not an independent run.",
    }
    return table, raw_files, support


def summary_table(table: pd.DataFrame) -> pd.DataFrame:
    summary = (table.groupby(["figure", "budget", "policy", "source_policy", "round"], as_index=False)
               .balanced_nll.agg(["mean", "std", "min", "max", "count"]).reset_index())
    assert summary["count"].eq(3).all()
    return summary


def style_axes(ax: plt.Axes, budget: str, left: bool) -> None:
    ax.set_xlim(0, 100)
    ax.set_xticks((0, 20, 40, 60, 80, 100))
    ax.set_xlabel("FL round")
    if left:
        ax.set_ylabel("Full-sequence validation NLL")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, color="black", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.03, 0.96, budget.upper(), transform=ax.transAxes,
            ha="left", va="top")


def plot_budget(ax: plt.Axes, table: pd.DataFrame, budget: str) -> None:
    for policy in POLICIES if budget != "dfull" else ("nr", "full"):
        subset = table.loc[(table.budget == budget.upper()) & (table.source_policy == policy)]
        for seed in SEEDS:
            individual = subset.loc[subset.seed.eq(seed)].sort_values("round")
            ax.plot(individual["round"], individual.balanced_nll,
                    color=LINE[policy]["color"], linestyle=LINE[policy]["linestyle"],
                    linewidth=0.55, alpha=0.28, zorder=2)
        rows = (subset.groupby("round", as_index=False).balanced_nll.mean()
                .sort_values("round"))
        ax.plot(rows["round"], rows.balanced_nll,
                label=DISPLAY[policy], color=LINE[policy]["color"],
                linestyle=LINE[policy]["linestyle"], marker=LINE[policy]["marker"],
                linewidth=1.4, markersize=3.3, zorder=3)


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert abs(fig.get_size_inches()[0] * 2.54 - fig._target_width_cm) < 1e-9
    assert abs(fig.get_size_inches()[1] * 2.54 - fig._target_height_cm) < 1e-9
    for extension, kwargs in (("pdf", {}), ("svg", {}), ("png", {"dpi": 600})):
        fig.savefig(path.with_suffix("." + extension), bbox_inches=None, **kwargs)
    plt.close(fig)


def make_figures(table: pd.DataFrame) -> list[dict]:
    outputs = []
    fig, axes = plt.subplots(1, 2, figsize=(17.8 / 2.54, 5.1 / 2.54))
    fig._target_width_cm, fig._target_height_cm = 17.8, 5.1
    fig.subplots_adjust(left=0.09, right=0.96, bottom=0.24, top=0.77, wspace=0.19)
    for ax, budget, left in zip(axes, ("d128", "d512"), (True, False)):
        plot_budget(ax, table, budget)
        style_axes(ax, budget, left)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 0.99),
               ncol=3, frameon=False, handlelength=2.6, columnspacing=1.4)
    target = OUT / "main/plot_training_budget_policies"
    save(fig, target)
    outputs.append({"path_base": str(target.resolve()), "width_cm": 17.8, "height_cm": 5.1,
                    "budgets": ["D128", "D512"], "layout": "double_column_2_panel"})
    for suffix, budget in (("a", "d128"), ("b", "d512"), ("S_dfull", "dfull")):
        fig, ax = plt.subplots(figsize=(8.6 / 2.54, 5.2 / 2.54))
        fig._target_width_cm, fig._target_height_cm = 8.6, 5.2
        fig.subplots_adjust(left=0.20, right=0.93, bottom=0.23, top=0.80)
        plot_budget(ax, table, budget)
        style_axes(ax, budget, True)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.36), ncol=3 if budget != "dfull" else 2,
                  frameon=False, handlelength=1.2, handletextpad=0.3, columnspacing=0.5)
        folder = "main" if budget != "dfull" else "supplementary"
        target = OUT / folder / (f"fig8_training_budget_policies_{suffix}" if budget != "dfull"
                                 else "figS_dfull_inactive_exposure_cap")
        save(fig, target)
        outputs.append({"path_base": str(target.resolve()), "width_cm": 8.6, "height_cm": 5.2,
                        "budgets": [budget.upper()], "layout": "single_column"})
    return outputs


def main() -> None:
    style_sha = load_style()
    table, _, support = verify_sources()
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "verification").mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "tables/fig8_plotted_values.csv", index=False)
    summary_table(table).to_csv(OUT / "tables/fig8_mean_values.csv", index=False)
    support["style_source"] = str(STYLE_SOURCE)
    support["style_sha256"] = style_sha
    support["outputs"] = make_figures(table)
    (OUT / "verification/fig8_source_evidence.json").write_text(json.dumps(support, indent=2) + "\n")
    with (OUT / "verification/fig8_data_mapping.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("panel", "source", "source_field", "plotted_field", "transformation", "measurement"))
        for panel, budget in (("8a", "D128"), ("8b", "D512"), ("S-DFull", "DFULL")):
            writer.writerow((panel, "raw/confirmation_runs/confirm_s{43,44,45}_" + budget.lower() +
                             "_{nr,full,exposurecap2}/validation.jsonl", "balanced_client_loss",
                             "balanced_nll", "Per-round arithmetic mean of 3 seed-level observed checkpoints; "
                             "thin lines show each original seed. DFull Cap-2 omitted as exact inactive alias.",
                             "RTX actual held-out full-sequence 64-token NLL"))


if __name__ == "__main__":
    main()
