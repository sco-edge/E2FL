#!/usr/bin/env python3
"""Rebuild single-panel WWW Fig. 6 from verified observations.

No training or model scoring occurs here. The checkpoint alignment is checked
against the hardware per-round record, checkpoint-score record, and policy
summary when the campaign files are available. Previously verified checkpoint
rows are retained as a local fallback when cloud-backed inputs are dataless.
Only round-matched seed means are drawn; seed observations remain in the audit.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT.parent / "exposure_policy/nano_nx_independent_primary_loss.csv"
BF16_JSON = ROOT.parent / "exposure_policy/raw/nano_nx_independent_bf16"
SOURCE = Path("/Users/jang-won-u/Documents/Codex/2026-10-10/new-chat/outputs/learning_efficiency_confirmation")
ALIGNMENT = SOURCE / "checkpoint_time_energy_alignment.csv"
LOSS = SOURCE / "checkpoint_validation_loss.csv"
ROUNDS = SOURCE / "measurements/per_round_results.csv"
SUMMARY = SOURCE / "measurements/policy_summary.csv"
QC = SOURCE / "analysis_qc.json"
HARDWARE_QC = SOURCE / "hardware_integrity_report.md"

POLICIES = ("U", "F", "P", "C")
NAMES = {"U": "FedAvg", "F": "Fixed-Hetero", "P": "PyramidFL-style", "C": "LoRACast"}
# Shared WWW policy encoding. Shape and line style remain distinct in grayscale.
STYLE = {
    "U": {"color": "#7F7F7F", "linestyle": "-", "marker": "o", "linewidth": 1.35},
    "F": {"color": "#4D4D4D", "linestyle": "--", "marker": "s", "linewidth": 1.35},
    "P": {"color": "#CC79A7", "linestyle": "-.", "marker": "D", "linewidth": 1.35},
    "C": {"color": "#0072B2", "linestyle": "-", "marker": "^", "linewidth": 1.8},
}
SEEDS = (43, 44, 45)
FULL_ROUNDS = tuple(range(13))
CACHED_ROUNDS = tuple(range(5, 13))


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, fieldnames: list[str], data: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def check_close(actual: float, expected: float, *, label: str, atol: float = 1e-5) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=atol):
        raise AssertionError(f"{label}: {actual} != {expected}")


def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_materialized(path: Path) -> bool:
    """Do not block on a macOS File Provider dataless placeholder."""
    return path.is_file() and not (getattr(os.stat(path), "st_flags", 0) & getattr(stat, "SF_DATALESS", 0))


def load_style() -> None:
    # Matched to the existing EEFL paper_style() sizing and embedded-font setup.
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
        "axes.linewidth": .8, "xtick.major.width": .8, "ytick.major.width": .8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "savefig.bbox": None,
    })


def validate_and_load() -> tuple[dict, dict, list[dict]]:
    if json.loads(QC.read_text())["status"] != "verified":
        raise AssertionError("source campaign analysis is not verified")
    if "All 12 seed-policy runs" not in HARDWARE_QC.read_text():
        raise AssertionError("hardware integrity report does not cover all runs")
    aligned = rows(ALIGNMENT)
    losses = rows(LOSS)
    hardware = rows(ROUNDS)
    summaries = rows(SUMMARY)
    if len(aligned) != 156 or len(losses) != 156 or len(hardware) != 144 or len(summaries) != 12:
        raise AssertionError("expected 3 seeds x 4 methods x (13 checkpoints, 12 rounds)")

    key = lambda row: (int(row["seed"]), row["policy"], int(row["round"]))
    aligned_by_key = {key(row): row for row in aligned}
    loss_by_key = {key(row): row for row in losses}
    hardware_by_key = {key(row): row for row in hardware}
    summary_by_key = {(int(row["scenario"].split("_")[0].replace("seed", "")), row["policy"]): row for row in summaries}
    if len(aligned_by_key) != 156 or len(loss_by_key) != 156 or len(hardware_by_key) != 144 or len(summary_by_key) != 12:
        raise AssertionError("duplicate seed/policy/round keys")

    for seed in SEEDS:
        for policy in POLICIES:
            cumulative_time = cumulative_energy = cumulative_tokens = 0.0
            run_ids = set()
            for round_no in range(13):
                k = (seed, policy, round_no)
                a = aligned_by_key[k]
                l = loss_by_key[k]
                run_ids.add(a["run_id"])
                if a["checkpoint_sha256"] != l["checkpoint_sha256"]:
                    raise AssertionError(f"checkpoint hash mismatch: {k}")
                if a["checkpoint_source"] != l["checkpoint_source"]:
                    raise AssertionError(f"checkpoint source mismatch: {k}")
                check_close(float(a["full_sequence_nll_bf16"]), float(l["full_sequence_nll_bf16"]), label=f"BF16 NLL {k}", atol=1e-12)
                if a["loss_source_type"] != "actual RTX checkpoint evaluation" or a["cost_source_type"] != "actual Nano+NX hardware measurement":
                    raise AssertionError(f"source measurement category mismatch: {k}")
                if not (SOURCE / a["checkpoint_source"]).is_file():
                    raise AssertionError(f"checkpoint file missing: {k}")
                if round_no:
                    h = hardware_by_key[k]
                    if a["run_id"] != h["run_id"]:
                        raise AssertionError(f"hardware run ID mismatch: {k}")
                    cumulative_time += float(h["server_active_round_s"])
                    cumulative_energy += float(h["client_device_active_energy_j"])
                    cumulative_tokens += float(h["aggregate_input_tokens"])
                check_close(float(a["cumulative_active_time_s"]), cumulative_time, label=f"cumulative active time {k}")
                check_close(float(a["cumulative_client_energy_j"]), cumulative_energy, label=f"cumulative client energy {k}")
                check_close(float(a["cumulative_processed_input_tokens"]), cumulative_tokens, label=f"cumulative tokens {k}")
                if len(a["checkpoint_sha256"]) != 64:
                    raise AssertionError(f"checkpoint digest invalid: {k}")
            if len(run_ids) != 1:
                raise AssertionError(f"inconsistent run IDs: {seed}/{policy}")
            s = summary_by_key[(seed, policy)]
            check_close(float(s["server_active_time_s"]), cumulative_time, label=f"run total active time {seed}/{policy}")
            check_close(float(s["client_device_active_energy_j"]), cumulative_energy, label=f"run total client energy {seed}/{policy}")
            check_close(float(s["aggregate_input_tokens"]), cumulative_tokens, label=f"run total tokens {seed}/{policy}")
            check_close(float(s["tokens_per_active_s"]), cumulative_tokens / cumulative_time, label=f"throughput {seed}/{policy}")
            check_close(float(s["j_per_1k_tokens"]), 1000 * cumulative_energy / cumulative_tokens, label=f"J/1K tokens {seed}/{policy}")

    expected_r12 = {"U": 1.53356, "F": 1.53235, "P": 1.53132, "C": 1.52988}
    for policy, target in expected_r12.items():
        observed = mean(float(aligned_by_key[(seed, policy, 12)]["full_sequence_nll_bf16"]) for seed in SEEDS)
        check_close(observed, target, label=f"R12 mean {policy}", atol=5e-6)
    for policy, expected_rate, expected_j in (("U", 206.46, 111.45), ("C", 230.65, 107.19)):
        observed_rate = mean(float(summary_by_key[(seed, policy)]["tokens_per_active_s"]) for seed in SEEDS)
        observed_j = mean(float(summary_by_key[(seed, policy)]["j_per_1k_tokens"]) for seed in SEEDS)
        check_close(observed_rate, expected_rate, label=f"run-wide rate {policy}", atol=.005)
        check_close(observed_j, expected_j, label=f"run-wide J/1K {policy}", atol=.005)
    return aligned_by_key, summary_by_key, aligned


def load_cached_observations() -> tuple[dict, dict, list[dict]]:
    """Recover the verified R5–R12 observations without inventing R0–R4."""
    archive = rows(ROOT / "verification/fig6_plot_data.csv")
    aligned: dict[tuple[int, str, int], dict[str, str]] = {}
    for panel_name, key, scale, unit in (
        ("time", "cumulative_active_time_s", 60.0, "min"),
        ("energy", "cumulative_client_energy_j", 1000.0, "kJ"),
    ):
        observed = [r for r in archive if r["panel"] == panel_name and
                    r["record_type"] == "observed_seed_checkpoint" and int(r["round"]) in CACHED_ROUNDS]
        if len(observed) != len(SEEDS) * len(POLICIES) * len(CACHED_ROUNDS):
            raise AssertionError(f"cached {panel_name} observations incomplete")
        seen = set()
        for row in observed:
            seed, policy, round_no = int(row["seed"]), row["policy_code"], int(row["round"])
            k = (seed, policy, round_no)
            if seed not in SEEDS or policy not in POLICIES or round_no not in CACHED_ROUNDS or k in seen:
                raise AssertionError(f"cached {panel_name} observation key invalid or duplicate: {k}")
            if row["cost_unit"] != unit or row["method"] != NAMES[policy]:
                raise AssertionError(f"cached {panel_name} unit or policy mismatch: {k}")
            seen.add(k)
            rec = aligned.setdefault(k, {
                "seed": str(seed), "policy": policy, "round": str(round_no),
                "run_id": row["run_id"], "checkpoint_sha256": row["checkpoint_sha256"],
                "checkpoint_source": row["checkpoint_source"],
                "full_sequence_nll_bf16": row["BF16_full_sequence_NLL"],
            })
            for field, expected in (("run_id", row["run_id"]),
                                    ("checkpoint_sha256", row["checkpoint_sha256"]),
                                    ("checkpoint_source", row["checkpoint_source"])):
                if rec[field] != expected:
                    raise AssertionError(f"cached panels disagree on {field}: {k}")
            check_close(float(rec["full_sequence_nll_bf16"]), float(row["BF16_full_sequence_NLL"]),
                        label=f"cached NLL {k}", atol=1e-11)
            rec[key] = str(float(row["cumulative_cost"]) * scale)
        if len(seen) != len(observed):
            raise AssertionError(f"cached {panel_name} has duplicate observations")
    if len(aligned) != len(SEEDS) * len(POLICIES) * len(CACHED_ROUNDS):
        raise AssertionError("time and energy checkpoint keys do not align")

    summary_rows = rows(ROOT / "verification/fig6_seed_run_summary.csv")
    if len(summary_rows) != len(SEEDS) * len(POLICIES):
        raise AssertionError("cached seed summary incomplete")
    summaries = {}
    for row in summary_rows:
        seed, policy = int(row["seed"]), row["policy_code"]
        k = (seed, policy)
        if seed not in SEEDS or policy not in POLICIES or k in summaries or row["method"] != NAMES[policy]:
            raise AssertionError(f"invalid or duplicate seed summary: {k}")
        final = aligned[(seed, policy, 12)]
        check_close(float(final["full_sequence_nll_bf16"]), float(row["R12_validation_NLL"]),
                    label=f"cached R12 NLL {k}", atol=1e-10)
        check_close(float(final["cumulative_active_time_s"]) / 60, float(row["total_active_time_min"]),
                    label=f"cached R12 active time {k}", atol=1e-9)
        check_close(float(final["cumulative_client_energy_j"]) / 1000, float(row["total_measured_client_energy_kJ"]),
                    label=f"cached R12 client energy {k}", atol=1e-9)
        if final["run_id"] != row["run_id"]:
            raise AssertionError(f"cached R12 run ID mismatch: {k}")
        summaries[k] = {
            "tokens_per_active_s": row["tokens_per_active_s"],
            "j_per_1k_tokens": row["J_per_1K_input_tokens"],
            "server_active_time_s": str(float(row["total_active_time_min"]) * 60),
            "client_device_active_energy_j": str(float(row["total_measured_client_energy_kJ"]) * 1000),
        }
    return aligned, summaries, archive


def load_local_primary() -> tuple[dict, dict, list[dict], dict]:
    """Verify the complete local checkpoint/cost join and its BF16 scores."""
    cached, summaries, _ = load_cached_observations()
    primary_rows = rows(PRIMARY)
    if len(primary_rows) != len(SEEDS) * len(POLICIES) * len(FULL_ROUNDS):
        raise AssertionError("local independent primary source is not 3 seeds x 4 policies x R0–R12")
    aligned: dict[tuple[int, str, int], dict[str, str]] = {}
    by_hash: dict[str, list[dict[str, str]]] = {}
    for row in primary_rows:
        seed, policy, round_no = int(row["seed"]), row["policy"], int(row["round"])
        k = (seed, policy, round_no)
        if seed not in SEEDS or policy not in POLICIES or round_no not in FULL_ROUNDS or k in aligned:
            raise AssertionError(f"invalid or duplicate local primary key: {k}")
        if row["policy_name"] != ("LoRACast-NR" if policy == "C" else NAMES[policy]):
            raise AssertionError(f"policy name mismatch: {k}")
        digest = row["checkpoint_sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise AssertionError(f"invalid checkpoint digest: {k}")
        nll = float(row["full_sequence_nll_bf16_l64"])
        active = float(row["cumulative_measured_active_s"])
        energy = float(row["cumulative_measured_client_j"])
        if not all(math.isfinite(value) for value in (nll, active, energy)) or active < 0 or energy < 0:
            raise AssertionError(f"invalid local primary values: {k}")
        if round_no == 0 and (active != 0 or energy != 0):
            raise AssertionError(f"R0 should have zero cumulative measured cost: {k}")
        aligned[k] = {
            "seed": str(seed), "policy": policy, "round": str(round_no),
            "run_id": row["run_id"], "checkpoint_sha256": digest,
            "checkpoint_source": str(BF16_JSON / (digest + ".json")),
            "full_sequence_nll_bf16": row["full_sequence_nll_bf16_l64"],
            "cumulative_active_time_s": row["cumulative_measured_active_s"],
            "cumulative_client_energy_j": row["cumulative_measured_client_j"],
        }
        by_hash.setdefault(digest, []).append(row)
        if k in cached:
            old = cached[k]
            for field in ("run_id", "checkpoint_sha256"):
                if aligned[k][field] != old[field]:
                    raise AssertionError(f"archived R5–R12 {field} differs from primary: {k}")
            for field in ("full_sequence_nll_bf16", "cumulative_active_time_s", "cumulative_client_energy_j"):
                check_close(float(aligned[k][field]), float(old[field]),
                            label=f"archived R5–R12 {field} {k}", atol=1e-6)
    if len(aligned) != len(primary_rows):
        raise AssertionError("local primary keys incomplete")

    score_protocols = set()
    score_file_hashes = []
    for digest, references in sorted(by_hash.items()):
        path = BF16_JSON / (digest + ".json")
        if not is_materialized(path):
            raise AssertionError(f"BF16 scorer JSON missing: {digest}")
        score = json.loads(path.read_text())
        if score["checkpoint_sha256"] != digest:
            raise AssertionError(f"BF16 scorer JSON checkpoint digest mismatch: {digest}")
        score_protocols.add((score["precision"], score["batch_size"], score["max_seq_len"],
                             score["dataset_sha256"], score["holdout_indices_sha256"],
                             score["validation_reference_manifest_sha256"]))
        scored_identity = (str(score["seed"]), score["policy"], str(score["round"]), score["run_id"])
        reference_identities = {(r["seed"], r["policy"], r["round"], r["run_id"]) for r in references}
        if scored_identity not in reference_identities:
            raise AssertionError(f"BF16 scorer JSON does not belong to the shared checkpoint references: {digest}")
        for row in references:
            if str(score["seed"]) != row["seed"] or str(score["round"]) != row["round"]:
                raise AssertionError(f"BF16 scorer JSON seed/round mismatch: {digest}")
            check_close(float(row["full_sequence_nll_bf16_l64"]),
                        float(score["full_sequence_balanced_global_nll"]),
                        label=f"balanced BF16 score {digest}", atol=1e-12)
        score_file_hashes.append(f"{digest}:{source_hash(path)}")
    if len(score_protocols) != 1:
        raise AssertionError("mixed BF16 evaluation protocols across independent checkpoints")
    precision, batch, seq_len, dataset_sha, holdout_sha, manifest_sha = next(iter(score_protocols))
    if (precision, batch, seq_len) != ("bf16", 32, 64):
        raise AssertionError("unexpected BF16 full-sequence evaluation settings")

    for seed in SEEDS:
        for policy in POLICIES:
            ordered = [aligned[(seed, policy, round_no)] for round_no in FULL_ROUNDS]
            if len({r["run_id"] for r in ordered}) != 1:
                raise AssertionError(f"mixed run IDs: {seed}/{policy}")
            for left, right in zip(ordered, ordered[1:]):
                if float(right["cumulative_active_time_s"]) <= float(left["cumulative_active_time_s"]) or \
                   float(right["cumulative_client_energy_j"]) <= float(left["cumulative_client_energy_j"]):
                    raise AssertionError(f"non-increasing cumulative measured cost: {seed}/{policy}")
            final = ordered[-1]
            summ = summaries[(seed, policy)]
            check_close(float(final["cumulative_active_time_s"]), float(summ["server_active_time_s"]),
                        label=f"local primary R12 active time {seed}/{policy}", atol=1e-6)
            check_close(float(final["cumulative_client_energy_j"]), float(summ["client_device_active_energy_j"]),
                        label=f"local primary R12 client energy {seed}/{policy}", atol=1e-6)

    metadata = {
        "unique_checkpoint_sha256_count": len(by_hash),
        "BF16_scorer_json_count": len(score_file_hashes),
        "BF16_scorer_json_hash_manifest_sha256": hashlib.sha256("\n".join(score_file_hashes).encode()).hexdigest(),
        "BF16_protocol": {"precision": precision, "batch_size": batch, "max_seq_len": seq_len,
                          "dataset_sha256": dataset_sha, "holdout_indices_sha256": holdout_sha,
                          "validation_reference_manifest_sha256": manifest_sha},
        "R5_R12_archived_observations_crosschecked": len(cached),
    }
    return aligned, summaries, primary_rows, metadata


def make_table(aligned: dict, summaries: dict) -> list[dict]:
    result = []
    for policy in POLICIES:
        nll = [float(aligned[(seed, policy, 12)]["full_sequence_nll_bf16"]) for seed in SEEDS]
        rate = [float(summaries[(seed, policy)]["tokens_per_active_s"]) for seed in SEEDS]
        joules_per_k = [float(summaries[(seed, policy)]["j_per_1k_tokens"]) for seed in SEEDS]
        time_min = [float(summaries[(seed, policy)]["server_active_time_s"]) / 60 for seed in SEEDS]
        energy_kj = [float(summaries[(seed, policy)]["client_device_active_energy_j"]) / 1000 for seed in SEEDS]
        result.append({
            "method": NAMES[policy], "seed_count": len(SEEDS),
            "R12_validation_NLL_mean": f"{mean(nll):.5f}", "R12_validation_NLL_SD": f"{stdev(nll):.5f}",
            "tokens_per_active_s_mean": f"{mean(rate):.2f}", "tokens_per_active_s_SD": f"{stdev(rate):.2f}",
            "J_per_1K_input_tokens_mean": f"{mean(joules_per_k):.2f}", "J_per_1K_input_tokens_SD": f"{stdev(joules_per_k):.2f}",
            "total_active_time_min_mean": f"{mean(time_min):.2f}", "total_active_time_min_SD": f"{stdev(time_min):.2f}",
            "total_measured_client_energy_kJ_mean": f"{mean(energy_kj):.2f}",
            "total_measured_client_energy_kJ_SD": f"{stdev(energy_kj):.2f}",
        })
    return result


def panel(ax, aligned: dict, *, cost: str, selected_rounds: tuple[int, ...]) -> list[dict]:
    if cost == "time":
        key, scale, label = "cumulative_active_time_s", 1 / 60, "Cumulative server active time (min)"
    elif cost == "energy":
        key, scale, label = "cumulative_client_energy_j", 1 / 1000, "Cumulative client energy (kJ)"
    else:
        raise ValueError(cost)
    plotted = []
    for policy in POLICIES:
        for round_no in selected_rounds:
            for seed in SEEDS:
                rec = aligned[(seed, policy, round_no)]
                xx = float(rec[key]) * scale
                yy = float(rec["full_sequence_nll_bf16"])
                plotted.append({"panel": cost, "record_type": "observed_seed_checkpoint", "policy_code": policy,
                                "method": NAMES[policy], "seed": seed, "round": round_no,
                                "cumulative_cost": f"{xx:.12g}", "cost_unit": "min" if cost == "time" else "kJ",
                                "BF16_full_sequence_NLL": f"{yy:.12g}", "run_id": rec["run_id"],
                                "checkpoint_sha256": rec["checkpoint_sha256"], "checkpoint_source": rec["checkpoint_source"]})
        x_mean = []
        y_mean = []
        for round_no in selected_rounds:
            selected = [aligned[(seed, policy, round_no)] for seed in SEEDS]
            xx = mean(float(r[key]) * scale for r in selected)
            yy = mean(float(r["full_sequence_nll_bf16"]) for r in selected)
            x_mean.append(xx)
            y_mean.append(yy)
            plotted.append({"panel": cost, "record_type": "round_matched_seed_mean", "policy_code": policy,
                            "method": NAMES[policy], "seed": "43|44|45", "round": round_no,
                            "cumulative_cost": f"{xx:.12g}", "cost_unit": "min" if cost == "time" else "kJ",
                            "BF16_full_sequence_NLL": f"{yy:.12g}", "run_id": "mean of 3 run IDs",
                            "checkpoint_sha256": "mean of 3 observed checkpoint scores",
                            "checkpoint_source": "round-matched mean of listed seed observations"})
        visual = STYLE[policy]
        curve_start = 1 if selected_rounds[0] == 0 else 0
        curve_x, curve_y = x_mean[curve_start:], y_mean[curve_start:]
        marker_round_indices = list(range(0, len(curve_x), 2))
        if len(curve_x) - 1 not in marker_round_indices:
            marker_round_indices.append(len(curve_x) - 1)
        ax.plot(curve_x, curve_y, label=NAMES[policy], color=visual["color"],
                linestyle=visual["linestyle"], marker=visual["marker"],
                markersize=3.8, markerfacecolor="white", markeredgewidth=.85,
                markevery=marker_round_indices, linewidth=visual["linewidth"],
                zorder=4 if policy == "C" else 3)
        if curve_start:
            ax.plot(x_mean[0], y_mean[0], linestyle="none", marker=visual["marker"],
                    markersize=3.8, color=visual["color"], markerfacecolor="white",
                    markeredgewidth=.85, zorder=4 if policy == "C" else 3)
        if selected_rounds[-1] == 12:
            ax.plot(x_mean[-1], y_mean[-1], linestyle="none", marker=visual["marker"],
                    markersize=4.5, color=visual["color"], markerfacecolor=visual["color"],
                    markeredgecolor="white", markeredgewidth=.45, zorder=6)
    ax.set_xlabel(label)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=.8, length=3)
    return plotted


def set_full_round_nll_scale(ax) -> None:
    """Show observed R0 with an explicitly marked break above R1–R12."""
    low, high, compressed_gap, upper_scale = 1.59, 2.08, .0025, .4

    def forward(value):
        value = np.asarray(value)
        return np.where(value <= low, value,
                        np.where(value >= high, low + compressed_gap + (value - high) * upper_scale,
                                 low + (value - low) * compressed_gap / (high - low)))

    def inverse(value):
        value = np.asarray(value)
        return np.where(value <= low, value,
                        np.where(value >= low + compressed_gap,
                                 high + (value - low - compressed_gap) / upper_scale,
                                 low + (value - low) * (high - low) / compressed_gap))

    ax.set_yscale("function", functions=(forward, inverse))
    ax.set_ylim(1.525, 2.105)
    ax.set_yticks([1.53, 1.54, 1.55, 1.56, 1.57, 1.58, 2.09])
    ax.set_yticklabels(["1.53", "1.54", "1.55", "1.56", "1.57", "1.58", "2.09"])
    break_y = float((forward((low + high) / 2) - forward(1.525)) /
                    (forward(2.105) - forward(1.525)))
    for x in (0, 1):
        ax.plot([x - .012, x + .012], [break_y - .012, break_y + .012],
                transform=ax.transAxes, color="black", linewidth=.8, clip_on=False)


def save_all(fig, stem: str) -> None:
    base = ROOT / "main" / stem
    fig.savefig(str(base) + ".svg", bbox_inches=None)
    fig.savefig(str(base) + ".pdf", bbox_inches=None)
    fig.savefig(str(base) + ".png", dpi=600, bbox_inches=None)
    plt.close(fig)


def build_figures(aligned: dict, selected_rounds: tuple[int, ...]) -> list[dict]:
    plot_rows = []
    for cost, stem in (("time", "fig6a_learning_vs_active_time"),
                       ("energy", "fig6b_learning_vs_client_energy")):
        single, ax = plt.subplots(figsize=(8.6 / 2.54, 5.2 / 2.54))
        single.subplots_adjust(left=.205, right=.975, bottom=.255, top=.745)
        panel_rows = panel(ax, aligned, cost=cost, selected_rounds=selected_rounds)
        plot_rows.extend(panel_rows)
        ax.set_ylabel("Validation NLL")
        means = [row for row in panel_rows if row["record_type"] == "round_matched_seed_mean"]
        xmin = min(float(row["cumulative_cost"]) for row in means)
        xmax = max(float(row["cumulative_cost"]) for row in means)
        xpad = .04 * (xmax - xmin)
        ax.set_xlim(xmin - xpad, xmax + xpad)
        if selected_rounds[0] == 0:
            set_full_round_nll_scale(ax)
            ax.annotate("R0", (0, mean(float(aligned[(seed, "U", 0)]["full_sequence_nll_bf16"])
                                       for seed in SEEDS)), xytext=(5, -1), textcoords="offset points",
                        fontsize=7.5, va="center")
        else:
            ymin = min(float(row["BF16_full_sequence_NLL"]) for row in means)
            ymax = max(float(row["BF16_full_sequence_NLL"]) for row in means)
            ax.set_ylim(ymin - .0018, ymax + .0018)
            ax.yaxis.set_major_locator(MultipleLocator(.005))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        h, lab = ax.get_legend_handles_labels()
        # Matplotlib fills legend columns before rows. Reorder to read U, F, P, C.
        order = (0, 2, 1, 3)
        single.legend([h[i] for i in order], [lab[i] for i in order], ncol=2,
                      loc="upper center", bbox_to_anchor=(.52, .995), frameon=False,
                      handlelength=2.0, columnspacing=.9)
        save_all(single, stem)
    write_csv(ROOT / "verification/fig6_plot_data.csv", list(plot_rows[0]), plot_rows)
    return plot_rows


def main() -> None:
    load_style()
    input_paths = (ALIGNMENT, LOSS, ROUNDS, SUMMARY, QC, HARDWARE_QC)
    prior_audit_path = ROOT / "verification/fig6_validation.json"
    prior_audit = json.loads(prior_audit_path.read_text()) if prior_audit_path.is_file() else {}
    score_metadata = {}
    if is_materialized(PRIMARY):
        aligned, summary, _, score_metadata = load_local_primary()
        selected_rounds = FULL_ROUNDS
        source_mode = "workspace_independent_primary_156_rows"
        input_hashes = {str(PRIMARY): source_hash(PRIMARY)}
        source_hash_status = "local primary and all scorer JSON files verified now"
    elif all(is_materialized(path) for path in input_paths):
        aligned, summary, _ = validate_and_load()
        selected_rounds = FULL_ROUNDS
        source_mode = "complete_original_campaign"
        input_hashes = {str(path): source_hash(path) for path in input_paths}
        source_hash_status = "reverified_now"
    else:
        aligned, summary, _ = load_cached_observations()
        selected_rounds = CACHED_ROUNDS
        source_mode = "previously_verified_R5_R12_checkpoint_archive"
        input_hashes = prior_audit.get("input_source_sha256", {})
        source_hash_status = "copied_from_prior_audit; cloud sources not re-read"
    table = make_table(aligned, summary)
    plotted = build_figures(aligned, selected_rounds)
    mean_rows = [row for row in plotted if row["record_type"] == "round_matched_seed_mean"]
    r12_points = []
    for panel_name, cost_key in (("time", "total_active_time_min_mean"),
                                 ("energy", "total_measured_client_energy_kJ_mean")):
        for policy in POLICIES:
            point = next(row for row in mean_rows if row["panel"] == panel_name and
                         row["policy_code"] == policy and row["round"] == 12)
            tab = next(row for row in table if row["method"] == NAMES[policy])
            check_close(float(point["BF16_full_sequence_NLL"]), float(tab["R12_validation_NLL_mean"]),
                        label=f"Fig. 6/Table 1 R12 NLL {panel_name}/{policy}", atol=5.1e-6)
            check_close(float(point["cumulative_cost"]), float(tab[cost_key]),
                        label=f"Fig. 6/Table 1 R12 cost {panel_name}/{policy}", atol=.0051)
            r12_points.append({"panel": panel_name, "policy_code": policy,
                               "mean_cost": float(point["cumulative_cost"]),
                               "mean_NLL": float(point["BF16_full_sequence_NLL"])})
    font = font_manager.findfont(font_manager.FontProperties(family="Times New Roman"))
    audit = {
        "status": "PASS" if selected_rounds == FULL_ROUNDS else "PARTIAL_SOURCE_UNAVAILABLE",
        "figure": "Fig. 6", "seed_ids": list(SEEDS), "rounds_plotted": list(selected_rounds),
        "rounds_not_available_in_local_archive": [] if selected_rounds == FULL_ROUNDS else [0, 1, 2, 3, 4],
        "source_mode": source_mode,
        "source_limitation": None if selected_rounds == FULL_ROUNDS else
            "Original campaign CSV files are macOS dataless cloud placeholders. R0–R4 were not estimated or plotted.",
        "single_panel_outputs": [
            str(ROOT / "main/fig6a_learning_vs_active_time.pdf"),
            str(ROOT / "main/fig6b_learning_vs_client_energy.pdf"),
        ],
        "intended_panel_size_cm": [8.6, 5.2],
        "policy_codes": list(POLICIES), "metric": "BF16 batch-32 full-sequence 64-token balanced validation NLL",
        "data_points": {"per_panel_observed_checkpoints": 3 * 4 * len(selected_rounds),
                        "per_panel_plotted_round_matched_means": 4 * len(selected_rounds)},
        "averaging": "For each policy and round, arithmetic mean of three cumulative costs and arithmetic mean of the three NLLs; no equal-budget interpolation.",
        "y_axis": {"type": "single plotting area with explicitly marked break",
                   "break_interval_NLL": [1.59, 2.08],
                   "R0_marked_separately": True, "R0_to_R1_line_segment": False,
                   "R1_R12_band_linear": True},
        "cost_boundary": "Measured Nano Super + Orin NX whole-device VDD_IN during server-active rounds; excludes server, AP, and inter-round cooling",
        "style_source": "local EEFL-compatible Times New Roman settings in this script",
        "policy_visual_encoding": STYLE,
        "resolved_Times_New_Roman_font": font,
        "input_source_sha256": input_hashes, "source_hash_status": source_hash_status,
        "BF16_scorer_verification": score_metadata,
        "prior_original_campaign_source_sha256": prior_audit.get("prior_original_campaign_source_sha256",
                                                            prior_audit.get("input_source_sha256", {})),
        "R12_table": table,
        "R12_mean_points": r12_points,
        "plot_rows": len(plotted),
    }
    prior_audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"status": audit["status"], "source_mode": source_mode,
                      "rounds_plotted": list(selected_rounds), "plot_rows": len(plotted),
                      "R12_mean_points": r12_points}, indent=2))


if __name__ == "__main__":
    main()
