#!/usr/bin/env python3
"""Reproduce the two single-panel WWW Figure 5 files from verified records.

Panel (a) reports July Pi5/NX and October Nano Super logged local compute-time
prediction errors. The Nano results are separated by the prespecified normal,
contention, and recovery intervals because pooling them hides the transition
error. These are different campaigns, not a controlled device comparison.
Panel (b) shows the actual
optimizer steps executed in independent Nano/NX seeds 43--45.

By default, render from the frozen, SHA-checked plot-value CSVs. Pass
--verify-nano-raw to recheck the Nano decision JSON and original Flower CSV;
--verify-july rechecks the July CSVs. Pass --from-raw to repeat the original
NX-step raw Flower cross-check for panel (b).
No training is launched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


HERE = Path(__file__).resolve().parent
OUT = HERE.parent
# Archived Fig. 5 revision. The current three-panel Fig. 5 suite is generated
# by plot_estimator_contention.py; keep this script reproducible without
# letting it overwrite the current main-paper exports.
MAIN = OUT / "comparison" / "superseded_fig5"
VERIFY = OUT / "verification"
STYLE = Path(
    "/Users/jang-won-u/Documents/GitHub/EEFL/plot/revision/"
    "plot_loracast_revision.py"
)
STYLE_SHA256_FROM_PRIOR_AUDIT = "d93dd049968a0e06de0acde9ed302588f5f3d345cc3d07145d107affd3af9707"
EEFL = Path("/Users/jang-won-u/Documents/GitHub/EEFL")
JULY = EEFL / "E2FL/eval/evaluate/20260511"
HISTORICAL = {
    "MAXN": JULY / "20. power (LoRACast-Online_MAXN)/bkp/"
    "fedavg_2026-07-08_22-47-51_token_pred_updated.csv",
    "15W": JULY / "22. power (LoRACast-Online_15W)/"
    "fedavg_2026-07-13_01-33-10.csv",
    "10W": JULY / "23. power (LoRACast-Online_10W)/"
    "fedavg_2026-07-12_17-19-03.csv",
}
NANO_SOURCE = OUT.parent / "exposure_policy/raw/hardware_seed42/estimator_predictions.csv"
NANO_SOURCE_SHA256 = "3a554be9291d79fbae20526f33cafa208bbec933908b29262f12a89312b900df"
RAW_NANO_RUN = Path(
    "/Users/jang-won-u/Documents/Codex/2026-10-10/new-chat/outputs/"
    "nano_nx_final_campaign/raw/standard_short/C/"
    "nano-nx-final-s42-standard_short-c-r12-20261022/server_results/server"
)
RAW_NANO_FLOWER = RAW_NANO_RUN / "fedavg_2026-10-22_17-18-24.csv"
CONFIRM = Path(
    "/Users/jang-won-u/Documents/Codex/2026-10-10/new-chat/outputs/"
    "learning_efficiency_confirmation"
)
INDEPENDENT = CONFIRM / "per_round_hardware_results.csv"
CACHE_SHA256 = {
    "fig5a_historical_prediction_values.csv": "9114002a4bf09f98042cf10d710d0cdedb4324e6d103f3a72a8ed6796f61ad24",
    "fig5a_historical_mare.csv": "ceef08b95b27ba69232675cefbbe9939c4951dd28479f4b0cc4c252eaaab9c32",
    "fig5b_executed_steps_by_seed.csv": "b8a736d39b6ee4100d9d2c9cec688d866610a041646f6bb8c72d28f79e0fa4e0",
    "fig5b_executed_steps_mean.csv": "d7aed93f8d8ebf9aa604bd74774d39c40b750daf4f3660f7049fcc63b8a70a43",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, values: list[dict[str, object]]) -> None:
    if not values:
        raise ValueError(f"empty Figure 5 table: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


class ReferenceStyle:
    """Local copy of the audited EEFL paper-style settings and save helper.

    The reference script can be an iCloud dataless placeholder on this host;
    opening it would hang. These values match its prior audited paper_style().
    """

    @staticmethod
    def cm_to_inch(width: float, height: float) -> tuple[float, float]:
        return width / 2.54, height / 2.54

    @staticmethod
    def savefig(fig, out_base: str, png_dpi: int = 600) -> None:
        fig.savefig(out_base + ".pdf")
        fig.savefig(out_base + ".png", dpi=png_dpi)
        plt.close(fig)


def load_style() -> ReferenceStyle:
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "lines.linewidth": 1.4,
        "savefig.bbox": None,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    return ReferenceStyle()


def historical_points() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    point_rows: list[dict[str, object]] = []
    grouped: dict[str, list[float]] = defaultdict(list)
    for power_mode, path in HISTORICAL.items():
        digest = sha256(path)
        for source_row, row in enumerate(rows(path), start=2):
            if row["phase"].strip().lower() != "train":
                continue
            round_no = int(float(row["round"]))
            if round_no <= 2:  # prespecified July plotting warm-up exclusion
                continue
            native_device = row["device_name"].lower()
            if "rpi" in native_device:
                device = "Pi 5"
            elif "jetson" in native_device or "orin" in native_device:
                device = "Orin NX"
            else:
                continue
            actual = float(row["actual_train_time_self"])
            predicted = float(row["pred_train_time_self"])
            if not all(map(math.isfinite, (actual, predicted))) or actual <= 0:
                raise ValueError(f"bad historical prediction at {path}:{source_row}")
            error = abs(predicted - actual) / actual * 100
            grouped[device].append(error)
            point_rows.append(
                {
                    "power_mode": power_mode,
                    "device": device,
                    "round": round_no,
                    "actual_compute_s": actual,
                    "logged_predicted_compute_s": predicted,
                    "absolute_relative_error_pct": error,
                    "source_csv": str(path),
                    "source_csv_sha256": digest,
                    "source_csv_row": source_row,
                    "predictor_version": "July Pi5/NX LoRACast-Online",
                }
            )
    summary = [
        {"device": device, "n_predictions": len(grouped[device]), "mare_pct": mean(grouped[device])}
        for device in ("Pi 5", "Orin NX")
    ]
    expected = {"Pi 5": (24, 3.9488220268236187), "Orin NX": (12, 1.1221454775186537)}
    for record in summary:
        count, mare = expected[record["device"]]
        if record["n_predictions"] != count or not math.isclose(record["mare_pct"], mare, abs_tol=1e-9):
            raise ValueError(f"historical source disagrees with audited MARE: {record}")
    if len(point_rows) != 36:
        raise ValueError("Figure 5(a) requires 36 directly logged post-warm-up predictions")
    return point_rows, summary


def nano_prediction_points() -> tuple[list[dict[str, object]], dict[str, object], list[dict[str, object]]]:
    """Use only native, pre-execution Nano predictions matched to local compute.

    R1--R2 are excluded to match the July post-warm-up rule. The campaign has
    fixed normal, contention, and recovery phases; the pooled MARE is explicitly
    descriptive of all three phases rather than an uncontended device ranking.
    """
    digest = sha256(NANO_SOURCE)
    if digest != NANO_SOURCE_SHA256:
        raise ValueError("preserved Nano prediction source checksum changed")
    values: list[dict[str, object]] = []
    phase_errors: dict[str, list[float]] = defaultdict(list)
    for source_row, row in enumerate(rows(NANO_SOURCE), start=2):
        if row["scenario"] != "standard_short" or row["policy"] != "C":
            continue
        round_no = int(row["round"])
        if round_no <= 2:
            continue
        if row["source_type"] != "native_online_prediction":
            raise ValueError(f"Nano R{round_no} is not a native pre-execution prediction")
        predicted = float(row["reference_predicted_compute_s"])
        actual = float(row["nano_actual_compute_s"])
        if not all(map(math.isfinite, (predicted, actual))) or min(predicted, actual) <= 0:
            raise ValueError(f"invalid Nano matched compute duration: {NANO_SOURCE}:{source_row}")
        expected_phase = "normal" if round_no <= 4 else "contention" if round_no <= 8 else "recovery"
        if row["phase"] != expected_phase:
            raise ValueError(f"unexpected Nano phase in R{round_no}")
        error = 100 * abs(predicted - actual) / actual
        phase_errors[expected_phase].append(error)
        values.append({
            "device": "Nano Super", "scenario": row["scenario"],
            "policy": row["policy"], "run_id": row["run_id"], "seed": 42,
            "round": round_no, "phase": expected_phase,
            "logged_predicted_compute_s": predicted,
            "actual_compute_s": actual,
            "absolute_relative_error_pct": error,
            "source_csv": str(NANO_SOURCE), "source_csv_sha256": digest,
            "source_csv_row": source_row,
            "source_json_recorded_path": row["source_json"],
        })
    if [int(r["round"]) for r in values] != list(range(3, 13)):
        raise ValueError("expected all ten matched post-warm-up Nano rounds R3--R12")
    summary = {"device": "Nano Super", "n_predictions": len(values),
               "mare_pct": mean(float(r["absolute_relative_error_pct"]) for r in values)}
    phase_summary = [{"device": "Nano Super", "phase": phase,
                      "n_predictions": len(phase_errors[phase]),
                      "mare_pct": mean(phase_errors[phase])}
                     for phase in ("normal", "contention", "recovery")]
    expected = {"normal": (2, 1.076002288617472),
                "contention": (4, 16.741410093167584),
                "recovery": (4, 44.22758023080356)}
    for record in phase_summary:
        count, mare = expected[record["phase"]]
        if record["n_predictions"] != count or not math.isclose(record["mare_pct"], mare, abs_tol=1e-9):
            raise ValueError(f"Nano phase MARE disagrees with preserved source: {record}")
    if not math.isclose(summary["mare_pct"], 24.602796587311953, abs_tol=1e-9):
        raise ValueError("Nano pooled MARE disagrees with preserved source")
    return values, summary, phase_summary


def verify_nano_raw(points: list[dict[str, object]]) -> dict[str, str]:
    """Match every plotted Nano prediction to the original pre-round decision.

    Flower's own pred_train_time_self is a client-side prediction and is not
    substituted for the earlier server decision value used in this panel.
    """
    flower = rows(RAW_NANO_FLOWER)
    nano_by_round: dict[int, dict[str, str]] = {}
    for row in flower:
        if row["phase"] == "train" and row["device_name"] == "jetson_orin_nano_super":
            round_no = int(row["round"])
            if round_no in nano_by_round:
                raise ValueError(f"duplicate raw Nano round R{round_no}")
            nano_by_round[round_no] = row
    sources = {str(RAW_NANO_FLOWER): sha256(RAW_NANO_FLOWER)}
    for point in points:
        round_no = int(point["round"])
        decision_path = RAW_NANO_RUN / f"nano_decision_round_{round_no:02d}.json"
        decision = json.loads(decision_path.read_text())
        raw = nano_by_round[round_no]
        if int(decision["round"]) != round_no:
            raise ValueError(f"raw Nano decision round mismatch at R{round_no}")
        if not math.isclose(float(decision["reference_predicted_seconds"]),
                            float(point["logged_predicted_compute_s"]),
                            rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"raw Nano decision prediction mismatch at R{round_no}")
        if not math.isclose(float(raw["loop_time"]), float(point["actual_compute_s"]),
                            rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"raw Nano Flower compute time mismatch at R{round_no}")
        if not math.isclose(float(raw["actual_train_time_self"]), float(raw["loop_time"]),
                            rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"raw Nano loop/actual duration mismatch at R{round_no}")
        requested = decision["requested"]["jetson_orin_nano_super"]
        if (int(requested["samples"]), int(requested["steps"]),
            int(raw["num_examples"]), int(raw["local_steps"])) != (320, 320, 320, 320):
            raise ValueError(f"raw Nano requested/executed workload mismatch at R{round_no}")
        sources[str(decision_path)] = sha256(decision_path)
    if len(points) != 10:
        raise ValueError("expected ten checked Nano decision/Flower pairs")
    return sources


def verify_raw_nx_step(run: dict[str, str], cache: dict[str, tuple[Path, str, dict[int, int]]]) -> tuple[Path, str]:
    run_id = run["run_id"]
    if run_id not in cache:
        folder = CONFIRM / "raw" / run["scenario"] / run["policy"] / run_id / "server_results/server"
        candidates = list(folder.glob("fedavg_*.csv"))
        if len(candidates) != 1:
            raise ValueError(f"expected one raw Flower metric CSV for {run_id}; found {candidates}")
        path = candidates[0]
        by_round: dict[int, int] = {}
        for row in rows(path):
            if row["phase"].lower() != "train":
                continue
            name = row.get("device_name", "").lower()
            # Nano Super also contains "jetson_orin" in its device name.
            if name != "jetson_orin_nx":
                continue
            round_no = int(float(row["round"]))
            if round_no in by_round:
                raise ValueError(f"duplicate raw NX round {run_id}/{round_no}")
            by_round[round_no] = int(float(row["local_steps"]))
        if sorted(by_round) != list(range(1, 13)):
            raise ValueError(f"incomplete original NX steps in {path}")
        cache[run_id] = (path, sha256(path), by_round)
    path, digest, by_round = cache[run_id]
    round_no = int(run["round"])
    if by_round[round_no] != int(run["nx_steps"]):
        raise ValueError(f"derived and raw executed steps disagree for {run_id}/{round_no}")
    return path, digest


def independent_steps() -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, tuple[Path, str, dict[int, int]]]]:
    selected = [r for r in rows(INDEPENDENT) if r["policy"] in ("C", "F")]
    raw_cache: dict[str, tuple[Path, str, dict[int, int]]] = {}
    values: list[dict[str, object]] = []
    digest = sha256(INDEPENDENT)
    for record in selected:
        if record["phase"] != ("normal" if int(record["round"]) <= 4 else "contention" if int(record["round"]) <= 8 else "recovery"):
            raise ValueError(f"phase/round mismatch: {record['run_id']}/{record['round']}")
        raw_path, raw_sha = verify_raw_nx_step(record, raw_cache)
        values.append(
            {
                "policy": "LoRACast" if record["policy"] == "C" else "Fixed-Hetero",
                "source_policy_code": record["policy"],
                "seed": int(record["seed"]),
                "run_id": record["run_id"],
                "round": int(record["round"]),
                "phase": record["phase"],
                "executed_optimizer_steps": int(record["nx_steps"]),
                "requested_sample_cap": int(record["nx_requested_samples"]),
                "requested_step_cap": int(record["nx_requested_step_cap"]),
                "source_summary_csv": str(INDEPENDENT),
                "source_summary_sha256": digest,
                "source_raw_flower_csv": str(raw_path),
                "source_raw_flower_sha256": raw_sha,
            }
        )
    if len(values) != 72:
        raise ValueError(f"expected 72 measured policy/seed/round records, got {len(values)}")
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for value in values:
        grouped[(str(value["policy"]), int(value["round"]))].append(value)
    means: list[dict[str, object]] = []
    for policy in ("LoRACast", "Fixed-Hetero"):
        for round_no in range(1, 13):
            group = grouped[policy, round_no]
            if sorted(r["seed"] for r in group) != [43, 44, 45]:
                raise ValueError(f"missing seed or duplicate at {policy} R{round_no}")
            steps = [int(r["executed_optimizer_steps"]) for r in group]
            means.append(
                {
                    "policy": policy,
                    "round": round_no,
                    "phase": group[0]["phase"],
                    "seed_count": 3,
                    "mean_executed_optimizer_steps": mean(steps),
                    "min_seed_steps": min(steps),
                    "max_seed_steps": max(steps),
                    "seed43_steps": steps[0],
                    "seed44_steps": steps[1],
                    "seed45_steps": steps[2],
                }
            )
    if any(r["mean_executed_optimizer_steps"] != 360 for r in means if r["policy"] == "Fixed-Hetero"):
        raise ValueError("Fixed-Hetero must remain at 360 actual NX steps in all 12 rounds")
    return values, means, raw_cache


def verified_cache() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, str]]:
    """Load unchanged plotted values from the previous raw-source audit.

    The frozen checksums are those in the original figure_source_manifest.json.
    Recompute every plotted statistic so this is more than a visual file copy.
    """
    for name, expected in CACHE_SHA256.items():
        path = VERIFY / name
        if sha256(path) != expected:
            raise ValueError(f"audited Figure 5 cache checksum changed: {path}")
    points_a = rows(VERIFY / "fig5a_historical_prediction_values.csv")
    prior_summary_a = rows(VERIFY / "fig5a_historical_mare.csv")
    points_b = rows(VERIFY / "fig5b_executed_steps_by_seed.csv")
    prior_means_b = rows(VERIFY / "fig5b_executed_steps_mean.csv")
    if len(points_a) != 36 or len(points_b) != 72:
        raise ValueError("Figure 5 cache row count disagrees with prior source audit")

    grouped_errors: dict[str, list[float]] = defaultdict(list)
    for r in points_a:
        actual = float(r["actual_compute_s"])
        predicted = float(r["logged_predicted_compute_s"])
        expected = abs(predicted - actual) / actual * 100
        observed = float(r["absolute_relative_error_pct"])
        if actual <= 0 or not math.isclose(expected, observed, abs_tol=1e-10):
            raise ValueError(f"July cached prediction error changed: {r}")
        grouped_errors[r["device"]].append(observed)
    summary_a: list[dict[str, object]] = []
    for device in ("Pi 5", "Orin NX"):
        errors = grouped_errors[device]
        summary_a.append({"device": device, "n_predictions": len(errors), "mare_pct": mean(errors)})
    for computed, original in zip(summary_a, prior_summary_a, strict=True):
        if (computed["device"] != original["device"]
                or computed["n_predictions"] != int(original["n_predictions"])
                or not math.isclose(float(computed["mare_pct"]), float(original["mare_pct"]), abs_tol=1e-10)):
            raise ValueError("computed July MARE disagrees with original audited plot")

    grouped_steps: dict[tuple[str, int], dict[int, int]] = defaultdict(dict)
    for r in points_b:
        policy, round_no, seed = r["policy"], int(r["round"]), int(r["seed"])
        expected_phase = "normal" if round_no <= 4 else "contention" if round_no <= 8 else "recovery"
        if r["phase"] != expected_phase or seed in grouped_steps[policy, round_no]:
            raise ValueError(f"bad or duplicate cached observation: {policy}/R{round_no}/s{seed}")
        grouped_steps[policy, round_no][seed] = int(r["executed_optimizer_steps"])
    means_b: list[dict[str, object]] = []
    for policy in ("LoRACast", "Fixed-Hetero"):
        for round_no in range(1, 13):
            by_seed = grouped_steps[policy, round_no]
            if sorted(by_seed) != [43, 44, 45]:
                raise ValueError(f"missing cached seed: {policy}/R{round_no}")
            steps = [by_seed[seed] for seed in (43, 44, 45)]
            means_b.append({
                "policy": policy, "round": round_no,
                "phase": "normal" if round_no <= 4 else "contention" if round_no <= 8 else "recovery",
                "seed_count": 3, "mean_executed_optimizer_steps": mean(steps),
                "min_seed_steps": min(steps), "max_seed_steps": max(steps),
                **{f"seed{seed}_steps": by_seed[seed] for seed in (43, 44, 45)},
            })
    for computed, original in zip(means_b, prior_means_b, strict=True):
        if computed["policy"] != original["policy"] or computed["round"] != int(original["round"]):
            raise ValueError("cached step policy/round changed")
        if not math.isclose(float(computed["mean_executed_optimizer_steps"]),
                            float(original["mean_executed_optimizer_steps"]), abs_tol=1e-10):
            raise ValueError("computed NX step mean disagrees with original audited plot")

    verified_sources: dict[str, str] = {str(STYLE): STYLE_SHA256_FROM_PRIOR_AUDIT}
    for r in points_a:
        verified_sources[r["source_csv"]] = r["source_csv_sha256"]
    for r in points_b:
        verified_sources[r["source_summary_csv"]] = r["source_summary_sha256"]
        verified_sources[r["source_raw_flower_csv"]] = r["source_raw_flower_sha256"]
    return points_a, summary_a, points_b, means_b, verified_sources


def crosscheck_july_raw(cached_points: list[dict[str, object]],
                        cached_summary: list[dict[str, object]]) -> None:
    """Optionally re-open the locally available July logs without touching October files."""
    raw_points, raw_summary = historical_points()
    for raw, cached in zip(raw_points, cached_points, strict=True):
        for key in ("power_mode", "device", "round", "source_csv", "source_csv_sha256", "source_csv_row"):
            if str(raw[key]) != str(cached[key]):
                raise ValueError(f"July raw/cache identity mismatch: {key}")
        for key in ("actual_compute_s", "logged_predicted_compute_s", "absolute_relative_error_pct"):
            if not math.isclose(float(raw[key]), float(cached[key]), abs_tol=1e-10):
                raise ValueError(f"July raw/cache numeric mismatch: {key}")
    for raw, cached in zip(raw_summary, cached_summary, strict=True):
        if raw["device"] != cached["device"] or raw["n_predictions"] != cached["n_predictions"]:
            raise ValueError("July raw/cache summary identity mismatch")
        if not math.isclose(float(raw["mare_pct"]), float(cached["mare_pct"]), abs_tol=1e-10):
            raise ValueError("July raw/cache MARE mismatch")


def style_axes(ax) -> None:
    ax.grid(axis="y", color="0.78", linestyle="-", linewidth=0.45, alpha=0.48)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.tick_params(axis="both", width=0.8, direction="out", top=False, right=False)


def plot_accuracy(ax, summary: list[dict[str, object]]) -> None:
    style_axes(ax)
    expected = ["Pi 5", "Orin NX", "Nano normal", "Nano contention", "Nano recovery"]
    if [r["device"] for r in summary] != expected:
        raise ValueError("Figure 5(a) requires the audited device/phase order")
    xs = (0, 1, 2, 3, 4)
    for idx, record in enumerate(summary):
        ax.bar(
            xs[idx], record["mare_pct"], width=0.56,
            facecolor=("0.82", "0.94", "0.77", "0.69", "0.60")[idx],
            edgecolor="0.18", linewidth=0.9,
            hatch=("///", "\\\\\\", "xxx", "xxx", "xxx")[idx], zorder=3,
        )
        ax.annotate(
            f"{record['mare_pct']:.2f}%", (xs[idx], record["mare_pct"]),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=8.2, fontweight="bold",
        )
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(0, 52)
    ax.set_yticks((0, 10, 20, 30, 40, 50))
    ax.set_xticks(xs, ["Pi 5", "Orin NX", "Nano\nnormal", "Nano\ncontention", "Nano\nrecovery"])
    ax.tick_params(axis="x", labelsize=8.1)
    ax.set_ylabel("Compute-time MARE (%)")


def plot_adaptation(ax, per_seed: list[dict[str, object]], averages: list[dict[str, object]]) -> None:
    style_axes(ax)
    # The visible traces are policy/round means. The per-seed observations
    # remain in the prior raw-audited verification CSV. Redundant thin
    # trajectories are omitted from the main panel.
    for policy, color, marker, linestyle, zorder, marker_size in (
        ("LoRACast", "#0072B2", "^", "-", 4, 5.2),
        ("Fixed-Hetero", "#4D4D4D", "s", "--", 5, 4.1),
    ):
        series = [r for r in averages if r["policy"] == policy]
        ax.plot([r["round"] for r in series],
                [r["mean_executed_optimizer_steps"] for r in series],
                color=color, linewidth=1.8 if policy == "LoRACast" else 1.5,
                marker=marker, markersize=marker_size,
                markerfacecolor="white", markeredgecolor=color,
                markeredgewidth=1.0, linestyle=linestyle,
                label=policy, zorder=zorder)
    for at in (4.5, 8.5):
        ax.axvline(at, color="0.45", linestyle=":", linewidth=0.8, zorder=1)
    for x, phase in ((2.5, "Normal"), (6.5, "Contention"), (10.5, "Recovery")):
        ax.text(x, 627, phase, ha="center", va="center", fontsize=8.1,
                color="0.25")
    ax.set_xlim(0.72, 12.28)
    ax.set_ylim(325, 650)
    ax.set_xticks((1, 3, 5, 7, 9, 12))
    ax.set_yticks((350, 400, 500, 600))
    ax.set_xlabel("FL round")
    ax.set_ylabel("NX executed steps")
    # The legend sits in the data-free space above the shared Normal trace.
    ax.legend(loc="center left", bbox_to_anchor=(0.012, 0.57),
              frameon=False, fontsize=8.0, handlelength=1.3,
              handletextpad=0.35, borderpad=0)


def export(fig, stem: str, reference) -> dict[str, str]:
    # The reference helper handles PDF/PNG export and figure closure. SVG is
    # saved first because that older helper predates the editable-SVG request.
    fig.savefig(MAIN / f"{stem}.svg", bbox_inches=None)
    reference.savefig(fig, str(MAIN / stem), png_dpi=600)
    return {extension: str(MAIN / f"{stem}.{extension}")
            for extension in ("pdf", "svg", "png")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-raw", action="store_true",
                        help="re-derive and verify values from all July and October source CSVs")
    parser.add_argument("--verify-july", action="store_true",
                        help="cross-check July source CSVs against the frozen values")
    parser.add_argument("--verify-nano-raw", action="store_true",
                        help="cross-check Nano decision JSON and original Flower CSV against plotted values")
    parser.add_argument("--only-a", action="store_true",
                        help="re-render panel (a) while preserving the existing panel (b) exports")
    args = parser.parse_args()
    MAIN.mkdir(parents=True, exist_ok=True)
    VERIFY.mkdir(parents=True, exist_ok=True)
    reference = load_style()
    if args.from_raw:
        points_a, summary_a = historical_points()
        points_b, means_b, raw_cache = independent_steps()
        write_csv(VERIFY / "fig5a_historical_prediction_values.csv", points_a)
        write_csv(VERIFY / "fig5a_historical_mare.csv", summary_a)
        write_csv(VERIFY / "fig5b_executed_steps_by_seed.csv", points_b)
        write_csv(VERIFY / "fig5b_executed_steps_mean.csv", means_b)
        verified_sources = {str(STYLE): STYLE_SHA256_FROM_PRIOR_AUDIT}
        verified_sources.update({str(path): sha256(path) for path in [*HISTORICAL.values(), INDEPENDENT]})
        verified_sources.update({str(path): digest for path, digest, _ in raw_cache.values()})
        source_access_mode = "raw_source_files_reopened_and_crosschecked"
    else:
        points_a, summary_a, points_b, means_b, verified_sources = verified_cache()
        raw_cache = {}
        source_access_mode = "frozen_raw_audited_value_csvs"
        if args.verify_july:
            crosscheck_july_raw(points_a, summary_a)
            source_access_mode += "; July originals reopened and crosschecked"

    fig_a, ax_a = plt.subplots(figsize=reference.cm_to_inch(8.6, 5.2))
    nano_points, nano_summary, nano_phases = nano_prediction_points()
    nano_raw_sources = verify_nano_raw(nano_points) if args.verify_nano_raw else {}
    verified_sources.update(nano_raw_sources)
    if args.verify_nano_raw:
        source_access_mode += "; original Nano decisions and Flower compute records reopened and crosschecked"
    summary_a.append(nano_summary)
    plotted_a = summary_a[:2] + [
        {"device": f"Nano {r['phase']}", "n_predictions": r["n_predictions"],
         "mare_pct": r["mare_pct"]} for r in nano_phases
    ]
    write_csv(VERIFY / "fig5a_nano_prediction_values.csv", nano_points)
    write_csv(VERIFY / "fig5a_nano_phase_mare.csv", nano_phases)
    write_csv(VERIFY / "fig5a_three_device_mare.csv", summary_a)
    write_csv(VERIFY / "fig5a_plotted_phase_aware_mare.csv", plotted_a)
    verified_sources[str(NANO_SOURCE)] = NANO_SOURCE_SHA256
    fig_a.subplots_adjust(left=0.21, right=0.99, bottom=0.25, top=0.94)
    plot_accuracy(ax_a, plotted_a)
    single_a = export(fig_a, "fig5a_historical_compute_prediction_error", reference)

    if args.only_a:
        single_b = {extension: str(MAIN / f"fig5b_runtime_workload_adaptation.{extension}")
                    for extension in ("pdf", "svg", "png")}
        if not all(Path(path).is_file() for path in single_b.values()):
            raise FileNotFoundError("panel (b) export missing; rerun without --only-a")
    else:
        fig_b, ax_b = plt.subplots(figsize=reference.cm_to_inch(8.6, 5.2))
        fig_b.subplots_adjust(left=0.21, right=0.985, bottom=0.23, top=0.94)
        plot_adaptation(ax_b, points_b, means_b)
        single_b = export(fig_b, "fig5b_runtime_workload_adaptation", reference)

    result = {
        "status": "PHASE_AWARE_CROSS_CAMPAIGN_PANEL_A; READY_PANEL_B",
        "font_resolved": font_manager.findfont("Times New Roman", fallback_to_default=False),
        "reference_style": str(STYLE),
        "reference_style_sha256": STYLE_SHA256_FROM_PRIOR_AUDIT,
        "reference_style_access_mode": "local compatibility copy of previously audited rcParams; original EEFL file is iCloud dataless on this host",
        "sources_sha256": verified_sources,
        "source_access_mode": source_access_mode,
        "audited_cache_sha256": CACHE_SHA256,
        "panel_a": {
            "metric": "mean absolute relative error of logged predicted versus actual local training compute duration",
            "formula": "mean(100*abs(pre-execution predicted local compute seconds - observed local compute seconds)/observed local compute seconds)",
            "filter": "July Pi5/NX LoRACast-Online MAXN/15W/10W train rows R3 onward; October Nano Super seed42 standard_short C run R3--R12, native pre-execution predictions only",
            "estimator_identity": "July Pi5/NX versus October Nano Super code/config equivalence not established; descriptive cross-campaign display only",
            "seed": "July not recorded in original CSVs; October Nano seed 42",
            "device_n": {r["device"]: r["n_predictions"] for r in summary_a},
            "mare_pct": {r["device"]: r["mare_pct"] for r in summary_a},
            "plotted_bars": plotted_a,
            "nano_phase_mare_pct": {r["phase"]: r["mare_pct"] for r in nano_phases},
            "run_ids": sorted({r["run_id"] for r in nano_points}),
            "input_rows": len(points_a) + len(nano_points),
            "july_input_rows": len(points_a),
            "nano_input_rows": len(nano_points),
            "nano_raw_flower_and_decisions_crosschecked_this_run": args.verify_nano_raw,
            "nano_raw_source_sha256": nano_raw_sources,
            "july_raw_crosschecked_this_run": args.from_raw or args.verify_july,
        },
        "panel_b": {
            "metric": "client-reported actual executed local_steps, Orin NX",
            "aggregation": "mean of independent seed-level FL runs at same policy and round",
            "seed_ids": [43, 44, 45],
            "run_ids": sorted({r["run_id"] for r in points_b}),
            "rounds": list(range(1, 13)),
            "phase_boundaries_after_round": [4, 8],
            "input_rows": len(points_b),
            "mean_rows": len(means_b),
            "R12_mean_executed_steps": {
                policy: next(float(r["mean_executed_optimizer_steps"]) for r in means_b
                             if r["policy"] == policy and int(r["round"]) == 12)
                for policy in ("LoRACast", "Fixed-Hetero")
            },
            "raw_flower_steps_crosschecked_in_prior_audit": True,
            "raw_flower_steps_crosschecked_this_run": args.from_raw,
            "scope": "Nano Super and Orin NX; policy-level adaptation, not an isolated online-alpha ablation",
        },
        "outputs": {"panel_a": single_a, "panel_b": single_b},
        "intended_cm": {"individual": [8.6, 5.2]},
        "plot_style": {"base_font_pt": 10, "axis_label_pt": 10, "tick_pt": 9,
                       "fig5b_legend_pt": 8.0, "fig5b_loracast_line_width_pt": 1.8,
                       "fig5b_fixed_hetero_line_width_pt": 1.5,
                       "fig5b_policy_encoding": {
                           "LoRACast": "#0072B2 solid upward triangle",
                           "Fixed-Hetero": "#4D4D4D dashed square"},
                       "axes_width_pt": 0.8, "pdf_fonttype": 42, "svg_fonttype": "none"},
    }
    (VERIFY / "fig5_source_and_plot_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "panel_a": result["panel_a"]["mare_pct"],
                      "panel_b_rows": len(points_b), "outputs": result["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
