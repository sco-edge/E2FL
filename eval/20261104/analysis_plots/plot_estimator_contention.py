#!/usr/bin/env python3
"""Rebuild WWW Fig. 5(a--c) from original July and Nano/NX records.

No training, scheduling, or estimator update is executed. The Frozen series is
an offline prediction-only replay: each seed's native R3 pre-round prediction
(computed from R1--R2 warm-up state) is held fixed at the constant, actually
executed Nano S/H/E=320/320/1 workload in R3--R12.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from plot_cost_prediction_adaptation import load_style, style_axes
from plot_historical_estimator_errors import SAMPLES, SUMMARY, raw_path, validated_samples


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "main"
VERIFY = ROOT / "verification"
OVERVIEW = ROOT / "comparison"
CONFIRM = Path(
    "/Users/jang-won-u/Documents/Codex/2026-10-10/new-chat/outputs/"
    "learning_efficiency_confirmation"
)
PREDICTIONS = CONFIRM / "measurements/estimator_predictions.csv"
HARDWARE = CONFIRM / "per_round_hardware_results.csv"
CAMPAIGN = CONFIRM / "campaign_manifest.json"
NANO_ENERGY_INTEGRATION = CONFIRM / "analyze_incremental.py"
ORIGINAL_ENERGY_ESTIMATOR = Path("/Users/jang-won-u/Documents/GitHub/EEFL/E2FL/e2fl/estimator.py")
JULY_MAXN_ORIGINAL = Path(
    "/Users/jang-won-u/Documents/GitHub/EEFL/E2FL/eval/evaluate/20260511/"
    "20. power (LoRACast-Online_MAXN)/bkp/fedavg_2026-07-08_22-47-51.csv"
)
JULY_MAXN_ORIGINAL_SHA256 = "f01108bbeb1afbf32d03a7d0f6798ed272f44cefb50b3f815e585b39392d5f93"
SEEDS = (43, 44, 45)
POLICIES = ("C", "F", "P")
STEMS = (
    "fig5a_normal_prediction_mare",
    "fig5b_contention_prediction_ape",
    "fig5c_runtime_workload_adaptation",
)
PRIOR_VALIDATION = VERIFY / "fig5_estimator_contention_validation_initial_pass.json"
RAW_UNAVAILABLE: list[str] = []
RECHECK_RAW = os.environ.get("FIG5_RECHECK_ORIGINAL_RECORDS") == "1"


def prior_source_hash(path: Path) -> str:
    if not PRIOR_VALIDATION.is_file():
        raise AssertionError(f"original-record audit unavailable: {PRIOR_VALIDATION}")
    prior = json.loads(PRIOR_VALIDATION.read_text())
    result = prior.get("source_sha256", {}).get(str(path))
    if not result:
        raise AssertionError(f"source absent from original-record audit: {path}")
    return result


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, values: list[dict]) -> None:
    if not values:
        raise ValueError(f"empty plot table: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def close(a: float, b: float, label: str, tolerance: float = 1e-8) -> None:
    if not math.isclose(a, b, rel_tol=0, abs_tol=tolerance):
        raise AssertionError(f"{label}: {a} != {b}")


def phase(round_no: int) -> str:
    if 1 <= round_no <= 4:
        return "normal"
    if 5 <= round_no <= 8:
        return "contention"
    if 9 <= round_no <= 12:
        return "recovery"
    raise ValueError(f"outside confirmed 12-round campaign: {round_no}")


def figure_a(sources: dict[str, str]) -> tuple[list[dict], list[dict]]:
    """Use the existing INFOCOM parser, which checks each derived row to raw."""
    checked, raw_hashes, checked_summary = validated_samples()
    if digest(JULY_MAXN_ORIGINAL) != JULY_MAXN_ORIGINAL_SHA256:
        raise AssertionError("July MAXN original Flower CSV changed")
    sources[str(JULY_MAXN_ORIGINAL)] = JULY_MAXN_ORIGINAL_SHA256
    sources[str(SAMPLES)] = digest(SAMPLES)
    sources[str(SUMMARY)] = digest(SUMMARY)
    sources.update(raw_hashes)
    source_rows = [r for r in read_csv(SAMPLES)
                   if r["compute_time_error_pct"] and int(r["round"]) > 2]
    original_maxn = {(int(r["round"]), r["device_name"]): r
                     for r in read_csv(JULY_MAXN_ORIGINAL) if r["phase"] == "train"}
    transformed_maxn_path = next(raw_path(r["source_csv"]) for r in source_rows if r["run"] == "MAXN")
    transformed_maxn = {(int(r["round"]), r["device_name"]): r
                        for r in read_csv(transformed_maxn_path) if r["phase"] == "train"}
    if set(original_maxn) != set(transformed_maxn):
        raise AssertionError("July MAXN transformed copy omits an original training row")
    for key, original in original_maxn.items():
        transformed = transformed_maxn[key]
        for field in ("loop_time", "pred_train_time_self", "energy", "est_alpha_self",
                      "est_beta_self", "actual_train_time_self"):
            if original[field] == transformed[field] == "":
                continue
            close(float(original[field]), float(transformed[field]), f"July MAXN {key} {field}")
    if len(source_rows) != 36 or len(checked) != 36:
        raise AssertionError("July post-warm-up set must contain 36 observations")
    details: list[dict] = []
    for src, prior in zip(source_rows, checked, strict=True):
        path = raw_path(src["source_csv"])
        device = "Pi 5" if src["device"] == "RPi 5" else "Orin NX"
        close(float(src["compute_time_error_pct"]), prior["compute_time_error_pct"], "July time APE")
        close(float(src["compute_energy_error_pct"]), prior["compute_energy_error_pct"], "July energy APE")
        details.append({
            "device": device, "power_setting": src["run"], "round": int(src["round"]),
            "predicted_compute_s": float(src["predicted_compute_time_s"]),
            "observed_compute_s": float(src["actual_compute_time_s"]),
            "predicted_compute_energy_proxy_j": float(src["predicted_compute_energy_j"]),
            "observed_compute_energy_proxy_j": float(src["actual_compute_energy_j"]),
            "warmup_calibrated_power_w": float(src["calibrated_power_w"]),
            "compute_time_ape_pct": float(src["compute_time_error_pct"]),
            "compute_energy_proxy_ape_pct": float(src["compute_energy_error_pct"]),
            "source_raw_csv": str(path), "source_raw_sha256": raw_hashes[str(path)],
            "source_derived_csv": str(SAMPLES),
        })
    plotted = []
    for device in ("Pi 5", "Orin NX"):
        matches = [r for r in details if r["device"] == device]
        plotted.append({
            "device": device, "n_predictions": len(matches),
            "compute_time_mare_pct": mean(r["compute_time_ape_pct"] for r in matches),
            "compute_energy_proxy_mare_pct": mean(r["compute_energy_proxy_ape_pct"] for r in matches),
            "power_settings": "MAXN,15W,10W",
        })
    if [r["n_predictions"] for r in plotted] != [24, 12]:
        raise AssertionError("July Pi/NX observation counts changed")
    for dev, label in (("Pi 5", "RPi 5"), ("Orin NX", "Jetson Orin NX")):
        record = next(r for r in plotted if r["device"] == dev)
        for metric, key in (("compute_time_error_pct", "compute_time_mare_pct"),
                            ("compute_energy_error_pct", "compute_energy_proxy_mare_pct")):
            close(record[key], checked_summary[f"{label}/{metric}"]["mean_pct"],
                  f"July {dev} {metric}")
    write_csv(VERIFY / "fig5a_normal_prediction_values.csv", details)
    write_csv(VERIFY / "fig5a_normal_prediction_mare.csv", plotted)
    return details, plotted


def flower_rounds(run: dict[str, str], cache: dict[tuple[str, str], tuple[Path, str, dict[int, dict[str, str]]]],
                  sources: dict[str, str], device_name: str = "jetson_orin_nx") -> dict[int, dict[str, str]]:
    run_id = run["run_id"]
    key = (run_id, device_name)
    if key not in cache:
        folder = CONFIRM / "raw" / run["scenario"] / run["policy"] / run_id / "server_results/server"
        files = list(folder.glob("fedavg_*.csv"))
        if len(files) != 1:
            raise AssertionError(f"expected one original Flower CSV for {run_id}: {files}")
        path = files[0]
        if not RECHECK_RAW:
            source_hash = prior_source_hash(path)
            cache[key] = path, source_hash, {}
            sources[str(path)] = source_hash
            if str(path) not in RAW_UNAVAILABLE:
                RAW_UNAVAILABLE.append(str(path))
            return cache[key][2]
        by_round = {}
        for row in read_csv(path):
            if row["phase"] != "train" or row["device_name"] != device_name:
                continue
            round_no = int(row["round"])
            if round_no in by_round:
                raise AssertionError(f"duplicate original NX row: {run_id}/R{round_no}")
            by_round[round_no] = row
        if not by_round:
            # macOS can expose an iCloud dataless file with a nonzero stat size
            # while reads return EOF. Do not claim a fresh raw-row check in that
            # state. Keep the hash from the earlier successful source audit.
            prior = json.loads(PRIOR_VALIDATION.read_text()) if PRIOR_VALIDATION.is_file() else {}
            source_hash = prior.get("source_sha256", {}).get(str(path))
            if not source_hash:
                raise AssertionError(f"original Flower source unavailable and unaudited: {path}")
            if str(path) not in RAW_UNAVAILABLE:
                RAW_UNAVAILABLE.append(str(path))
        else:
            if sorted(by_round) != list(range(1, 13)):
                raise AssertionError(f"incomplete original Flower execution: {run_id}/{device_name}")
            source_hash = digest(path)
            prior = json.loads(PRIOR_VALIDATION.read_text()) if PRIOR_VALIDATION.is_file() else {}
            old_hash = prior.get("source_sha256", {}).get(str(path))
            if old_hash and source_hash != old_hash:
                raise AssertionError(f"original Flower source changed: {path}")
        cache[key] = path, source_hash, by_round
        sources[str(path)] = source_hash
    return cache[key][2]


def figure_bc(sources: dict[str, str]) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict], list[dict]]:
    sources[str(PREDICTIONS)] = digest(PREDICTIONS)
    sources[str(HARDWARE)] = digest(HARDWARE)
    sources[str(CAMPAIGN)] = digest(CAMPAIGN)
    source_predictions = sorted((r for r in read_csv(PREDICTIONS) if r["policy"] == "C"),
                                key=lambda r: (int(r["scenario"].split("_")[0][4:]), int(r["round"])))
    source_hardware = [r for r in read_csv(HARDWARE) if r["policy"] in POLICIES]
    if len(source_predictions) != 36 or len(source_hardware) != 108:
        raise AssertionError("expected 3 seeds × 12 rounds of predictions and 3 policies")
    hardware_key = {(int(r["seed"]), r["policy"], int(r["round"])): r for r in source_hardware}
    if len(hardware_key) != len(source_hardware):
        raise AssertionError("duplicate hardware policy/seed/round")
    flower_cache: dict[tuple[str, str], tuple[Path, str, dict[int, dict[str, str]]]] = {}
    b_by_seed = []
    first_prediction: dict[int, float] = {}
    c_by_seed = []
    c_slack = []
    for src in source_predictions:
        seed = int(src["scenario"].split("_")[0].replace("seed", ""))
        round_no = int(src["round"])
        if seed not in SEEDS or src["phase"] != phase(round_no):
            raise AssertionError(f"Nano phase or seed mismatch: {src['run_id']} R{round_no}")
        if src["source_type"] != "native_online_prediction":
            raise AssertionError(f"Nano prediction is not native pre-round: {src['run_id']} R{round_no}")
        observed = float(src["nano_actual_compute_s"])
        hardware = hardware_key[(seed, "C", round_no)]
        close(observed, float(hardware["nano_compute_s"]), "Nano observed compute")
        close(float(src["nx_actual_compute_s"]), float(hardware["nx_compute_s"]),
              "NX observed compute in prediction table")
        original_nano_rounds = flower_rounds(hardware, flower_cache, sources,
                                             device_name="jetson_orin_nano_super")
        if original_nano_rounds:
            original_nano = original_nano_rounds[round_no]
            close(observed, float(original_nano["loop_time"]), "Nano original Flower loop time")
            if (int(original_nano["num_examples"]), int(original_nano["local_steps"])) != (320, 320):
                raise AssertionError(f"original Nano execution changed: seed{seed} R{round_no}")
        if (int(hardware["nano_requested_samples"]), int(hardware["nano_requested_step_cap"]),
            int(hardware["nano_steps"])) != (320, 320, 320):
            raise AssertionError(f"Nano workload changed: seed{seed} R{round_no}")
        decision_path = CONFIRM / src["source_json"]
        predicted = float(src["reference_predicted_compute_s"])
        if RECHECK_RAW:
            decision = json.loads(decision_path.read_text())
            sources[str(decision_path)] = digest(decision_path)
            close(predicted, float(decision["reference_predicted_seconds"]), "native pre-round prediction")
            if int(decision["round"]) != round_no:
                raise AssertionError(f"wrong Nano decision round {decision_path}")
            requested_nano = decision["requested"]["jetson_orin_nano_super"]
            if (int(requested_nano["samples"]), int(requested_nano["steps"])) != (320, 320):
                raise AssertionError(f"original Nano decision changed: seed{seed} R{round_no}")
        else:
            sources[str(decision_path)] = prior_source_hash(decision_path)
            if str(decision_path) not in RAW_UNAVAILABLE:
                RAW_UNAVAILABLE.append(str(decision_path))
        if round_no == 3:
            if seed in first_prediction or predicted <= 0:
                raise AssertionError(f"invalid seed{seed} R3 warm-up-frozen anchor")
            first_prediction[seed] = predicted
        if round_no <= 2:
            continue  # Warm-up is never scored as a prediction error.
        if predicted <= 0 or observed <= 0:
            raise AssertionError(f"nonpositive Nano prediction/observation: seed{seed} R{round_no}")
        # R3's pre-round value is computed from R1--R2 only. The Nano workload
        # stays 320/320/1, so freezing its full calibrated state holds this
        # duration fixed. This is not a second executed scheduling policy.
        if seed not in first_prediction:
            raise AssertionError("prediction source is not in chronological order")
        frozen = first_prediction[seed]
        b_by_seed.append({
            "seed": seed, "run_id": src["run_id"], "round": round_no,
            "phase": src["phase"], "nano_requested_samples": 320,
            "nano_executed_steps": 320, "observed_nano_compute_s": observed,
            "native_pre_round_online_prediction_s": predicted,
            "frozen_after_r2_prediction_s": frozen,
            "online_ape_pct": 100 * abs(predicted - observed) / observed,
            "frozen_ape_pct": 100 * abs(frozen - observed) / observed,
            "frozen_method": "hold_native_R3_pre_round_calibrated_prediction_at_fixed_Nano_workload",
            "source_decision_json": str(decision_path),
            "source_decision_sha256": sources[str(decision_path)],
            "source_prediction_csv": str(PREDICTIONS),
        })
    if sorted(first_prediction) != list(SEEDS) or len(b_by_seed) != 30:
        raise AssertionError("missing R3 frozen anchor or R3--R12 Nano observation")
    for row in source_hardware:
        seed, policy, round_no = int(row["seed"]), row["policy"], int(row["round"])
        if seed not in SEEDS or row["phase"] != phase(round_no):
            raise AssertionError(f"hardware phase or seed mismatch: {row['run_id']} R{round_no}")
        raw_rounds = flower_rounds(row, flower_cache, sources)
        executed = int(row["nx_steps"])
        if raw_rounds:
            raw = raw_rounds[round_no]
            if int(raw["local_steps"]) != executed:
                raise AssertionError(f"derived/raw NX steps disagree: {row['run_id']} R{round_no}")
            if int(raw["max_samples"]) != int(row["nx_requested_samples"]):
                raise AssertionError(f"derived/raw NX requested S disagree: {row['run_id']} R{round_no}")
            close(float(raw["loop_time"]), float(row["nx_compute_s"]), "NX local compute")
        c_by_seed.append({
            "policy": {"C": "LoRACast", "F": "Fixed-Hetero", "P": "PyramidFL-style"}[policy],
            "source_policy_code": policy, "seed": seed, "run_id": row["run_id"],
            "round": round_no, "phase": row["phase"],
            "nx_requested_samples": int(row["nx_requested_samples"]),
            "nx_requested_step_cap": int(row["nx_requested_step_cap"]),
            "nx_executed_optimizer_steps": executed,
            "nx_measured_local_compute_s": float(row["nx_compute_s"]),
            "nano_measured_local_compute_s": float(row["nano_compute_s"]),
            "source_original_flower_csv": str(flower_cache[(row["run_id"], "jetson_orin_nx")][0]),
            "source_original_flower_sha256": flower_cache[(row["run_id"], "jetson_orin_nx")][1],
        })
        if policy == "C":
            nano = float(row["nano_compute_s"])
            nx = float(row["nx_compute_s"])
            c_slack.append({
                "seed": seed, "run_id": row["run_id"], "round": round_no,
                "phase": row["phase"], "nano_measured_local_compute_s": nano,
                "nx_measured_local_compute_s": nx,
                "nx_minus_nano_local_compute_s": nx - nano,
                "nx_overrun_positive_s": max(0, nx - nano),
                "measurement_boundary": "client_training_loop_only_not_full_round_barrier",
            })
    if len(c_by_seed) != 108 or len(c_slack) != 36 or len(flower_cache) != 12:
        raise AssertionError("incomplete original policy execution cross-check")
    b_mean = []
    for round_no in range(3, 13):
        matched = sorted((r for r in b_by_seed if r["round"] == round_no), key=lambda r: r["seed"])
        if [r["seed"] for r in matched] != list(SEEDS):
            raise AssertionError(f"missing independent Nano seed at R{round_no}")
        b_mean.append({
            "round": round_no, "phase": phase(round_no), "seed_count": 3,
            "online_mean_ape_pct": mean(r["online_ape_pct"] for r in matched),
            "frozen_mean_ape_pct": mean(r["frozen_ape_pct"] for r in matched),
            "online_seed43_ape_pct": matched[0]["online_ape_pct"],
            "online_seed44_ape_pct": matched[1]["online_ape_pct"],
            "online_seed45_ape_pct": matched[2]["online_ape_pct"],
            "frozen_seed43_ape_pct": matched[0]["frozen_ape_pct"],
            "frozen_seed44_ape_pct": matched[1]["frozen_ape_pct"],
            "frozen_seed45_ape_pct": matched[2]["frozen_ape_pct"],
        })
    stage_definitions = (
        ("Normal", (3, 4)), ("Contention onset", (5,)),
        ("Sustained contention", (6, 7, 8)), ("Recovery onset", (9,)),
        ("Recovery", (10, 11, 12)),
    )
    stage_summary = []
    for name, round_numbers in stage_definitions:
        subset = [r for r in b_by_seed if r["round"] in round_numbers]
        stage_summary.append({
            "phase": name, "rounds": ",".join(str(x) for x in round_numbers),
            "seed_round_observations": len(subset),
            "online_time_mare_pct": mean(r["online_ape_pct"] for r in subset),
            "frozen_time_mare_pct": mean(r["frozen_ape_pct"] for r in subset),
        })
    c_mean = []
    for policy in ("LoRACast", "Fixed-Hetero", "PyramidFL-style"):
        for round_no in range(1, 13):
            matched = sorted((r for r in c_by_seed if r["policy"] == policy and r["round"] == round_no),
                             key=lambda r: r["seed"])
            if [r["seed"] for r in matched] != list(SEEDS):
                raise AssertionError(f"missing independent policy seed: {policy}/R{round_no}")
            c_mean.append({
                "policy": policy, "round": round_no, "phase": phase(round_no),
                "seed_count": 3,
                "mean_nx_executed_optimizer_steps": mean(r["nx_executed_optimizer_steps"] for r in matched),
                "seed43_executed_steps": matched[0]["nx_executed_optimizer_steps"],
                "seed44_executed_steps": matched[1]["nx_executed_optimizer_steps"],
                "seed45_executed_steps": matched[2]["nx_executed_optimizer_steps"],
            })
    write_csv(VERIFY / "fig5b_contention_prediction_by_seed.csv", b_by_seed)
    write_csv(VERIFY / "fig5b_contention_prediction_ape.csv", b_mean)
    write_csv(VERIFY / "fig5b_contention_phase_summary.csv", stage_summary)
    write_csv(VERIFY / "fig5c_executed_steps_by_seed.csv", c_by_seed)
    write_csv(VERIFY / "fig5c_runtime_workload_adaptation.csv", c_mean)
    write_csv(VERIFY / "fig5c_local_compute_window_diagnostics.csv", c_slack)
    return b_by_seed, b_mean, stage_summary, c_by_seed, c_mean, c_slack


def nano_normal_energy_proxy(nano_normal: list[dict], sources: dict[str, str]) -> list[dict]:
    """Reconstruct the warm-up-power × predicted-time proxy for Nano.

    Nano VDD_IN joules are integrated over the server-active round, not the
    client training loop. Dividing by server-active seconds yields a measured
    *round-average* power; multiplying by local-compute seconds is therefore a
    power-times-duration proxy, never a measured compute-only joule integral.
    """
    hardware_rows = [r for r in read_csv(HARDWARE) if r["policy"] == "C"]
    sources[str(NANO_ENERGY_INTEGRATION)] = digest(NANO_ENERGY_INTEGRATION)
    sources[str(ORIGINAL_ENERGY_ESTIMATOR)] = digest(ORIGINAL_ENERGY_ESTIMATOR)
    by_seed_round = {(int(r["seed"]), int(r["round"])): r for r in hardware_rows}
    if len(by_seed_round) != 36:
        raise AssertionError("Nano energy proxy requires 3 complete 12-round hardware traces")
    sources[str(HARDWARE)] = digest(HARDWARE)

    def active_power(row: dict[str, str]) -> float:
        energy_j = float(row["nano_active_energy_j"])
        active_s = float(row["server_active_round_s"])
        if energy_j <= 0 or active_s <= 0:
            raise AssertionError("nonpositive Nano active-round energy or duration")
        return energy_j / active_s

    calibrated = {}
    for seed in SEEDS:
        warmup = [by_seed_round[(seed, round_no)] for round_no in (1, 2)]
        if any(r["phase"] != "normal" for r in warmup):
            raise AssertionError("Nano power calibration uses only normal R1-R2")
        calibrated[seed] = median(active_power(r) for r in warmup)

    rows = []
    for r in nano_normal:
        seed, round_no = int(r["seed"]), int(r["round"])
        hardware = by_seed_round[(seed, round_no)]
        if hardware["run_id"] != r["run_id"] or hardware["phase"] != "normal":
            raise AssertionError("Nano energy proxy run/phase mismatch")
        observed_time = float(hardware["nano_compute_s"])
        close(observed_time, float(r["observed_nano_compute_s"]),
              "Nano proxy observed local-compute duration")
        predicted_time = float(r["native_pre_round_online_prediction_s"])
        observed_power = active_power(hardware)
        predicted_proxy = calibrated[seed] * predicted_time
        observed_proxy = observed_power * observed_time
        ape = 100 * abs(predicted_proxy - observed_proxy) / observed_proxy
        rows.append({
            "device": "Nano Super", "seed": seed, "run_id": r["run_id"],
            "round": round_no, "phase": r["phase"],
            "predicted_compute_s": predicted_time,
            "observed_compute_s": observed_time,
            "compute_time_ape_pct": float(r["online_ape_pct"]),
            "warmup_r1_active_round_power_w": active_power(by_seed_round[(seed, 1)]),
            "warmup_r2_active_round_power_w": active_power(by_seed_round[(seed, 2)]),
            "calibrated_active_round_power_w": calibrated[seed],
            "observed_active_round_energy_j": float(hardware["nano_active_energy_j"]),
            "observed_server_active_round_s": float(hardware["server_active_round_s"]),
            "observed_active_round_mean_power_w": observed_power,
            "predicted_compute_energy_proxy_j": predicted_proxy,
            "observed_compute_energy_proxy_j": observed_proxy,
            "compute_energy_proxy_ape_pct": ape,
            "energy_proxy_boundary": "active_round_mean_VDD_IN_power_times_local_compute_duration",
            "source_decision_json": r["source_decision_json"],
            "source_decision_sha256": r["source_decision_sha256"],
            "source_prediction_csv": r["source_prediction_csv"],
            "source_hardware_csv": str(HARDWARE),
            "source_hardware_sha256": sources[str(HARDWARE)],
            "source_reply_trace": hardware["source_reply_trace"],
        })
    if len(rows) != 6 or any(row["round"] not in (3, 4) for row in rows):
        raise AssertionError("Nano normal energy proxy requires six R3-R4 observations")
    return rows


def format_phase_axis(ax, *, y_for_label: float) -> None:
    ax.axvspan(0.5, 4.5, color="0.96", zorder=0)
    ax.axvspan(4.5, 8.5, color="0.90", zorder=0)
    ax.axvspan(8.5, 12.5, color="0.96", zorder=0)
    for at in (4.5, 8.5):
        ax.axvline(at, color="0.45", linestyle=":", linewidth=0.8, zorder=1)
    for x, label in ((2.7, "Normal"), (6.5, "Contention"), (10.5, "Recovery")):
        ax.text(x, y_for_label, label, ha="center", va="center", fontsize=7.8, color="0.28")
    ax.set_xlim(0.7, 12.3)
    ax.set_xticks((1, 3, 5, 7, 9, 12))
    ax.set_xlabel("FL round")


def plot_a(data: list[dict], reference) -> None:
    fig, ax = plt.subplots(figsize=reference.cm_to_inch(8.6, 5.2))
    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.20, top=0.77)
    style_axes(ax)
    width = 0.21
    for index, row in enumerate(data):
        for offset, field, color, hatch in (
            (-0.19, "compute_time_mare_pct", "#0072B2", "///"),
            (0.19, "compute_energy_proxy_mare_pct", "0.68", "xxx"),
        ):
            if row[field] == "":
                ax.text(index + offset, 0.30, "N/A", ha="center", va="bottom",
                        fontsize=7.8, color="0.35")
                continue
            value = float(row[field])
            ax.bar(index + offset, value, width=width, color=color,
                   edgecolor="0.18", linewidth=0.8, hatch=hatch, zorder=3,
                   label=("Compute time" if field == "compute_time_mare_pct" else "Energy proxy"))
            ax.annotate(f"{value:.2f}%", (index + offset, value),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.8, zorder=5)
    ax.set_xlim(-0.48, 2.48)
    ax.set_ylim(0, 5.2)
    ax.set_yticks((0, 1, 2, 3, 4, 5))
    ax.set_xticks((0, 1, 2), ("Pi 5", "Orin NX", "Nano Super"))
    ax.set_ylabel("MARE (%)")
    # Use the handles from one device, avoiding duplicated legend entries.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles[:2], labels[:2], loc="upper center", ncol=2,
               bbox_to_anchor=(0.52, 0.98), frameon=True, edgecolor="0.4",
               fontsize=8.0, handlelength=1.6, columnspacing=0.9)
    export(fig, STEMS[0], reference)


def plot_b(data: list[dict], reference) -> None:
    fig, ax = plt.subplots(figsize=reference.cm_to_inch(8.6, 5.2))
    fig.subplots_adjust(left=0.20, right=0.985, bottom=0.23, top=0.78)
    style_axes(ax)
    format_phase_axis(ax, y_for_label=97)
    ax.plot([r["round"] for r in data], [r["online_mean_ape_pct"] for r in data],
            color="#0072B2", linestyle="-", linewidth=1.8, marker="^",
            markersize=4.5, markerfacecolor="white", markeredgewidth=0.9,
            label="Online", zorder=4)
    ax.plot([r["round"] for r in data], [r["frozen_mean_ape_pct"] for r in data],
            color="#4D4D4D", linestyle="--", linewidth=1.5, marker="s",
            markersize=3.8, markerfacecolor="white", markeredgewidth=0.9,
            label="Frozen replay", zorder=3)
    ax.set_ylim(-3, 103)
    ax.set_yticks((0, 25, 50, 75, 100))
    # The device is identified in the caption; the longer label extends beyond
    # the single-column canvas when rotated.
    ax.set_ylabel("Compute-time APE (%)")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=2,
               bbox_to_anchor=(0.54, 0.99), frameon=True, edgecolor="0.4",
               fontsize=8.0, handlelength=1.7, columnspacing=0.8)
    export(fig, STEMS[1], reference)


def plot_c(data: list[dict], reference) -> None:
    fig, ax = plt.subplots(figsize=reference.cm_to_inch(8.6, 5.2))
    fig.subplots_adjust(left=0.20, right=0.985, bottom=0.23, top=0.94)
    style_axes(ax)
    format_phase_axis(ax, y_for_label=632)
    for policy, color, linestyle, marker, line_width, zorder in (
        ("LoRACast", "#0072B2", "-", "^", 1.8, 5),
        ("Fixed-Hetero", "#4D4D4D", "--", "s", 1.4, 3),
        ("PyramidFL-style", "#CC79A7", "-.", "D", 1.4, 4),
    ):
        series = [r for r in data if r["policy"] == policy]
        ax.plot([r["round"] for r in series],
                [r["mean_nx_executed_optimizer_steps"] for r in series],
                color=color, linestyle=linestyle, linewidth=line_width,
                marker=marker, markersize=4.3, markerfacecolor="white",
                markeredgewidth=0.9, label=policy, zorder=zorder)
    ax.set_ylim(300, 650)
    ax.set_yticks((350, 400, 500, 600))
    ax.set_ylabel("NX executed steps")
    ax.legend(loc="center left", bbox_to_anchor=(0.012, 0.64), frameon=True,
              edgecolor="0.5", facecolor="white", fontsize=7.8,
              handlelength=1.55, handletextpad=0.35, borderpad=0.25,
              labelspacing=0.3)
    export(fig, STEMS[2], reference)


def export(fig, stem: str, reference) -> None:
    base = MAIN / stem
    fig.savefig(str(base) + ".svg", bbox_inches=None)
    reference.savefig(fig, str(base), png_dpi=600)


def make_overview() -> None:
    images = []
    for stem in STEMS:
        with Image.open(MAIN / f"{stem}.png") as image:
            copy = image.convert("RGB")
            copy.thumbnail((850, 530), Image.Resampling.LANCZOS)
            images.append(copy)
    width = 850 * 3 + 24 * 4
    height = 590
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font_path = Path("/System/Library/Fonts/Supplemental/Times New Roman.ttf")
    font = ImageFont.truetype(str(font_path), 27) if font_path.is_file() else ImageFont.load_default()
    for index, (image, label) in enumerate(zip(images, (
        "(a) Normal prediction accuracy",
        "(b) Contention prediction error",
        "(c) Runtime workload adaptation",
    ), strict=True)):
        x = 24 + index * (850 + 24)
        sheet.paste(image, (x + (850 - image.width) // 2, 42))
        draw.text((x + 10, 5), label, fill="#222222", font=font)
    OVERVIEW.mkdir(parents=True, exist_ok=True)
    sheet.save(OVERVIEW / "fig5_estimator_contention_overview.png")


def main() -> None:
    MAIN.mkdir(parents=True, exist_ok=True)
    VERIFY.mkdir(parents=True, exist_ok=True)
    reference = load_style()
    sources: dict[str, str] = {}
    a_details, a_summary = figure_a(sources)
    b_seed, b_mean, b_phase, c_seed, c_mean, c_slack = figure_bc(sources)
    nano_normal = sorted((r for r in b_seed if r["round"] in (3, 4)),
                         key=lambda r: (r["seed"], r["round"]))
    if len(nano_normal) != 6 or {(r["seed"], r["round"]) for r in nano_normal} != {
        (seed, round_no) for seed in SEEDS for round_no in (3, 4)
    }:
        raise AssertionError("Nano normal-condition Fig. 5(a) observations incomplete")
    nano_proxy_rows = nano_normal_energy_proxy(nano_normal, sources)
    nano_mare = mean(r["compute_time_ape_pct"] for r in nano_proxy_rows)
    nano_energy_proxy_mare = mean(r["compute_energy_proxy_ape_pct"] for r in nano_proxy_rows)
    close(nano_mare, b_phase[0]["online_time_mare_pct"], "Nano R3-R4 MARE")
    close(nano_energy_proxy_mare, 1.2378314191, "Nano R3-R4 energy proxy MARE", tolerance=1e-9)
    a_summary.append({
        "device": "Nano Super", "n_predictions": 6,
        "compute_time_mare_pct": nano_mare,
        "compute_energy_proxy_mare_pct": nano_energy_proxy_mare,
        "power_settings": "normal R3-R4; seeds 43-45; round-average VDD_IN proxy",
    })
    write_csv(VERIFY / "fig5a_nano_normal_prediction_values.csv", nano_proxy_rows)
    write_csv(VERIFY / "fig5a_normal_prediction_mare.csv", a_summary)
    plot_a(a_summary, reference)
    plot_b(b_mean, reference)
    plot_c(c_mean, reference)
    make_overview()
    outputs = {stem: {extension: str(MAIN / f"{stem}.{extension}")
                      for extension in ("pdf", "svg", "png")} for stem in STEMS}
    audit = {
        "status": "PASS; panel_a_energy_is_warmup_power_times_logged_compute_duration_proxy",
        "script": str(Path(__file__).resolve()),
        "source_sha256": sources,
        "original_records_not_reread_this_run": RAW_UNAVAILABLE,
        "original_record_audit_snapshot": str(PRIOR_VALIDATION) if RAW_UNAVAILABLE else None,
        "original_record_recheck_mode": RECHECK_RAW,
        "seeds_panel_b_c": list(SEEDS),
        "phase_rounds": {"normal": [1, 2, 3, 4], "contention": [5, 6, 7, 8],
                         "recovery": [9, 10, 11, 12]},
        "panel_a": {"runs": ["July MAXN", "July 15W", "July 10W",
                             "October Nano LoRACast-NR seeds 43-45"],
                    "warmup_excluded": [1, 2],
                    "n_observations": len(a_details) + len(nano_normal),
                    "july_observations": len(a_details), "nano_observations": len(nano_normal),
                    "nano_energy_proxy": {
                        "mare_pct": nano_energy_proxy_mare,
                        "n": len(nano_proxy_rows),
                        "formula": "median_R1_R2(nano_active_energy_j/server_active_round_s) * native_pre_round_predicted_compute_s versus (same_round_nano_active_energy_j/server_active_round_s) * observed_nano_compute_s",
                        "measurement_boundary": "VDD_IN power averaged over server-active round, then multiplied by local-compute duration; derived proxy, not native energy prediction or measured compute-only joules",
                    },
                    "summary": a_summary,
                    "energy_boundary": "All devices: R1-R2 calibrated power times pre-round predicted loop time versus observed power times measured loop time. Pi5/NX power is client-logged mean since startup; Nano power is external VDD_IN mean over server-active round. These are distinct derived proxy boundaries, not phase-isolated compute energy.",
                    "plotted_values": str(VERIFY / "fig5a_normal_prediction_mare.csv"),
                    "nano_observation_values": str(VERIFY / "fig5a_nano_normal_prediction_values.csv")},
        "panel_b": {"device": "Jetson Orin Nano Super", "n_seed_rounds": len(b_seed),
                    "warmup_excluded": [1, 2],
                    "online": "native_before_round_server_prediction",
                    "frozen": "offline_prediction_only; each seed's R3 native pre-round prediction held constant at observed Nano S/H/E=320/320/1",
                    "stage_metrics": b_phase,
                    "plotted_values": str(VERIFY / "fig5b_contention_prediction_ape.csv")},
        "panel_c": {"policies": ["C", "F", "P"], "n_seed_rounds": len(c_seed),
                    "metric": "client-reported actual NX local_steps, not requested S or H",
                    "local_compute_diagnostics": str(VERIFY / "fig5c_local_compute_window_diagnostics.csv"),
                    "plotted_values": str(VERIFY / "fig5c_runtime_workload_adaptation.csv")},
        "outputs": outputs,
        "overview": str(OVERVIEW / "fig5_estimator_contention_overview.png"),
        "intended_cm": [8.6, 5.2], "style": "EEFL INFOCOM revision Times New Roman; vector PDF; editable SVG; 600 dpi PNG",
    }
    (VERIFY / "fig5_estimator_contention_validation.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"status": audit["status"], "a": a_summary,
                      "b_stage": b_phase, "c_rows": len(c_mean)}, indent=2))


if __name__ == "__main__":
    main()
