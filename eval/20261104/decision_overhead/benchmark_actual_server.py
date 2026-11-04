#!/usr/bin/env python3
"""Server-only timing of the shipped Flower LoRACast decision hook.

This harness does not reimplement allocation. It invokes the strategy's
update_train_config method on a fresh Flower ConfigRecord for every decision.
The timer includes the config copy that configure_train performs immediately
before the hook. Flower message creation, networking and client work are out
of scope. Synthetic multi-client histories are explicitly marked in output.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import math
import os
import platform
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    seq = sorted(values)
    x = (len(seq) - 1) * q
    a = int(math.floor(x))
    b = int(math.ceil(x))
    return seq[a] + (seq[b] - seq[a]) * (x - a)


def cpu_model() -> str:
    path = Path("/proc/cpuinfo")
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unavailable"


def actual_anchors(csv_path: Path) -> dict[int, dict[str, dict]]:
    out: dict[int, dict[str, dict]] = {}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("phase") != "train" or row.get("device_name") not in {"RPi5", "jetson_orin_nx"}:
                continue
            r = int(row["round"])
            if r not in (1, 2):
                continue
            device = row["device_name"]
            if device in out.setdefault(r, {}):
                raise RuntimeError("anchor CSV has duplicate physical device/round")
            out[r][device] = {
                "samples": int(row["num_examples"]),
                "tokens": int(row["input_tokens"]),
                "loop_time": float(row["loop_time"]),
                "upload_prep_time": float(row["client_upload_elapsed_time"] or 0),
                "uid": row["device_uid"],
            }
    if set(out) != {1, 2} or any(set(out[r]) != {"RPi5", "jetson_orin_nx"} for r in out):
        raise RuntimeError("physical anchor requires Pi 5 and NX training in R1/R2")
    return out


def make_histories(n: int, anchors: dict[int, dict[str, dict]], *, case: str, seed: int) -> tuple[dict, list]:
    """Nine completed rounds before the measured R10 decision.

    Measured R1/R2 device rates anchor synthetic replay states. One NX and
    N-1 Pi clients are present. The NX schedules sum to the matching exposure
    state (72, 232, or 256 samples) where relevant. This is a fixed-state
    server stress test, not an observed multi-client FL trace.
    """
    rng = random.Random(seed + 1009 * n)
    round_stats: dict[int, dict] = {}
    window_history: list[dict] = []
    schedules = {
        "normal": [8] * 9,
        "trigger": [8, 64, 64, 64, 64, 64, 64, 64, 64],
        "nonbinding": [8] * 9,
        "partial": [8, 64, 32, 32, 24, 24, 16, 16, 16],
        "exhausted": [8, 64, 32, 32, 24, 24, 24, 24, 24],
    }
    nx_schedule = schedules[case]
    for r, nx_samples in enumerate(nx_schedule, start=1):
        anchor_round = 1 if r == 1 else 2
        pi_anchor, nx_anchor = anchors[anchor_round]["RPi5"], anchors[anchor_round]["jetson_orin_nx"]
        clients: dict[str, dict] = {}
        pi_entries = []
        for j in range(n - 1):
            uid = f"rpi5-bench-{j:03d}"
            # The first Pi is exactly the measured anchor in a normal N=2 case.
            factor = 1.0 if j == 0 else rng.uniform(0.82, 1.18)
            undercompleted = case == "trigger" and j % 4 == 0
            sample_fraction = 0.15 if undercompleted else 1.0
            samples = max(1, round(pi_anchor["samples"] * sample_fraction))
            tokens = max(1, round(pi_anchor["tokens"] * factor * samples / pi_anchor["samples"]))
            loop_time = pi_anchor["loop_time"] * factor * samples / pi_anchor["samples"]
            entry = {"cid": str(j + 1), "device_uid": uid, "device_name": "RPi5",
                     "tokens": tokens, "tokens_per_client": tokens, "train_time": loop_time,
                     "num_examples": samples}
            clients[f"RPi5:{uid}"] = entry
            pi_entries.append(entry)
        nx_uid = "jetson-orin-nx-bench-0"
        nx_tokens = max(1, round(nx_anchor["tokens"] * nx_samples / nx_anchor["samples"]))
        nx_loop_time = nx_anchor["loop_time"] * nx_samples / nx_anchor["samples"]
        clients[f"jetson_orin_nx:{nx_uid}"] = {
            "cid": str(n), "device_uid": nx_uid, "device_name": "jetson_orin_nx",
            "tokens": nx_tokens, "tokens_per_client": nx_tokens,
            "train_time": nx_loop_time, "num_examples": nx_samples,
        }
        pi_mean_tokens = statistics.mean(e["tokens"] for e in pi_entries)
        pi_mean_time = statistics.mean(e["train_time"] for e in pi_entries)
        round_stats[r] = {
            "clients": clients,
            "RPi5": {"tokens": pi_mean_tokens, "tokens_per_client": pi_mean_tokens,
                     "train_time": pi_mean_time, "num_examples": statistics.mean(e["num_examples"] for e in pi_entries)},
            "jetson_orin_nx": {"tokens": nx_tokens,
                                "tokens_per_client": nx_tokens,
                                "train_time": nx_loop_time,
                                "num_examples": nx_samples},
        }
        window_history.append({"RPi5": sum(e["tokens"] for e in pi_entries),
                               "jetson_orin_nx": nx_tokens})
    return round_stats, window_history


def base_config():
    from flwr.app import ConfigRecord
    # Values mirror the frozen physical Pi5/NX integration configuration,
    # including the deployed rho=0.6. Trigger states are supplied by
    # synthetic under-completion histories rather than altering the threshold.
    return ConfigRecord({
        "batch_size": 1, "local_epochs": 1, "local_steps": 0,
        "max_samples_per_round": 9, "max_seq_len": 64, "lora-r": 16,
        "participation-guard-mode": "per-client",
        "participation-fairness-rho": 0.6,
        "policy-warmup-rounds": 1, "window-W": 5,
        "rpi-normal-quota": 9, "rpi-sample-step": 1,
        "rpi-correction-max-quota": 9, "max-correction-time-s": 120.0,
        "jetson-base-quota": 8, "jetson-max-quota": 64,
        "jetson-max-steps": 64, "jetson-sample-step": 4,
        "jetson-step-increment": 1, "jetson-target-margin": 0.95,
    })


def make_strategy(policy: str, case: str, n: int, histories: tuple, app_root: Path,
                  output_dir: Path, repetition: int):
    from e2fl.strategy.LoRACast import LoRACastCommAware
    from e2fl.strategy.FinalExposureLoRACast import FinalExposureLoRACast
    kwargs = dict(
        fraction_train=1.0, fraction_evaluate=1.0, min_available_nodes=n,
        log_dir=str(Path("strategy_logs") / policy / case / str(n) / str(repetition)),
        ROOT=str(output_dir.resolve()), model_name="Qwen2.5-0.5B-Instruct", latte=False,
        model_info=None, device_lookup={j + 1: "RPi5" for j in range(n - 1)} | {n: "jetson_orin_nx"},
        device_profiles_path=str(app_root / "eval/profiling/20260218/profiling_merged.json"),
        tokens_per_example=64.0, enable_rpi_only_rounds=True, use_token_ratio=True,
        enable_deadline_guard=False, participation_guard_mode="per-client",
        participation_fairness_rho=0.6, aggregation_mode="processed_samples",
        experiment_id="", rebalancing_enabled=(policy == "Rebalancing"),
        comm_ema_alpha=0.3, comm_safety_factor=1.0,
        use_comm_prediction=False, static_allocation_after_round=0,
        residual_calibration_enabled=True, estimator_freeze_after_round=0,
    )
    if policy == "Exposure-(2)":
        strategy = FinalExposureLoRACast(
            exposure_max=2.0, nx_eligible_unique=128, nx_minimum_samples=0,
            nx_maximum_samples=64, nx_maximum_epochs=1, **kwargs,
        )
        strategy.nx_cumulative_samples = {"nonbinding": 72, "partial": 232,
                                          "exhausted": 256}[case]
        strategy.nx_unique_visited = min(strategy.nx_cumulative_samples, 128)
        strategy.nx_seen_ids = set(range(strategy.nx_unique_visited))
    else:
        strategy = LoRACastCommAware(**kwargs)
    strategy.round_stats, strategy.window_history = histories
    strategy.slowest_device_name = "RPi5"
    return strategy


def verify_decision(policy: str, case: str, n: int, cfg, strategy) -> dict:
    mode = str(cfg.get("round_mode", ""))
    pi = int(cfg.get("max_samples_per_round_RPi5", 0) or 0)
    nx = int(cfg.get("max_samples_per_round_jetson_orin_nx", 0) or 0)
    nx_h = int(cfg.get("local_steps_jetson_orin_nx", 0) or 0)
    result = {"policy": policy, "state": case, "clients": n, "round_mode": mode,
              "pi_samples": pi, "nx_samples": nx, "nx_steps": nx_h,
              "uid_assignments": sum(k.startswith("max_samples_per_round_uid_") for k in cfg),
              "protected_count": len(strategy.protected_client_ids)}
    if mode not in ("BOTH", "PROTECTED_ONLY", "RPIONLY") or pi <= 0:
        raise AssertionError(f"invalid controller decision: {result}")
    if result["uid_assignments"] != n:
        raise AssertionError(f"missing client workload decisions: {result}")
    for uid, device in strategy._known_client_device_map().items():
        suffix = strategy._safe_config_suffix(uid)
        assigned = int(cfg[f"max_samples_per_round_uid_{suffix}"])
        if device == "RPi5" and not (0 < assigned <= strategy.rpi_correction_max_quota):
            raise AssertionError(f"invalid Pi quota for {uid}: {assigned}")
        if device == "jetson_orin_nx" and not (0 <= assigned <= strategy.jetson_max_quota):
            raise AssertionError(f"invalid NX quota for {uid}: {assigned}")
    if policy == "Rebalancing" and case == "trigger" and mode != "PROTECTED_ONLY":
        raise AssertionError(f"synthetic deficit failed to trigger correction: {result}")
    if policy == "Rebalancing" and case == "normal" and mode != "BOTH":
        raise AssertionError(f"normal-history correction unexpectedly triggered: {result}")
    if policy == "Unbounded" and mode != "BOTH":
        raise AssertionError(f"unbounded path corrected: {result}")
    if policy == "Exposure-(2)":
        decision = strategy.exposure_decisions[10]
        result.update({"budget_before": decision.get("remaining_before"),
                       "budget_action": decision["action"],
                       "expected_processed_samples": decision.get("expected_processed_samples")})
        if case == "nonbinding" and (decision["action"] != "normal" or nx <= 0):
            raise AssertionError(f"nonbinding case did not remain normal: {result}")
        if case == "partial" and not (0 < nx < int(decision["normal_s"]) and nx <= 24):
            raise AssertionError(f"partial budget not respected: {result}")
        if case == "exhausted" and not (nx == 0 and nx_h == 0 and mode == "PROTECTED_ONLY"):
            raise AssertionError(f"exhaustion did not omit NX pretraining: {result}")
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("refusing empty CSV")
    fields = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--app-root", type=Path, required=True)
    p.add_argument("--source-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--warmups", type=int, default=20)
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--counts", nargs="+", type=int, default=[2, 4, 8, 16, 32, 64, 128, 250])
    p.add_argument("--seed", type=int, default=20271024)
    p.add_argument("--reference-commit", default="")
    args = p.parse_args()
    started_at = datetime.now(timezone.utc).isoformat()
    loadavg_at_start = os.getloadavg()
    sys.path.insert(0, str(args.app_root.resolve()))
    logging.disable(logging.INFO)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    anchors = actual_anchors(args.source_csv)
    from flwr.app import ConfigRecord  # noqa: F401; checks installed Flower
    import flwr

    raw, decisions = [], []
    policies = [
        ("Unbounded", "normal"),
        ("Rebalancing", "normal"), ("Rebalancing", "trigger"),
        ("Exposure-(2)", "nonbinding"), ("Exposure-(2)", "partial"),
        ("Exposure-(2)", "exhausted"),
    ]
    base = base_config()
    for n in args.counts:
        if n < 2:
            raise ValueError("need one Pi and one NX")
        for policy, case in policies:
            histories = make_histories(n, anchors, case=case, seed=args.seed)
            for rep in range(args.repetitions):
                strategy = make_strategy(policy, case, n, histories, args.app_root,
                                         args.output_dir, rep)
                # State persists through trials. Rebalancing trigger trials
                # restore only the cooldown marker so each decision sees the
                # same pre-round history; estimator caches remain warmed.
                for trial in range(-args.warmups, args.trials):
                    if policy == "Rebalancing" and case == "trigger":
                        strategy.last_per_client_correction_round = -1
                    t0 = time.perf_counter_ns()
                    cfg = strategy.update_train_config(10, copy.deepcopy(base))
                    elapsed = time.perf_counter_ns() - t0
                    if trial in (-args.warmups, 0):
                        decision = verify_decision(policy, case, n, cfg, strategy)
                        decision.update({"repetition": rep, "trial": trial})
                        decisions.append(decision)
                    if trial >= 0:
                        raw.append({"clients": n, "policy": policy, "state": case,
                                    "repetition": rep, "trial": trial,
                                    "latency_ns": elapsed, "latency_ms": elapsed / 1e6})
            print(f"completed N={n} policy={policy} state={case}", flush=True)
    summaries = []
    for n in args.counts:
        for policy, case in policies:
            values = [r["latency_ms"] for r in raw if r["clients"] == n
                      and r["policy"] == policy and r["state"] == case]
            summaries.append({"clients": n, "policy": policy, "state": case,
                              "observations": len(values), "mean_ms": statistics.mean(values),
                              "median_ms": statistics.median(values),
                              "p95_ms": percentile(values, 0.95),
                              "p99_ms": percentile(values, 0.99),
                              "std_ms": statistics.stdev(values) if len(values) > 1 else 0.0})
    write_csv(args.output_dir / "raw_timings.csv", raw)
    write_csv(args.output_dir / "summary.csv", summaries)
    write_csv(args.output_dir / "decision_checks.csv", decisions)
    source_files = [
        args.app_root / "e2fl/strategy/FedAvgLoRA.py",
        args.app_root / "e2fl/strategy/LoRACast.py",
        args.app_root / "e2fl/strategy/FinalExposureLoRACast.py",
        args.app_root / "exposure_controller.py",
        args.app_root / "pyproject.toml",
        args.app_root / "eval/profiling/20260218/profiling_merged.json",
        args.source_csv,
        Path(__file__),
    ]
    metadata = {
        "boundary": "deepcopy(Flower ConfigRecord) + actual strategy.update_train_config(round=10)",
        "exclusions": ["Flower message construction", "aggregation", "network", "client training", "inference"],
        "source_sha256": {str(x.resolve()): sha256(x) for x in source_files},
        "flwr_version": flwr.__version__, "python": platform.python_version(),
        "platform": platform.platform(), "processor": platform.processor(),
        "cpu_model": cpu_model(), "logical_cpu_count": os.cpu_count(),
        "reference_git_commit": args.reference_commit,
        "started_utc": started_at,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "uname": platform.uname()._asdict(),
        "loadavg_at_start": loadavg_at_start,
        "loadavg_at_end": os.getloadavg(), "counts": args.counts,
        "policies_and_states": policies, "warmups": args.warmups,
        "trials_per_repetition": args.trials, "repetitions": args.repetitions,
        "seed": args.seed,
        "client_state": "one NX and N-1 Pi5; measured R1/R2 per-workload rates anchor nine-round synthetic histories; N-2 extra Pi5 clients have deterministic +/-18% variation; trigger history has synthetic undercompletion for every fourth Pi",
        "exposure_state": "D_NX=128, c=2; nine previous rounds sum to NX cumulative samples: nonbinding=72, partial=232, exhausted=256; unique visited=min(cumulative,128); all are synthetic policy states",
        "comm_prediction": "disabled to match physical Pi5/NX integration run; actual decision path does not invoke energy prediction",
        "logging": "Flower INFO and lower suppressed; policy decision JSON writes remain inside measured hook",
        "distributed_2": "unsupported: no Flower server strategy is registered for this policy",
    }
    (args.output_dir / "benchmark_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
