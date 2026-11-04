#!/usr/bin/env python3
"""Correctness-only replay of the shipped Flower decision strategies.

This invokes actual server decision methods. It does not train a model or
fabricate client replies. A selected exposure quota is manually committed as
an exactly executed sample count, with this replay explicitly identified in
the output. The cost/history inputs remain frozen to the benchmark's
synthetic pre-R10 stress state for the transition check.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from benchmark_actual_server import (
    actual_anchors,
    base_config,
    make_histories,
    make_strategy,
)


def decision_record(cfg, strategy, round_no: int, n: int) -> dict:
    per_uid = {}
    for uid, device in strategy._known_client_device_map().items():
        suffix = strategy._safe_config_suffix(uid)
        per_uid[uid] = {
            "device": device,
            "samples": int(cfg[f"max_samples_per_round_uid_{suffix}"]),
            "steps_cap": int(cfg.get(f"local_steps_uid_{suffix}", 0) or 0),
        }
    if len(per_uid) != n:
        raise AssertionError(f"expected {n} per-client decisions, got {len(per_uid)}")
    nx = [x for x in per_uid.values() if x["device"] == "jetson_orin_nx"]
    if len(nx) != 1:
        raise AssertionError(f"expected one NX, got {len(nx)}")
    return {
        "round": round_no,
        "mode": str(cfg["round_mode"]),
        "pi_device_samples": int(cfg["max_samples_per_round_RPi5"]),
        "nx_device_samples": int(cfg["max_samples_per_round_jetson_orin_nx"]),
        "nx_device_steps_cap": int(cfg["local_steps_jetson_orin_nx"]),
        "nx_uid_samples": nx[0]["samples"],
        "nx_uid_steps_cap": nx[0]["steps_cap"],
        "protected_client_count": len(strategy.protected_client_ids),
        "per_uid_count": len(per_uid),
        "last_rebalancing_round": int(strategy.last_per_client_correction_round),
    }


def check_rebalancing(n: int, anchors: dict, app_root: Path, output_dir: Path, seed: int) -> dict:
    histories = make_histories(n, anchors, case="trigger", seed=seed)
    strategy = make_strategy("Rebalancing", "trigger", n, histories, app_root, output_dir, 987)
    base = base_config()
    # Consecutive decisions exercise the actual one-round cooldown marker.
    # No fabricated R10 client reply is inserted between calls.
    cfg10 = strategy.update_train_config(10, copy.deepcopy(base))
    r10 = decision_record(cfg10, strategy, 10, n)
    if r10["mode"] != "PROTECTED_ONLY" or r10["nx_device_samples"] != 0:
        raise AssertionError(f"R10 did not trigger omission: {r10}")
    if r10["last_rebalancing_round"] != 10:
        raise AssertionError(f"R10 cooldown marker was not set: {r10}")
    cfg11 = strategy.update_train_config(11, copy.deepcopy(base))
    r11 = decision_record(cfg11, strategy, 11, n)
    if r11["mode"] != "BOTH" or r11["nx_device_samples"] <= 0:
        raise AssertionError(f"R11 did not return to normal: {r11}")
    if r11["last_rebalancing_round"] != 10:
        raise AssertionError(f"R11 unexpectedly advanced correction marker: {r11}")
    return {
        "clients": n,
        "check": "rebalancing_trigger_then_one_round_cooldown",
        "status": "PASS",
        "rounds": [r10, r11],
        "replay_boundary": "Same pre-R10 synthetic completed-work history is held fixed; no R10 execution or replies are invented.",
    }


def check_exposure(n: int, anchors: dict, app_root: Path, output_dir: Path, seed: int) -> dict:
    histories = make_histories(n, anchors, case="partial", seed=seed)
    strategy = make_strategy("Exposure-(2)", "partial", n, histories, app_root, output_dir, 988)
    base = base_config()
    cap = int(strategy.exposure_max * strategy.nx_eligible_unique)
    initial_cumulative = int(strategy.nx_cumulative_samples)
    if initial_cumulative != 232 or cap != 256:
        raise AssertionError(f"unexpected frozen state: cumulative={initial_cumulative}, cap={cap}")

    rounds = []
    for round_no in range(10, 30):
        before = int(strategy.nx_cumulative_samples)
        cfg = strategy.update_train_config(round_no, copy.deepcopy(base))
        entry = decision_record(cfg, strategy, round_no, n)
        policy_decision = strategy.exposure_decisions[round_no]
        selected = int(policy_decision["selected_s"])
        expected_executed = int(policy_decision.get("expected_processed_samples", selected))
        if expected_executed != selected:
            raise AssertionError(f"E=1/B=1 executed samples differ from S: {policy_decision}")
        if selected != entry["nx_device_samples"] or selected != entry["nx_uid_samples"]:
            raise AssertionError(f"server and per-UID quotas differ: {entry}")
        if before + expected_executed > cap:
            raise AssertionError(f"exposure budget exceeded: before={before}, selected={selected}")
        entry.update({
            "action": str(policy_decision["action"]),
            "normal_s": int(policy_decision["normal_s"]),
            "remaining_before": int(policy_decision["remaining_before"]),
            "cumulative_before": before,
            "manually_replayed_executed_samples": expected_executed,
            "cumulative_after_replay": before + expected_executed,
            "remaining_after_replay": cap - before - expected_executed,
        })
        rounds.append(entry)
        if selected == 0:
            native_minimum = int(policy_decision.get("native_minimum_s", strategy.jetson_base_quota))
            if not (cap - before < native_minimum) or entry["mode"] != "PROTECTED_ONLY" or entry["nx_device_steps_cap"] != 0:
                raise AssertionError(f"budget exhaustion failed to omit NX pretraining: {entry}")
            break
        # This is manual state replay of exactly the controller's selected S.
        # It deliberately avoids constructing fake Flower replies or claiming
        # the real aggregate_train path has run.
        strategy.nx_cumulative_samples += expected_executed
        strategy.nx_unique_visited = min(strategy.nx_cumulative_samples, strategy.nx_eligible_unique)
    else:
        raise AssertionError("did not reach omission within 20 replayed rounds")
    if not rounds or not (0 < rounds[0]["nx_device_samples"] < rounds[0]["normal_s"]):
        raise AssertionError(f"initial partial quota missing: {rounds[:1]}")
    if rounds[-1]["remaining_before"] >= int(strategy.jetson_base_quota) or rounds[-1]["action"] != "omit_before_training":
        raise AssertionError(f"final budget state invalid: {rounds[-1]}")
    return {
        "clients": n,
        "check": "exposure_partial_then_budget_exhaustion",
        "status": "PASS",
        "eligible_unique": int(strategy.nx_eligible_unique),
        "exposure_max": float(strategy.exposure_max),
        "sample_cap": cap,
        "initial_cumulative_samples": initial_cumulative,
        "rounds": rounds,
        "replay_boundary": "After each decision, NX cumulative samples are manually advanced by exactly selected executed samples. No LLM training, client report, aggregation, or checkpoint update occurs; cost/history inputs remain fixed.",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--app-root", type=Path, required=True)
    p.add_argument("--source-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20271024)
    args = p.parse_args()
    sys.path.insert(0, str(args.app_root.resolve()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    anchors = actual_anchors(args.source_csv)
    results = []
    for n in (2, 250):
        results.append(check_rebalancing(n, anchors, args.app_root, args.output_dir, args.seed))
        results.append(check_exposure(n, anchors, args.app_root, args.output_dir, args.seed))
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "purpose": "Server decision state-transition correctness; not a latency or FL quality result.",
        "profile_basis": "Physical Pi5/NX R1/R2 anchors plus fixed synthetic histories from benchmark_actual_server.py.",
        "policy_implementation": "Copied, unchanged physical_pi5_app Flower strategies.",
        "results": results,
    }
    (args.output_dir / "state_transition_checks.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "checks": len(results),
                      "output": str(args.output_dir / "state_transition_checks.json")}, indent=2))


if __name__ == "__main__":
    main()
