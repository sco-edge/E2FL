#!/usr/bin/env python3
"""Freeze the two static policies and the shared R50–R100 Nano trace.

This script prepares metadata only. It never starts RTX learning or edits any
earlier run. The corresponding training wrapper imports the existing Nano RTX
run_adaptive_training and its original model, sampler, evaluator, and aggregation code.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NANO = ROOT / "training_cost_projection/edge_protocol"
PYRAMID = ROOT / "profile_slack"
BRANCH = ROOT / "repeated_contention"
PHYSICAL = Path("/Users/jang-won-u/Documents/Codex/2026-10-10/new-chat/outputs/nano_nx_profile_slack_20261026")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    branch = json.loads((BRANCH / "experiment_manifest.json").read_text())
    profile = json.loads((PYRAMID / "experiment_manifest.json").read_text())
    static = json.loads((NANO / "nano_experiment_manifest.json").read_text())
    factors = branch["factor_rounds"]
    if [row["round"] for row in factors] != list(range(50, 101)):
        raise ValueError("source repeated-contention schedule is incomplete")
    if [row["phase"] for row in factors[:12]] != ["normal"] * 4 + ["contention"] * 4 + ["recovery"] * 4:
        raise ValueError("source first stress cycle changed")
    if branch["source_run_common"]["seed"] != 42 or static["seed"] != 42 or profile["seed"] != 42:
        raise ValueError("source seed mismatch")
    if branch["source_run_common"]["strong_partition_mapping"] != static["strong_partition_mapping"]:
        raise ValueError("partition mapping mismatch")
    decisions = profile["new_policy_frozen_decision"]
    if (decisions["nano"]["H"], decisions["nx"]["H"]) != (320, 342):
        raise ValueError("profile-slack fixed decision changed")

    schedule = [
        {"round": rnd, "phase": "normal", "fig9_reference_round": None,
         "nano_compute_seconds_per_token_factor": 1.0}
        for rnd in range(1, 50)
    ] + factors
    if [row["round"] for row in schedule] != list(range(1, 101)):
        raise ValueError("100-round schedule has a gap")

    manifest = {
        "schema_version": 1,
        "experiment_id": "nanonx_s42_dfull_qwen_three_policy_matched_repeat4x_r50_100_20261026",
        "status": "prepared_for_isolated_smoke_then_sequential_training",
        "static_launch_allowed": True,
        "claim_scope": "Actual RTX adapter training and balanced validation NLL; measured Nano contention trace affects adaptive Flower feedback but for these two fixed-quota policies it is post hoc local-compute timing only.",
        "run_common": {
            "seed": 42,
            "rounds": 100,
            "model": static["model"],
            "optimizer": static["optimizer"],
            "lora": static["lora"],
            "batch_size": static["batch_size"],
            "local_epochs": static["local_epochs"],
            "sequence_cap": static["sequence_cap"],
            "strong_partition_mapping": static["strong_partition_mapping"],
            "source_sampler": static["source_sampler"],
            "holdout": static["holdout"],
            "evaluation_rounds": static["evaluation_rounds"],
            "evaluation_metric": static["evaluation_metric"],
            "aggregation": static["aggregation"],
        },
        "runs": [
            {"run_id": "nanonx_s42_dfull_qwen_baseline_repeat4x_r50_100_v1",
             "policy": "Baseline", "requested_steps": {"nano": 320, "nx": 320},
             "source_workload": {"nano": [320, 1, 0], "nx": [320, 1, 0]}},
            {"run_id": "nanonx_s42_dfull_qwen_pyramid_profile_slack_repeat4x_r50_100_v1",
             "policy": "PyramidFL-profile-slack", "requested_steps": {"nano": 320, "nx": 342},
             "source_workload": {"nano": [320, 1, 320], "nx": [342, 1, 342]}},
        ],
        "profile_slack_component": {
            "beta": profile["beta"],
            "base_optimizer_steps": 320,
            "frozen_decision": decisions,
            "full_pyramidfl_system": False,
        },
        "stress_application": "R1-R49 normal; R50-R100 four normal/four contention/four recovery. For Baseline and frozen PyramidFL profile-slack, keep S=H and E=1 fixed. Apply Nano measured factor only to a separate post hoc edge local-compute time projection; it cannot change the RTX NLL trajectory.",
        "factor_rounds": schedule,
        "source_hashes": {
            "nano_qwen_rtx_run_adaptive_training.py": sha256(NANO / "nano_qwen_rtx_run_adaptive_training.py"),
            "edge_profile_oracle.py": sha256(NANO / "edge_profile_oracle.py"),
            "nano_experiment_manifest.json": sha256(NANO / "nano_experiment_manifest.json"),
            "run_qwen_training.py": sha256(NANO.parent / "quality_campaign/run_qwen_training.py"),
            "profile_rule.py": sha256(PYRAMID / "profile_rule.py"),
            "frozen_completion_profile.json": sha256(PYRAMID / "frozen_completion_profile.json"),
            "repeated_contention_manifest.json": sha256(BRANCH / "experiment_manifest.json"),
        },
        "expected_inputs": {
            "initial_adapter_round0.pt": "b3ff342f8a6a0b9c42263a311fd6a845ccad2cbba106e84c485779b9cfd0511d",
            "loracast_strong.json": "ee13a0fb12e4f917a01b6ff6fe5fc1f1b867fed3599e1c1dcd79ae36451b6493",
            "fixed_partitions.json": "8b700bd14f40bfce951971fe929a641f49c0730bf131b827494480f49d559371",
            "prior_verified_serialized_R0_adapter": "7ae6c591bc59c7c31f781e4b5c689b19ef5703b23aaa732a9e0b9d8cb4f8975b",
        },
        "remote_workdir": "/mnt/ssd4t/loracast_www_fig8_fig9_20261024_wwjang/edge_protocol/matched_repeat4x_static_20261026",
        "source_run_ids": {
            "baseline": "nanonx_s42_dfull_qwen_baseline_100r",
            "profile_slack": profile["run_id"],
            "schedule": branch["branch_run_id"],
        },
        "physical_workload_provenance": {
            "scope": "Independent physical Nano Super + Orin NX 12-round profile-slack verification, seeds 43–45. Supports the 320/342 fixed workload only; its IID training and evaluation data are not the seed-42 Strong DFull RTX 100-round protocol.",
            "campaign_complete_path": str(PHYSICAL / "campaign_complete.json"),
            "campaign_complete_sha256": sha256(PHYSICAL / "campaign_complete.json"),
            "comparison_source_manifest_path": str(PHYSICAL / "comparison/source_manifest.json"),
            "comparison_source_manifest_sha256": sha256(PHYSICAL / "comparison/source_manifest.json"),
        },
    }
    manifest_path = HERE / "experiment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    locked_files = ("experiment_manifest.json", "prepare_manifest.py", "run_static_training.py", "launch_static.sh")
    if not all((HERE / name).is_file() for name in locked_files):
        print(f"Wrote {manifest_path}; create remaining wrapper files before locking")
        return
    lock = {name: sha256(HERE / name) for name in locked_files}
    (HERE / "artifact_lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PREPARED", "manifest_sha256": lock["experiment_manifest.json"],
                      "rounds": len(schedule), "runs": [r["run_id"] for r in manifest["runs"]]}, indent=2))


if __name__ == "__main__":
    main()
