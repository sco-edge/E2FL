#!/usr/bin/env python3
"""Freeze an isolated, single-policy RTX comparison before R0."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from profile_rule import choose_steps

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
OLD = PROJECT / "training_cost_projection" / "edge_protocol"
STRESS = PROJECT / "training_cost_projection" / "nano_r93_100_stress_branch" / "experiment_manifest.json"
UPSTREAM = Path("/Users/jang-won-u/Documents/GitHub/EEFL/_SOTA/PyramidFL-main/training/param_server.py")
PROFILE = HERE / "frozen_completion_profile.json"
OUTPUT = HERE / "experiment_manifest.json"
RUN_ID = "nanonx_s42_dfull_qwen_pyramid_profile_slack_beta05_stress_100r"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict:
    profile = json.loads(PROFILE.read_text())
    chosen = choose_steps(profile, {"nano": 28143, "nx": 18441})
    stress = json.loads(STRESS.read_text())
    if [s["extension_round"] for s in stress["schedule"]] != list(range(93, 101)):
        raise ValueError("matched stress schedule is incomplete")
    return {
        "schema_version": 1,
        "experiment_id": "pyramidfl_original_profile_slack_component_beta05_nanonx_qwen_s42_dfull_100r",
        "run_id": RUN_ID,
        "runs": [{"policy": "PyramidFL-profile-slack", "run_id": RUN_ID}],
        "static_launch_allowed": True,
        "seed": 42,
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "partition": "original Strong DFull Nano+NX mapping",
        "rounds": 100,
        "batch_size": 1,
        "sequence_cap": 64,
        "local_epochs": 1,
        "aggregation": "actual executed-sample-weighted LoRA A/B factors, same RTX core as existing Fig8",
        "validation": "same held-out full-sequence 64-token scorer at R0 and every 10 rounds",
        "baseline_module": "public PyramidFL completion/slack adaptive local iteration rule only; no selection or update dropout",
        "step_translation": "upstream upload_epoch local iteration count -> Flower optimizer H, with S=H, B=1, E=1 and fresh DFull samples",
        "beta": 0.5,
        "profile_updates_during_run": False,
        "new_policy_frozen_decision": chosen,
        "common_max_steps": 600,
        "stress_interpretation": "R1-R92 normal; R93-R96 Nano contention; R97-R100 recovery; apply measured factors to Nano local-compute cost after each completed RTX round; static profile rule never sees those factors",
        "stress_schedule": stress["schedule"],
        "source_hashes": {
            "official_pyramidfl_param_server.py": sha256(UPSTREAM),
            "hardware_calibration_csv": profile["source_sha256"],
            "frozen_completion_profile.json": sha256(PROFILE),
            "profile_rule.py": sha256(HERE / "profile_rule.py"),
            "run_profile_slack.py": sha256(HERE / "run_profile_slack.py"),
            "existing_nano_qwen_rtx_run_adaptive_training.py": sha256(OLD / "nano_qwen_rtx_run_adaptive_training.py"),
            "existing_edge_profile_oracle.py": sha256(OLD / "edge_profile_oracle.py"),
            "existing_static_manifest.json": sha256(OLD / "nano_experiment_manifest.json"),
            "matched_stress_manifest.json": sha256(STRESS),
        },
        "source_paths": {
            "official_pyramidfl_param_server.py": str(UPSTREAM),
            "hardware_calibration_csv": profile["source_path"],
            "existing_edge_protocol": str(OLD),
            "matched_stress_manifest": str(STRESS),
        },
        "status": "frozen_before_R0",
    }


if __name__ == "__main__":
    payload = build()
    if OUTPUT.exists() and json.loads(OUTPUT.read_text()) != payload:
        raise ValueError("existing frozen manifest differs")
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(OUTPUT), "sha256": sha256(OUTPUT),
                      "decision": payload["new_policy_frozen_decision"]}, indent=2))
