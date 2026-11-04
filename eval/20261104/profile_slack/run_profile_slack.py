#!/usr/bin/env python3
"""Run the isolated PyramidFL profile-slack *component* with the Fig8 RTX core.

The previous Fig8 run_adaptive_training, model inputs, sampler, scorer, and aggregation code
are imported read-only.  This wrapper changes only the workload decision.
The original PyramidFL client selection and update dropout are not executed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from profile_rule import choose_steps

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "experiment_manifest.json"
PROFILE = HERE / "frozen_completion_profile.json"
RUN_ID = "nanonx_s42_dfull_qwen_pyramid_profile_slack_beta05_stress_100r"
BASE = Path(os.environ.get(
    "LORACAST_NANO_BASE_DIR",
    "/mnt/ssd4t/loracast_www_fig8_fig9_20261024_wwjang/edge_protocol",
))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate() -> tuple[dict, dict]:
    manifest = json.loads(MANIFEST.read_text())
    profile = json.loads(PROFILE.read_text())
    if manifest["schema_version"] != 1 or manifest["run_id"] != RUN_ID:
        raise ValueError("unexpected isolated experiment identity")
    if not manifest["static_launch_allowed"] or manifest["rounds"] != 100:
        raise ValueError("launch or horizon gate failed")
    if sha256(PROFILE) != manifest["source_hashes"]["frozen_completion_profile.json"]:
        raise ValueError("completion profile hash changed")
    if sha256(HERE / "profile_rule.py") != manifest["source_hashes"]["profile_rule.py"]:
        raise ValueError("decision source hash changed")
    if sha256(HERE / "run_profile_slack.py") != manifest["source_hashes"]["run_profile_slack.py"]:
        raise ValueError("run_adaptive_training source hash changed")
    for key, filename in (
        ("existing_nano_qwen_rtx_run_adaptive_training.py", "nano_qwen_rtx_run_adaptive_training.py"),
        ("existing_edge_profile_oracle.py", "edge_profile_oracle.py"),
        ("existing_static_manifest.json", "nano_experiment_manifest.json"),
    ):
        if sha256(BASE / filename) != manifest["source_hashes"][key]:
            raise ValueError(f"existing Fig8 source changed: {filename}")
    decisions = choose_steps(profile, {"nano": 28143, "nx": 18441})
    if decisions != manifest["new_policy_frozen_decision"]:
        raise ValueError("manifest workload decision changed")
    if any(not (320 <= d["H"] <= 600 and d["H"] == d["S"] and d["E"] == 1)
           for d in decisions.values()):
        raise ValueError("common workload limits violated")
    return manifest, decisions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--stop-after-round", type=int, choices=(1,))
    args = parser.parse_args()
    manifest, decisions = validate()
    if args.validate_only:
        print(json.dumps({"status": "PREFLIGHT_PASS", "run_id": RUN_ID,
                          "manifest_sha256": sha256(MANIFEST), "decisions": decisions}, indent=2))
        return
    os.environ.setdefault("LORACAST_NANO_RTX_APPROVED", "1")
    os.environ.setdefault("LORACAST_RTX_CORE_DIR", str(BASE.parent / "quality_campaign"))
    sys.path.insert(0, str(BASE))
    import nano_qwen_rtx_runner as original

    original.MANIFEST = MANIFEST
    original.OUT_ROOT = HERE / "rtx_runs"

    def profile_slack_quota(policy: str, rnd: int) -> dict:
        if policy != "PyramidFL-profile-slack" or not 1 <= rnd <= 100:
            raise ValueError("unexpected policy or round")
        return {
            "nano": (decisions["nano"]["S"], 1, decisions["nano"]["H"]),
            "nx": (decisions["nx"]["S"], 1, decisions["nx"]["H"]),
            "origin": "official_PyramidFL_profile_completion_positive_slack_iteration_rule_beta_0.5; frozen_profile_sha256=" + sha256(PROFILE),
        }

    original.frozen_quota = profile_slack_quota
    original.execute(RUN_ID, args.stop_after_round)


if __name__ == "__main__":
    main()
