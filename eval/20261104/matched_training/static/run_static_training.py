#!/usr/bin/env python3
"""R0–R100 Nano+NX static-policy RTX run_adaptive_training using the existing Fig. 8 core.

This wrapper selects only previously frozen S/H/E workloads. Its stress
schedule is deliberately metadata only for these fixed policies. Actual
training, sample selection, validation, and aggregation remain in the
read-only nano_qwen_rtx_run_adaptive_training.py and its existing RTX core.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REMOTE_BASE = "/mnt/ssd4t/loracast_www_fig8_fig9_20261024_wwjang/edge_protocol"
BASE = Path(os.environ.get("LORACAST_NANO_BASE_DIR", REMOTE_BASE))
PROFILE = Path(os.environ.get("LORACAST_PROFILE_DIR", f"{REMOTE_BASE}/profile_slack_component_v1"))
BRANCH = Path(os.environ.get("LORACAST_REPEATED_BRANCH_DIR", f"{REMOTE_BASE}/repeated_contention_r50_100_20261026"))
MANIFEST = HERE / "experiment_manifest.json"
LOCK = HERE / "artifact_lock.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_file(path: Path, expected: str) -> None:
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f"source missing or SHA-256 changed: {path}")


def load_profile_rule() -> object:
    path = PROFILE / "profile_rule.py"
    spec = importlib.util.spec_from_file_location("frozen_pyramid_profile_rule", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen profile rule: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate(strict_inputs: bool) -> dict:
    manifest = json.loads(MANIFEST.read_text())
    lock = json.loads(LOCK.read_text())
    for filename, expected in lock.items():
        check_file(HERE / filename, expected)
    source_files = {
        "nano_qwen_rtx_run_adaptive_training.py": BASE / "nano_qwen_rtx_run_adaptive_training.py",
        "edge_profile_oracle.py": BASE / "edge_profile_oracle.py",
        "nano_experiment_manifest.json": BASE / "nano_experiment_manifest.json",
        "run_qwen_training.py": BASE.parent / "quality_campaign/run_qwen_training.py",
        "profile_rule.py": PROFILE / "profile_rule.py",
        "frozen_completion_profile.json": PROFILE / "frozen_completion_profile.json",
        "repeated_contention_manifest.json": BRANCH / "experiment_manifest.json",
    }
    for name, path in source_files.items():
        check_file(path, manifest["source_hashes"][name])

    rows = manifest["factor_rounds"]
    if len(rows) != 100 or [int(row["round"]) for row in rows] != list(range(1, 101)):
        raise ValueError("matched 100-round factor schedule is incomplete")
    if any(row["phase"] != "normal" or float(row["nano_compute_seconds_per_token_factor"]) != 1.0
           for row in rows[:49]):
        raise ValueError("R1–R49 must be normal")
    source_branch = json.loads((BRANCH / "experiment_manifest.json").read_text())
    if rows[49:] != source_branch["factor_rounds"]:
        raise ValueError("R50–R100 schedule differs from completed LoRACast branch")

    profile = json.loads((PROFILE / "frozen_completion_profile.json").read_text())
    decisions = load_profile_rule().choose_steps(profile, {"nano": 28143, "nx": 18441})
    if decisions != manifest["profile_slack_component"]["frozen_decision"]:
        raise ValueError("frozen PyramidFL profile decision changed")
    runs = {r["policy"]: r for r in manifest["runs"]}
    if set(runs) != {"Baseline", "PyramidFL-profile-slack"}:
        raise ValueError("unexpected policy set")
    if runs["Baseline"]["requested_steps"] != {"nano": 320, "nx": 320}:
        raise ValueError("Baseline quota changed")
    if runs["PyramidFL-profile-slack"]["requested_steps"] != {"nano": 320, "nx": 342}:
        raise ValueError("PyramidFL profile-slack quota changed")
    if runs["Baseline"]["source_workload"] != {"nano": [320, 1, 0], "nx": [320, 1, 0]}:
        raise ValueError("Baseline source quota tuple changed")
    if runs["PyramidFL-profile-slack"]["source_workload"] != {"nano": [320, 1, 320], "nx": [342, 1, 342]}:
        raise ValueError("PyramidFL profile-slack source quota tuple changed")
    original_manifest = json.loads((BASE / "nano_experiment_manifest.json").read_text())
    common = manifest["run_common"]
    for key in ("seed", "model", "optimizer", "lora", "batch_size", "local_epochs", "sequence_cap",
                "strong_partition_mapping", "source_sampler", "holdout", "evaluation_rounds",
                "evaluation_metric", "aggregation"):
        if common[key] != original_manifest[key]:
            raise ValueError(f"existing Fig. 8 protocol mismatch: {key}")
    if strict_inputs:
        source = Path(os.environ.get("LORACAST_STRONG_INPUTS", "/home/wwjang/workload_policy/inputs"))
        partitions = Path(os.environ.get("LORACAST_FIXED_PARTITIONS", "/home/wwjang/loracast_long_horizon_100r/fixed_partitions.json"))
        expected = manifest["expected_inputs"]
        check_file(source / "initial_adapter_round0.pt", expected["initial_adapter_round0.pt"])
        check_file(source / "loracast_strong.json", expected["loracast_strong.json"])
        check_file(partitions, expected["fixed_partitions.json"])
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", choices=("nanonx_s42_dfull_qwen_baseline_repeat4x_r50_100_v1",
                                             "nanonx_s42_dfull_qwen_pyramid_profile_slack_repeat4x_r50_100_v1"))
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--check-production-prelaunch", action="store_true",
                        help="run the original run_adaptive_training's CPU-only prelaunch_check without training")
    parser.add_argument("--strict-inputs", action="store_true")
    parser.add_argument("--stop-after-round", type=int, choices=(1,))
    args = parser.parse_args()
    manifest = validate(args.strict_inputs)
    if args.validate_only:
        print(json.dumps({"status": "CPU_PREFLIGHT_PASS", "manifest_sha256": sha256(MANIFEST),
                          "strict_inputs": args.strict_inputs,
                          "policies": [r["policy"] for r in manifest["runs"]],
                          "rounds": len(manifest["factor_rounds"])}, indent=2))
        return
    if not args.run_id:
        parser.error("--run-id is required for training")
    if not args.strict_inputs:
        parser.error("--strict-inputs is required for training")
    if not args.check_production_prelaunch and os.environ.get("LORACAST_MATCHED_STATIC_RTX_APPROVED") != "1":
        raise RuntimeError("RTX launch guard is closed")
    os.environ.setdefault("LORACAST_NANO_RTX_APPROVED", "1")
    os.environ.setdefault("LORACAST_RTX_CORE_DIR", str(BASE.parent / "quality_campaign"))
    sys.path.insert(0, str(BASE))
    import nano_qwen_rtx_runner as original

    original.MANIFEST = MANIFEST
    original.OUT_ROOT = HERE / "rtx_runs"
    by_id = {r["run_id"]: r for r in manifest["runs"]}
    selected = by_id[args.run_id]

    def quota(policy: str, rnd: int) -> dict:
        if policy != selected["policy"] or not 1 <= rnd <= 100:
            raise ValueError("unexpected policy or round")
        workloads = selected["source_workload"]
        return {"nano": tuple(workloads["nano"]),
                "nx": tuple(workloads["nx"]),
                "origin": "frozen fixed-workload decision; matched contention trace affects post hoc cost only"}

    original.frozen_quota = quota
    if args.check_production_prelaunch:
        checked_manifest, checked_run = original.prelaunch_check(args.run_id)
        if checked_manifest["experiment_id"] != manifest["experiment_id"] or checked_run["policy"] != selected["policy"]:
            raise ValueError("production prelaunch selected the wrong experiment")
        print(json.dumps({"status": "PRODUCTION_PRELAUNCH_PASS", "run_id": args.run_id,
                          "quota_R2": quota(checked_run["policy"], 2),
                          "manifest_sha256": sha256(MANIFEST)}, indent=2))
        return
    original.execute(args.run_id, args.stop_after_round)


if __name__ == "__main__":
    main()
