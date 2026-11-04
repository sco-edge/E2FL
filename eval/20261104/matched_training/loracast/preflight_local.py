#!/usr/bin/env python3
"""Read-only structural preflight for the isolated matched LoRACast run.

The GPU host's run_adaptive_training.py --validate-only performs the full input/model/source
hash check. This local check intentionally does not launch training.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "experiment_manifest.json"
SOURCE = HERE / "source_repeated_contention_manifest_snapshot.json"
RUNNER = HERE / "run_adaptive_training.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    source = json.loads(SOURCE.read_text())
    schedule = manifest["stress_schedule"]
    cfg = manifest["controller"]["native_train_config"]["LoRACast-Exposure"]
    assert not manifest["adaptive_launch_allowed"], "GPU launch must remain closed during local setup"
    assert manifest["run_common"]["rounds"] == 100
    assert manifest["run_common"]["seed"] == 42
    assert manifest["run_common"]["model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert manifest["run_common"]["sequence_cap"] == 64
    assert int(cfg["jetson-base-quota"]) == 256
    assert int(cfg["jetson-max-quota"]) == 600
    assert int(cfg["local-epochs"]) == 1
    assert int(cfg["policy-warmup-rounds"]) == 4
    assert float(manifest["controller"]["exposure_limit"]) == 2.0
    assert schedule["normal_rounds"] == [1, 49]
    assert sha256(SOURCE) == schedule["source_sha256"]
    rows = schedule["factor_rounds"]
    assert [row["extension_round"] for row in rows] == list(range(50, 101))
    for row, original in zip(rows, source["factor_rounds"], strict=True):
        assert row["extension_round"] == original["round"]
        assert {k: v for k, v in row.items() if k != "extension_round"} == {
            k: v for k, v in original.items() if k != "round"}
        offset = (row["extension_round"] - 50) % 12
        expected_phase = "normal" if offset < 4 else "contention" if offset < 8 else "recovery"
        assert row["phase"] == expected_phase
        assert float(row["nano_compute_seconds_per_token_factor"]) > 0
        assert (row["fig9_reference_round"] ==
                (None if offset < 4 else 5 + offset - 4 if offset < 8 else 9 + offset - 8))
    text = RUNNER.read_text()
    for required in ("SOURCE / 'initial_adapter_round0.pt'", "current = 0",
                     "strategy.configure_train(", "strategy.aggregate_train(",
                     "resume.load_completed_round(", "core.evaluate(",
                     "pretraining_decision_gate(", "LORACAST_MATCHED_RTX_APPROVED"):
        assert required in text, f"missing run_adaptive_training operation/gate: {required}"
    assert "bootstrap_branch(" not in text
    assert not (HERE / "runs").exists(), "new-run output already exists; audit before any launch"
    print(json.dumps({
        "status": "STRUCTURAL_PASS_REMOTE_SOURCE_PREFLIGHT_PENDING",
        "gpu_work_performed": False,
        "run_id": manifest["runs"][0]["run_id"],
        "round0_source": "initial_adapter_round0.pt",
        "previous_r49_checkpoint_reused": False,
        "schedule": "R1-49 normal; R50-100 repeated 4/4/4",
        "manifest_sha256": sha256(MANIFEST),
        "runner_sha256": sha256(RUNNER),
        "source_repeated_schedule_sha256": sha256(SOURCE),
    }, indent=2))


if __name__ == "__main__":
    main()
