#!/usr/bin/env python3
"""Freeze a Nano/NX completion profile from prior normal hardware rounds.

The profile is independent of the seed-42 RTX learning trajectory.  The
non-compute portion uses the dispatch-to-first-reply boundary minus measured
local compute, which includes Flower's adapter preparation and exchange.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
SOURCE = Path(
    "/Users/jang-won-u/Documents/Codex/2026-10-10/new-chat/outputs/"
    "learning_efficiency_confirmation/per_round_hardware_results.csv"
)
OUTPUT = HERE / "frozen_completion_profile.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict:
    rows = list(csv.DictReader(SOURCE.open(newline="")))
    selected = [
        row for row in rows
        if row["policy"] == "U"
        and row["phase"] == "normal"
        and int(row["seed"]) in (43, 44, 45)
        and int(row["round"]) in (2, 3, 4)
    ]
    keys = {(int(row["seed"]), int(row["round"])) for row in selected}
    if keys != {(seed, rnd) for seed in (43, 44, 45) for rnd in (2, 3, 4)} or len(selected) != 9:
        raise ValueError("normal Baseline calibration requires exactly nine matched rows")
    devices = {}
    for name in ("nano", "nx"):
        samples = []
        for row in selected:
            if int(row[f"{name}_steps"]) != 320:
                raise ValueError("calibration workload changed")
            comp = float(row[f"{name}_compute_s"])
            first_reply = float(row[f"{name}_first_reply_s"])
            if not 0 < comp < first_reply:
                raise ValueError("invalid completion boundary")
            samples.append((comp, first_reply - comp))
        devices[name] = {
            "base_compute_s": mean(item[0] for item in samples),
            "base_noncompute_completion_s": mean(item[1] for item in samples),
            "observation_count": len(samples),
        }
    return {
        "schema_version": 1,
        "base_optimizer_steps": 320,
        "beta": 0.5,
        "maximum_optimizer_steps": 600,
        "batch_size": 1,
        "local_epochs": 1,
        "calibration": "Nano+NX Qwen, Baseline U, independent seeds 43-45, normal R2-R4",
        "compute_boundary": "measured local compute seconds",
        "noncompute_boundary": "dispatch-to-first-reply seconds minus measured local compute; includes Flower adapter preparation and exchange",
        "source_path": str(SOURCE),
        "source_sha256": sha256(SOURCE),
        "devices": devices,
    }


if __name__ == "__main__":
    payload = build()
    if OUTPUT.exists() and json.loads(OUTPUT.read_text()) != payload:
        raise ValueError("frozen profile already exists with different contents")
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"profile": str(OUTPUT), "sha256": sha256(OUTPUT), "devices": payload["devices"]}, indent=2))
