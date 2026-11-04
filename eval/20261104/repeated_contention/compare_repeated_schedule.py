#!/usr/bin/env python3
"""Compare verified 100-round Nano/NX trajectories under one repeated cost trace.

PyramidFL profile-slack has a fixed workload decision, so the repeated trace
changes its projected local-compute cost, not its RTX training trajectory.
LoRACast's R50--R100 records must come from the separate adaptive branch.
No contended energy is inferred here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

NX_UNIQUE_SAMPLES = 18441


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def cycle_schedule(round_number: int, source: dict) -> tuple[str, float, int | None]:
    if round_number < 50:
        return "normal", 1.0, None
    phase_index = (round_number - 50) % 12
    if phase_index < 4:
        return "normal", 1.0, None
    reference_round = 5 + phase_index - 4
    rows = {int(row["fig9_reference_round"]): row for row in source["schedule"]}
    if reference_round not in rows:
        raise ValueError(f"missing frozen Fig. 9 reference R{reference_round}")
    phase = "contention" if phase_index < 8 else "recovery"
    row = rows[reference_round]
    if row["phase"] != phase:
        raise ValueError("source phase differs from repeated schedule")
    return phase, float(row["nano_compute_seconds_per_token_factor"]), reference_round


def validate_run(run: Path, name: str) -> tuple[list[dict], list[dict]]:
    rounds = jsonl(run / "rounds.jsonl")
    evaluations = jsonl(run / "validation.jsonl")
    if [row["round"] for row in rounds] != list(range(1, 101)):
        raise ValueError(f"{name}: missing or repeated training rounds")
    if [row["round"] for row in evaluations] != list(range(0, 101, 10)):
        raise ValueError(f"{name}: missing or repeated validation checkpoints")
    if len({row["loss_mode"] for row in evaluations}) != 1:
        raise ValueError(f"{name}: mixed validation scorers")
    evaluation_identity = ("loss_mode", "cap_per_view", "scored_examples", "seed",
                           "nano_partition_origin")
    if len({tuple(row.get(key) for key in evaluation_identity)
            for row in evaluations}) != 1:
        raise ValueError(f"{name}: validation protocol changes across checkpoints")
    if any(not math.isfinite(float(row["balanced_client_loss"]))
           for row in evaluations):
        raise ValueError(f"{name}: invalid balanced validation NLL")
    previous = evaluations[0]["checkpoint_sha256"]
    if not previous:
        raise ValueError(f"{name}: missing R0 hash")
    for row in rounds:
        r = row["round"]
        if row["global_start_checkpoint_sha256"] != previous:
            raise ValueError(f"{name}: checkpoint chain broken at R{r}")
        previous = row["checkpoint_sha256"]
        clients = row["clients"]
        samples = sum(int(c["samples"]) for c in clients.values())
        if samples <= 0 or abs(sum(row["aggregation_weights"].values()) - 1.0) > 1e-9:
            raise ValueError(f"{name}: invalid aggregation at R{r}")
        for client, metric in clients.items():
            if int(metric["samples"]) != len(metric["visited_sample_ids"]):
                raise ValueError(f"{name}: sample visits disagree at R{r}/{client}")
            if int(metric["samples"]) < 0 or int(metric["input_tokens"]) < 0:
                raise ValueError(f"{name}: negative executed work at R{r}/{client}")
            requested = row["requested_workload"][client]
            requested_epochs = requested[1] if isinstance(requested, list) else requested["E"]
            if int(requested_epochs) not in (0, 1):
                raise ValueError(f"{name}: local epoch policy differs at R{r}/{client}")
            if ((int(metric["samples"]) > 0 and int(requested_epochs) != 1) or
                    (int(metric["samples"]) == 0 and int(requested_epochs) != 0) or
                    int(metric["optimizer_steps"]) != int(metric["samples"])):
                raise ValueError(f"{name}: E=1 executed-work accounting differs at R{r}/{client}")
            if abs(row["aggregation_weights"][client] - metric["samples"] / samples) > 1e-9:
                raise ValueError(f"{name}: aggregation weight differs at R{r}/{client}")
        for device in ("nano", "nx"):
            if row["edge_compute_projection"][device]["support"] == "unsupported":
                raise ValueError(f"{name}: unsupported cost projection at R{r}/{device}")
            compute_time = float(row["edge_compute_projection"][device]["compute_s"])
            if not math.isfinite(compute_time) or compute_time < 0:
                raise ValueError(f"{name}: invalid projected compute time at R{r}/{device}")
    by_round = {row["round"]: row for row in rounds}
    for row in evaluations[1:]:
        if row["checkpoint_sha256"] != by_round[row["round"]]["checkpoint_sha256"]:
            raise ValueError(f"{name}: evaluated checkpoint differs at R{row['round']}")
    return rounds, evaluations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pyramid-run", required=True, type=Path)
    parser.add_argument("--loracast-run", required=True, type=Path)
    parser.add_argument("--frozen-source", required=True, type=Path)
    parser.add_argument("--branch-manifest", required=True, type=Path)
    parser.add_argument("--loracast-parent-rounds", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = json.loads(args.frozen_source.read_text())
    expected = list(range(5, 13))
    if [int(row["fig9_reference_round"]) for row in source["schedule"]] != expected:
        raise ValueError("Fig. 9 source does not contain R5--R12 in order")
    pyramid, pyramid_val = validate_run(args.pyramid_run, "PyramidFL profile-slack")
    loracast, loracast_val = validate_run(args.loracast_run, "LoRACast")
    branch_manifest = json.loads(args.branch_manifest.read_text())
    if (branch_manifest["source_round"] != 49 or
            branch_manifest["first_branch_round"] != 50 or
            branch_manifest["last_round"] != 100):
        raise ValueError("LoRACast branch horizon differs from comparison")
    parent_rounds = jsonl(args.loracast_parent_rounds)
    if loracast[:49] != parent_rounds[:49]:
        raise ValueError("LoRACast R1--R49 prefix differs from verified parent")
    if [row["round"] for row in branch_manifest["factor_rounds"]] != list(range(50, 101)):
        raise ValueError("branch manifest factor schedule is incomplete")
    for item in branch_manifest["factor_rounds"]:
        phase, factor, reference = cycle_schedule(item["round"], source)
        if (item["phase"] != phase or
                abs(float(item["nano_compute_seconds_per_token_factor"]) - factor) > 1e-12 or
                item["fig9_reference_round"] != reference):
            raise ValueError(f"branch manifest differs from frozen trace at R{item['round']}")
    if any((row.get("phase") not in (None, "normal") or
            float(row["edge_compute_projection"]["nano"].get("nano_time_factor", 1.0)) != 1.0)
           for row in loracast[:49]):
        raise ValueError("LoRACast R1--R49 prefix is not a normal-profile prefix")
    if pyramid_val[0]["checkpoint_sha256"] != loracast_val[0]["checkpoint_sha256"]:
        raise ValueError("policies do not share the same R0 adapter")
    for key in ("loss_mode", "cap_per_view", "scored_examples", "seed",
                "nano_partition_origin"):
        if pyramid_val[0].get(key) != loracast_val[0].get(key):
            raise ValueError(f"policies use different validation protocol: {key}")
    if any((row["clients"]["nano"]["optimizer_steps"],
            row["clients"]["nx"]["optimizer_steps"]) != (320, 342)
           for row in pyramid):
        raise ValueError("profile-slack workload is not fixed 320/342")

    records: list[dict] = []
    summary: dict = {
        "source": {
            "frozen_trace": str(args.frozen_source),
            "frozen_trace_sha256": sha256(args.frozen_source),
            "branch_manifest": str(args.branch_manifest),
            "branch_manifest_sha256": sha256(args.branch_manifest),
            "loracast_parent_rounds": str(args.loracast_parent_rounds),
            "loracast_parent_rounds_sha256": sha256(args.loracast_parent_rounds),
            "pyramid_rounds": str(args.pyramid_run / "rounds.jsonl"),
            "pyramid_rounds_sha256": sha256(args.pyramid_run / "rounds.jsonl"),
            "loracast_rounds": str(args.loracast_run / "rounds.jsonl"),
            "loracast_rounds_sha256": sha256(args.loracast_run / "rounds.jsonl"),
        },
        "measurement_boundary": "RTX actual NLL; trace-projected edge local-compute time; no contended energy",
        "policies": {},
    }
    for name, rows, evaluations in (("PyramidFL profile-slack", pyramid, pyramid_val),
                                    ("LoRACast", loracast, loracast_val)):
        cumulative_tokens = 0
        cumulative_time = 0.0
        phase_totals: dict[str, dict[str, float]] = {}
        for row in rows:
            r = int(row["round"])
            phase, factor, reference = cycle_schedule(r, source)
            if not math.isfinite(factor) or factor <= 0:
                raise ValueError(f"invalid measured Nano time factor at R{r}")
            projection = row["edge_compute_projection"]
            nano_factor_in_record = float(projection["nano"].get("nano_time_factor", 1.0))
            if name == "LoRACast":
                if r >= 50 and (row.get("phase") != phase or
                                abs(nano_factor_in_record - factor) > 1e-9):
                    raise ValueError(f"adaptive branch factor/phase differs at R{r}")
                nano_time = float(projection["nano"]["compute_s"])
            else:
                nano_time = float(projection["nano"]["compute_s"]) * factor
            nx_time = float(projection["nx"]["compute_s"])
            window = max(nano_time, nx_time)
            if not math.isfinite(window) or window <= 0:
                raise ValueError(f"invalid projected compute window at R{r}")
            tokens = sum(int(c["input_tokens"]) for c in row["clients"].values())
            cumulative_tokens += tokens
            cumulative_time += window
            expected_exposure = sum(int(item["clients"]["nx"]["samples"])
                                    for item in rows[:r]) / NX_UNIQUE_SAMPLES
            if abs(float(row["nx_cumulative_exposure"]) - expected_exposure) > 1e-10:
                raise ValueError(f"{name}: cumulative NX exposure differs at R{r}")
            phase_totals.setdefault(phase, {"tokens": 0, "time_s": 0.0, "rounds": 0})
            phase_totals[phase]["tokens"] += tokens
            phase_totals[phase]["time_s"] += window
            phase_totals[phase]["rounds"] += 1
            records.append({
                "policy": name, "round": r, "phase": phase,
                "fig9_reference_round": reference or "", "nano_time_factor": factor,
                "nano_executed_steps": row["clients"]["nano"]["optimizer_steps"],
                "nx_executed_steps": row["clients"]["nx"]["optimizer_steps"],
                "nano_executed_samples": row["clients"]["nano"]["samples"],
                "nx_executed_samples": row["clients"]["nx"]["samples"],
                "input_tokens": tokens, "nano_compute_s": nano_time,
                "nx_compute_s": nx_time, "round_compute_window_s": window,
                "cumulative_tokens": cumulative_tokens,
                "cumulative_compute_window_s": cumulative_time,
                "cumulative_tokens_per_hour": cumulative_tokens / cumulative_time * 3600,
                "nx_cumulative_exposure": row["nx_cumulative_exposure"],
            })
        if name == "LoRACast" and max(row["nx_cumulative_exposure"] for row in rows) > 2 + 1e-12:
            raise ValueError("LoRACast exposure limit exceeded")
        for phase, totals in phase_totals.items():
            totals["tokens_per_hour"] = totals["tokens"] / totals["time_s"] * 3600
        summary["policies"][name] = {
            "r100_balanced_validation_nll": evaluations[-1]["balanced_client_loss"],
            "r100_global_validation_nll": evaluations[-1]["global_loss"],
            "loss_mode": evaluations[-1]["loss_mode"],
            "total_input_tokens": cumulative_tokens,
            "total_projected_local_compute_window_s": cumulative_time,
            "total_projected_tokens_per_hour": cumulative_tokens / cumulative_time * 3600,
            "nx_executed_samples": sum(row["clients"]["nx"]["samples"] for row in rows),
            "nx_final_exposure": rows[-1]["nx_cumulative_exposure"],
            "nx_omitted_rounds": sum(row["clients"]["nx"]["samples"] == 0 for row in rows),
            "phase_totals": phase_totals,
        }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "repeated_contention_round_values.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (args.output / "repeated_contention_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
