#!/usr/bin/env python3
"""Read-only, byte-level audit of the three *new* matched RTX trajectories.

Run this on the GPU host, where all 303 adapter files and LoRACast's 100 native
Flower snapshots are present.  The only write is the requested isolated report.
No model training, replay, or evaluation is performed by this script.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile


POLICIES = ("Baseline", "PyramidFL-style", "LoRACast")
POINTS = list(range(0, 101, 10))
HEX64 = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_SOURCE = {
    "initial_adapter_round0.pt": "b3ff342f8a6a0b9c42263a311fd6a845ccad2cbba106e84c485779b9cfd0511d",
    "loracast_strong.json": "ee13a0fb12e4f917a01b6ff6fe5fc1f1b867fed3599e1c1dcd79ae36451b6493",
    "fixed_partitions.json": "8b700bd14f40bfce951971fe929a641f49c0730bf131b827494480f49d559371",
}
EXPECTED_R0 = "7ae6c591bc59c7c31f781e4b5c689b19ef5703b23aaa732a9e0b9d8cb4f8975b"
EXPECTED_SCHEDULE = "c8a9b67c33ce900f58932d3a48267b3568499a7083c9b92780a8b5c4043b63de"
EXPECTED_ADAPTIVE_RUNNER = "a7a8146eea1673d7402b6d2836ec1da09301b02995ca2da49c589b75aae4301c"


def require(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError(why)


def sha(path: Path) -> str:
    require(path.is_file(), f"missing file: {path}")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_object(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def hex_sha(value: object, label: str) -> str:
    require(isinstance(value, str) and HEX64.fullmatch(value) is not None,
            f"{label}: invalid SHA-256")
    return value


def read_json(path: Path) -> dict:
    require(path.is_file(), f"missing JSON: {path}")
    obj = json.loads(path.read_text())
    require(isinstance(obj, dict), f"expected JSON object: {path}")
    return obj


def read_jsonl(path: Path) -> list[dict]:
    require(path.is_file(), f"missing JSONL: {path}")
    rows = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        require(isinstance(obj, dict), f"{path}:{lineno}: expected object")
        rows.append(obj)
    return rows


def finite_number(value: object, label: str, minimum: float | None = None) -> float:
    require(type(value) in (int, float) and math.isfinite(value),
            f"{label}: missing/non-finite number")
    v = float(value)
    require(minimum is None or v >= minimum, f"{label}: below {minimum}")
    return v


def same(a: object, b: object, label: str, tolerance: float = 1e-10) -> None:
    require(math.isclose(finite_number(a, label), finite_number(b, label),
                         rel_tol=0.0, abs_tol=tolerance), f"{label}: mismatch")


def audit_sources(args: argparse.Namespace) -> tuple[dict, dict, dict, dict]:
    static_dir, loracast_dir = args.static_dir, args.loracast_dir
    lock = read_json(static_dir / "artifact_lock.json")
    for name, expected in lock.items():
        require(sha(static_dir / name) == hex_sha(expected, name),
                f"static artifact lock failed: {name}")
    static = read_json(static_dir / "experiment_manifest.json")
    adaptive = read_json(loracast_dir / "experiment_manifest.json")
    require(sha(loracast_dir / "run_adaptive_training.py") == EXPECTED_ADAPTIVE_RUNNER,
            "isolated LoRACast training run_adaptive_training changed")
    native_hash_path = loracast_dir / "nano_adaptive_native_source_hashes_v1.json"
    require(sha(native_hash_path) == adaptive["adaptive_native_source_hashes_sha256"],
            "pinned native Flower source inventory changed")
    native_hashes = read_json(native_hash_path)
    native_root = Path(adaptive["controller"]["original_source_root"])
    for relative, expected in native_hashes["files"].items():
        require(sha(native_root / relative) == hex_sha(expected, relative),
                f"native Flower source changed: {relative}")
    profile_dir = args.edge_protocol / "profile_slack_component_v1"
    repeated_dir = args.edge_protocol / "repeated_contention_r50_100_20261026"
    source_paths = {
        "nano_qwen_rtx_run_adaptive_training.py": args.edge_protocol / "nano_qwen_rtx_run_adaptive_training.py",
        "edge_profile_oracle.py": args.edge_protocol / "edge_profile_oracle.py",
        "nano_experiment_manifest.json": args.edge_protocol / "nano_experiment_manifest.json",
        "run_qwen_training.py": args.edge_protocol.parent / "quality_campaign" / "run_qwen_training.py",
        "profile_rule.py": profile_dir / "profile_rule.py",
        "frozen_completion_profile.json": profile_dir / "frozen_completion_profile.json",
        "repeated_contention_manifest.json": repeated_dir / "experiment_manifest.json",
    }
    for key, path in source_paths.items():
        require(sha(path) == static["source_hashes"][key],
                f"static controller/training source changed: {key}")
    require(static["static_launch_allowed"] is True and
            adaptive["adaptive_launch_allowed"] is True,
            "training launch was not frozen/approved in both manifests")
    require(static["expected_inputs"]["prior_verified_serialized_R0_adapter"] == EXPECTED_R0,
            "static R0 provenance differs")
    require(static["expected_inputs"] == {
        **EXPECTED_SOURCE, "prior_verified_serialized_R0_adapter": EXPECTED_R0},
        "static source input pins differ")
    for name, expected in EXPECTED_SOURCE.items():
        path = (args.fixed_partitions if name == "fixed_partitions.json" else
                args.strong_inputs / name)
        require(sha(path) == expected, f"source input changed: {name}")
    require(adaptive["static_manifest_sha256"] ==
            sha(args.edge_protocol / "nano_experiment_manifest.json"),
            "LoRACast original static protocol changed")
    require(static["source_hashes"]["run_qwen_training.py"] ==
            adaptive["rtx_core_runner_sha256"],
            "policies do not pin the same RTX training/scoring core")
    require(static["run_common"] == {key: adaptive["run_common"][key]
                                     for key in static["run_common"]},
            "common training protocol differs between manifests")
    require(static["run_common"]["seed"] == 42 and
            static["run_common"]["model"] == "Qwen/Qwen2.5-0.5B-Instruct" and
            static["run_common"]["rounds"] == 100 and
            static["run_common"]["sequence_cap"] == 64 and
            static["run_common"]["local_epochs"] == 1 and
            static["run_common"]["evaluation_rounds"] == POINTS and
            static["run_common"]["evaluation_metric"] ==
                "full_sequence_64_token token-weighted NLL" and
            static["run_common"]["aggregation"] ==
                "actual processed-sample-weighted LoRA A/B factor parameters",
            "wrong model/seed/horizon/validation protocol")
    require(sha(loracast_dir / "source_repeated_contention_manifest_snapshot.json")
            == EXPECTED_SCHEDULE, "measured stress source changed")
    source_schedule = read_json(loracast_dir /
                                "source_repeated_contention_manifest_snapshot.json")
    source_rows = source_schedule["factor_rounds"]
    require([r["round"] for r in source_rows] == list(range(50, 101)),
            "stress schedule is not R50–R100")
    require(static["factor_rounds"][49:] == source_rows,
            "static factor schedule differs from measured source")
    require([r["round"] for r in static["factor_rounds"]] == list(range(1, 101)) and
            all(r["phase"] == "normal" and
                r["nano_compute_seconds_per_token_factor"] == 1.0
                for r in static["factor_rounds"][:49]),
            "static R1–R49 normal prefix differs")
    adaptive_rows = [dict(r, round=r["extension_round"]) for r in
                     adaptive["stress_schedule"]["factor_rounds"]]
    for r in adaptive_rows:
        del r["extension_round"]
    require(adaptive_rows == source_rows and
            adaptive["stress_schedule"]["source_sha256"] == EXPECTED_SCHEDULE,
            "LoRACast schedule differs from shared measured source")
    for row in static["factor_rounds"]:
        n = row["round"]
        i = (n - 50) % 12 if n >= 50 else 0
        phase = "normal" if n < 50 or i < 4 else ("contention" if i < 8 else "recovery")
        ref = None if phase == "normal" else (5 + i - 4 if phase == "contention"
                                               else 9 + i - 8)
        require(row["phase"] == phase and row["fig9_reference_round"] == ref and
                finite_number(row["nano_compute_seconds_per_token_factor"],
                              f"R{n} Nano factor", 0.0) > 0,
                f"R{n}: invalid matched 4/4/4 schedule")
    return static, adaptive, {r["round"]: r for r in static["factor_rounds"]}, lock


def workload_tuple(row: dict, name: str, policy: str) -> tuple[int, int, int]:
    w = row["requested_workload"][name]
    if policy == "LoRACast":
        return int(w["S"]), int(w["E"]), int(w["H"])
    require(isinstance(w, list) and len(w) == 3, f"{policy}: invalid workload tuple")
    return tuple(int(v) for v in w)


def audit_run(policy: str, path: Path, run_id: str, manifest_sha: str,
              schedule: dict[int, dict], population: dict[str, set[int]],
              adaptive: dict, expected_quota: dict[str, list[int]] | None) -> dict:
    require(path.name == run_id, f"{policy}: unexpected output run ID/path")
    rounds, evals, state = (read_jsonl(path / "rounds.jsonl"),
                           read_jsonl(path / "validation.jsonl"),
                           read_json(path / "state.json"))
    require(len(rounds) == 100 and [r.get("round") for r in rounds] ==
            list(range(1, 101)), f"{policy}: missing/duplicate committed rounds")
    require(len(evals) == 11 and [r.get("round") for r in evals] == POINTS,
            f"{policy}: missing/duplicate actual checkpoint evaluations")
    require(state.get("run_id") == run_id and state.get("round") == 100 and
            state.get("protocol_sha256") == manifest_sha,
            f"{policy}: final state/protocol mismatch")
    require(all(r.get("run_id") == run_id for r in rounds + evals),
            f"{policy}: mixed trajectories")
    require(sha(path / "adapter_0.pt") == EXPECTED_R0 ==
            evals[0]["checkpoint_sha256"], f"{policy}: common R0 differs")
    require({p.name for p in path.glob("adapter_*.pt")} ==
            {f"adapter_{r}.pt" for r in range(101)},
            f"{policy}: checkpoint file inventory is not exactly R0–R100")
    actual_checkpoints: dict[str, str] = {"0": EXPECTED_R0}
    counters: dict[str, Counter[int]] = {"nano": Counter(), "nx": Counter()}
    totals = {"nano": {"samples": 0, "input_tokens": 0},
              "nx": {"samples": 0, "input_tokens": 0}}
    previous = EXPECTED_R0
    native_hashes: dict[str, str] = {}
    cumulative_nx = 0
    for number, row in enumerate(rounds, 1):
        label = f"{policy} R{number}"
        require(row.get("global_start_checkpoint_sha256") == previous,
                f"{label}: global start breaks checkpoint chain")
        actual = sha(path / f"adapter_{number}.pt")
        require(actual == hex_sha(row.get("checkpoint_sha256"), label),
                f"{label}: adapter bytes differ from log")
        actual_checkpoints[str(number)] = actual
        previous = actual
        require(set(row.get("clients", {})) == {"nano", "nx"},
                f"{label}: client records missing")
        weights = row.get("aggregation_weights", {})
        require(set(weights) == {"nano", "nx"}, f"{label}: weights missing")
        total = sum(int(row["clients"][name]["samples"]) for name in weights)
        require(total > 0, f"{label}: no aggregate inputs")
        same(sum(finite_number(v, f"{label} weight") for v in weights.values()),
             1.0, f"{label}: aggregate sum", 1e-9)
        for name in ("nano", "nx"):
            metric = row["clients"][name]
            s, e, h = workload_tuple(row, name, policy)
            if policy != "LoRACast":
                require([s, e, h] == expected_quota[name],
                        f"{label}/{name}: static workload changed")
            else:
                require((s, e, h) == (0, 0, 0) or
                        (1 <= s <= 600 and e == 1 and 0 <= h <= s),
                        f"{label}/{name}: native workload out of bounds")
            samples = int(metric["samples"])
            steps = int(metric["optimizer_steps"])
            intended = min(s * e, h if h > 0 else s * e)
            require(samples == steps == intended,
                    f"{label}/{name}: executed S/H/E differs from request")
            selected, visited = metric["selected_sample_ids"], metric["visited_sample_ids"]
            require(isinstance(selected, list) and isinstance(visited, list) and
                    len(selected) == s and len(visited) == samples and
                    all(type(v) is int for v in selected + visited) and
                    not (Counter(visited) - Counter(selected)) and
                    set(selected) <= population[name],
                    f"{label}/{name}: sample IDs differ from the frozen partition")
            counters[name].update(visited)
            same(weights[name], samples / total, f"{label}/{name}: aggregation weight", 1e-9)
            require(metric.get("local_training_start_checkpoint_sha256") ==
                    (row["global_start_checkpoint_sha256"] if samples else None),
                    f"{label}/{name}: local model started from another checkpoint")
            for token_field in ("input_tokens", "supervised_tokens"):
                amount = metric[token_field]
                require(type(amount) is int and amount >= 0 and
                        (amount > 0 or samples == 0),
                        f"{label}/{name}: invalid {token_field}")
            totals[name]["samples"] += samples
            totals[name]["input_tokens"] += metric["input_tokens"]
        cumulative_nx += row["clients"]["nx"]["samples"]
        same(row["nx_cumulative_exposure"], cumulative_nx / 18441,
             f"{label}: NX cumulative exposure", 1e-12)
        require(not (set(row["clients"]["nano"]["visited_sample_ids"]) &
                     set(row["clients"]["nx"]["visited_sample_ids"])),
                f"{label}: cross-client sample overlap")
        if policy == "LoRACast":
            require(row.get("policy") == "LoRACast-Exposure",
                    f"{label}: wrong native controller policy marker")
            require(cumulative_nx <= 36882, f"{label}: Exposure-(2) cap exceeded")
            frozen = schedule[number]
            require(row.get("phase") == frozen["phase"] and
                    row.get("fig9_reference_round") == frozen["fig9_reference_round"],
                    f"{label}: logged phase/reference differs from frozen schedule")
            same(row.get("nano_compute_seconds_per_token_factor"),
                 frozen["nano_compute_seconds_per_token_factor"],
                 f"{label}: Nano slowdown factor", 1e-12)
            pred = row["pre_round_prediction"]
            require(pred.get("postwarmup_window_gate") ==
                    ("WARMUP_NOT_APPLICABLE" if number <= 4 else "PASS") and
                    pred.get("native_fallback_reason") is None,
                    f"{label}: prediction gate/fallback differs")
            same(pred["nx_remaining_exposure_samples_before"],
                 36882 - (cumulative_nx - row["clients"]["nx"]["samples"]),
                 f"{label}: pre-round remaining budget")
            require(pred["nx_expected_executed_samples"] ==
                    row["clients"]["nx"]["samples"],
                    f"{label}: pre-round budget/work mismatch")
            require(set(row["source_flower_destinations"]) ==
                    {name for name in ("nano", "nx")
                     if row["clients"][name]["samples"] > 0},
                    f"{label}: Flower destination/work mismatch")
            if number > 4 and row["clients"]["nx"]["samples"]:
                require(pred["nx_final_s"] <= pred["allowed_s"] + 1e-9,
                        f"{label}: final NX work exceeds predicted window")
            projection = row["edge_compute_projection"]["nano"]
            if row["clients"]["nano"]["samples"]:
                same(projection["compute_s"],
                     projection["normal_profile_compute_s"] *
                     frozen["nano_compute_seconds_per_token_factor"],
                     f"{label}: Nano time projection", 1e-8)
            require((projection["compute_energy_j"] is None) ==
                    (frozen["phase"] != "normal"),
                    f"{label}: contention energy support mislabeled")
            native_root = path / "native_flower_log" / "resume_state"
            snapshot = native_root / f"round_{number:02d}.pkl"
            metadata = read_json(native_root / f"round_{number:02d}.json")
            native_sha = sha(snapshot)
            require(native_sha == hex_sha(row.get("native_strategy_state_sha256"), label) ==
                    metadata.get("state_sha256") and
                    metadata.get("completed_round") == number and
                    metadata.get("experiment_id") == run_id and
                    metadata.get("checkpoint_sha256") == actual and
                    metadata.get("checkpoint") == str(path / f"adapter_{number}.pt"),
                    f"{label}: native Flower state/metadata hash mismatch")
            native_hashes[str(number)] = native_sha
        else:
            require(row.get("policy") == ("Baseline" if policy == "Baseline"
                     else "PyramidFL-profile-slack"),
                    f"{label}: static policy marker differs")
    require(state["checkpoint_sha256"] == previous,
            f"{policy}: final state/checkpoint hash mismatch")
    for name, key in (("nano", "nano_visits"), ("nx", "visits")):
        saved = state.get(key)
        require(isinstance(saved, dict) and
                all(isinstance(k, str) and type(v) is int and v > 0
                    for k, v in saved.items()) and
                saved == {str(k): n for k, n in counters[name].items()},
                f"{policy}: final visit counters differ from actual client samples")
    if policy == "LoRACast":
        require(state.get("native_strategy_state_sha256") == native_hashes["100"],
                "LoRACast: final native Flower state differs")
    validations = {}
    scorer = None
    for row in evals:
        number = row["round"]
        require(row.get("checkpoint_sha256") == actual_checkpoints[str(number)] and
                row.get("checkpoint") == str(path / f"adapter_{number}.pt") and
                row.get("measurement") == "RTX_actual_model_validation",
                f"{policy}: R{number} validation/checkpoint provenance differs")
        protocol = {key: row.get(key) for key in
                    ("loss_mode", "cap_per_view", "scored_examples", "seed",
                     "nano_partition_origin")}
        require(protocol == {"loss_mode": "full_sequence_64_token",
                             "cap_per_view": 512, "scored_examples": 1888,
                             "seed": 777,
                             "nano_partition_origin": "original Strong Pi5 client 1"},
                f"{policy}: R{number} scorer/view protocol differs")
        scorer = protocol if scorer is None else scorer
        require(protocol == scorer, f"{policy}: within-run scorer drift")
        for field in ("global_loss", "balanced_client_loss",
                      "nano_associated_loss", "nx_loss", "worst_client_loss"):
            finite_number(row.get(field), f"{policy} R{number} {field}", 0.0)
        same(row["balanced_client_loss"],
             (row["nano_associated_loss"] + row["nx_loss"]) / 2,
             f"{policy} R{number}: balanced client NLL", 1e-10)
        validations[str(number)] = row["balanced_client_loss"]
    return {
        "status": "PASS", "checkpoint_chain": "PASS", "round": 100,
        "round_count": 100, "checkpoint_count": len(actual_checkpoints),
        "native_flower_state_count": len(native_hashes), "run_id": run_id,
        "run_path": str(path.resolve()),
        "rounds_sha256": sha(path / "rounds.jsonl"),
        "validation_sha256": sha(path / "validation.jsonl"),
        "state_sha256": sha(path / "state.json"),
        "protocol_sha256": manifest_sha,
        "checkpoint_sha256_by_round": actual_checkpoints,
        "native_flower_state_sha256_by_round": native_hashes,
        "source_sample_totals": totals,
        "nx_cumulative_exposure_samples": cumulative_nx,
        "r0_balanced_validation_nll": validations["0"],
        "r100_balanced_validation_nll": validations["100"],
        "scorer": scorer,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-dir", type=Path, required=True,
                        help="isolated matched_repeat4x_static_20261026 directory")
    parser.add_argument("--loracast-dir", type=Path, required=True,
                        help="isolated matched_repeat4x_loracast_20261026 directory")
    parser.add_argument("--nano-protocol", type=Path, required=True,
                        help="original Nano RTX protocol source directory (read only)")
    parser.add_argument("--strong-inputs", type=Path, required=True)
    parser.add_argument("--fixed-partitions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="new isolated JSON report; refuses to overwrite")
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite an existing audit report")
    static, adaptive, schedule, _ = audit_sources(args)
    partition = read_json(args.fixed_partitions)
    population = {"nano": set(partition["pi_stationary"]),
                  "nx": set(partition["nx_budgets"]["DFull"])}
    require(len(population["nano"]) == 28143 and len(population["nx"]) == 18441 and
            not (population["nano"] & population["nx"]),
            "frozen Strong client partitions are wrong or overlap")
    static_runs = {r["policy"]: r for r in static["runs"]}
    require(set(static_runs) == {"Baseline", "PyramidFL-profile-slack"},
            "unexpected static policy set")
    adaptive_runs = adaptive["runs"]
    require(len(adaptive_runs) == 1 and
            adaptive_runs[0]["policy"] == "LoRACast-Exposure",
            "unexpected adaptive policy set")
    specs = {
        "Baseline": (args.static_dir / "rtx_runs" / static_runs["Baseline"]["run_id"],
                     static_runs["Baseline"]["run_id"],
                     sha(args.static_dir / "experiment_manifest.json"),
                     static_runs["Baseline"]["source_workload"]),
        "PyramidFL-style": (args.static_dir / "rtx_runs" /
                            static_runs["PyramidFL-profile-slack"]["run_id"],
                            static_runs["PyramidFL-profile-slack"]["run_id"],
                            sha(args.static_dir / "experiment_manifest.json"),
                            static_runs["PyramidFL-profile-slack"]["source_workload"]),
        "LoRACast": (args.loracast_dir / "runs" / adaptive_runs[0]["run_id"],
                     adaptive_runs[0]["run_id"],
                     sha(args.loracast_dir / "experiment_manifest.json"), None),
    }
    entries = {}
    for name in POLICIES:
        path, run_id, manifest_hash, quota = specs[name]
        entries[name] = audit_run(name, path, run_id, manifest_hash,
                                  schedule, population, adaptive, quota)
    require(len({e["run_id"] for e in entries.values()}) == 3,
            "three independent trajectories are required")
    require(len({e["checkpoint_sha256_by_round"]["0"]
                 for e in entries.values()}) == 1 and
            max(e["r0_balanced_validation_nll"] for e in entries.values()) -
            min(e["r0_balanced_validation_nll"] for e in entries.values()) < 1e-9,
            "R0 model/NLL is not shared across the three policies")
    require(len({sha_object(e["scorer"]) for e in entries.values()}) == 1,
            "validation scorer differs across policies")
    partition_hash = sha_object({"strong_manifest": EXPECTED_SOURCE["loracast_strong.json"],
                                 "fixed_partitions": EXPECTED_SOURCE["fixed_partitions.json"]})
    validation_hash = sha_object({"strong_holdout_manifest": EXPECTED_SOURCE["loracast_strong.json"],
                                  "scorer": entries["Baseline"]["scorer"]})
    shared = {"model": "Qwen2.5-0.5B-Instruct",
              "initial_checkpoint_sha256": EXPECTED_R0,
              "partition_manifest_sha256": partition_hash,
              "validation_manifest_sha256": validation_hash,
              "schedule_sha256": EXPECTED_SCHEDULE,
              "source_initial_adapter_sha256": EXPECTED_SOURCE["initial_adapter_round0.pt"],
              "strong_manifest_sha256": EXPECTED_SOURCE["loracast_strong.json"],
              "fixed_partitions_sha256": EXPECTED_SOURCE["fixed_partitions.json"],
              "training_core_sha256": adaptive["rtx_core_runner_sha256"],
              "validation_protocol": entries["Baseline"]["scorer"],
              "schedule_interpretation": ("R1–R49 normal; R50–R100 repeating four normal, "
                   "four Nano contention, four recovery. Fixed-policy schedule is a "
                   "post-training edge time projection; LoRACast uses it only as "
                   "completed-round Nano time feedback for future allocations.")}
    for entry in entries.values():
        entry.update({key: shared[key] for key in
                      ("initial_checkpoint_sha256", "partition_manifest_sha256",
                       "validation_manifest_sha256", "schedule_sha256")})
    report = {"status": "PASS", "matched_schedule_status": "PASS",
              "auditor_sha256": sha(Path(__file__)),
              "scope": "One-seed matched actual RTX LoRA training/NLL; edge contention is measured-trace feedback/projection, not physical 100-round Nano training",
              "shared": shared, "runs": entries,
              "limits": ("Byte hashes, visit counters, quota/aggregation arithmetic, and "
                         "LoRACast native-state bytes are audited. Internal GPU math and "
                         "NLL logits are not recomputed; frozen run_adaptive_training/core source and "
                         "commit-time parity/scoring gates attest those operations. "
                         "PyramidFL-style here is the profile-slack component only.")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=args.output.parent,
                                     prefix=".matched_audit_", suffix=".tmp",
                                     delete=False) as stream:
        temp = Path(stream.name)
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temp, args.output)
    print(json.dumps({"status": "PASS", "report": str(args.output),
                      "r100_nll": {p: entries[p]["r100_balanced_validation_nll"]
                                   for p in POLICIES}}, indent=2))


if __name__ == "__main__":
    main()
