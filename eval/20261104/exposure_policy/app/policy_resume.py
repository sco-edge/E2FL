"""Run-scoped, round-boundary recovery for the isolated Nano/NX policy pilot.

This module does not change the cost estimator or workload decision.  A recovery
starts only from a completed server checkpoint and the strategy state recorded
at that same boundary.  The Flower process is restarted; no partial round is
replayed or counted as complete.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path

from flwr.app import ConfigRecord
from flwr.serverapp.strategy.result import Result


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_completed_round(strategy, completed_round: int, checkpoint: str) -> None:
    """Commit a model/state pair; manifest appears only after both are durable."""
    r = int(completed_round)
    if r < 1:
        raise ValueError("only completed training rounds can be snapshotted")
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    directory = Path(strategy.log_dir) / "resume_state"
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / f"round_{r:02d}.pkl"
    manifest_path = directory / f"round_{r:02d}.json"
    if state_path.exists() or manifest_path.exists():
        raise FileExistsError(f"round {r} recovery state already exists")
    if r not in strategy.round_stats or r not in strategy.round_configs:
        raise RuntimeError("strategy did not commit the matching round")
    payload = {
        "schema": 1,
        "completed_round": r,
        "policy_class": type(strategy).__name__,
        "experiment_id": strategy.experiment_id,
        "state": strategy.__dict__,
    }
    temporary = state_path.with_suffix(".pkl.tmp")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, state_path)
        manifest = {
            "schema": 1,
            "completed_round": r,
            "policy_class": type(strategy).__name__,
            "experiment_id": strategy.experiment_id,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha(checkpoint_path),
            "state_file": state_path.name,
            "state_sha256": _sha(state_path),
        }
        temp_manifest = manifest_path.with_suffix(".json.tmp")
        with temp_manifest.open("w") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_manifest, manifest_path)
    finally:
        temporary.unlink(missing_ok=True)


def load_completed_round(strategy, completed_round: int, checkpoint: str) -> None:
    """Restore only a locally recorded model/state pair with matching hashes."""
    r = int(completed_round)
    directory = Path(strategy.log_dir) / "resume_state"
    manifest_path = directory / f"round_{r:02d}.json"
    manifest = json.loads(manifest_path.read_text())
    state_path = directory / manifest["state_file"]
    checkpoint_path = Path(checkpoint)
    if (
        manifest["schema"] != 1
        or manifest["completed_round"] != r
        or manifest["policy_class"] != type(strategy).__name__
        or manifest["experiment_id"] != strategy.experiment_id
        or Path(manifest["checkpoint"]) != checkpoint_path
        or _sha(checkpoint_path) != manifest["checkpoint_sha256"]
        or _sha(state_path) != manifest["state_sha256"]
    ):
        raise RuntimeError("round-boundary recovery manifest mismatch")
    # The pickle is generated and consumed only in this run-scoped output tree.
    payload = pickle.loads(state_path.read_bytes())
    if (
        payload["schema"] != 1
        or payload["completed_round"] != r
        or payload["policy_class"] != type(strategy).__name__
        or payload["experiment_id"] != strategy.experiment_id
    ):
        raise RuntimeError("recovery state identity mismatch")
    state = payload["state"]
    if r not in state["round_stats"] or r not in state["round_configs"]:
        raise RuntimeError("recovery state lacks the completed round")
    strategy.__dict__.update(state)
    strategy.resume_completed_round = r


def run_remaining_rounds(strategy, *, grid, initial_arrays, num_rounds: int,
                         completed_round: int, timeout: float,
                         train_config=None, evaluate_config=None,
                         evaluate_fn=None):
    """Same Flower FedAvg 1.21 round order, starting at the next logical round."""
    if not 0 < completed_round < num_rounds:
        raise ValueError("invalid completed round for recovery")
    train_config = ConfigRecord() if train_config is None else train_config
    evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
    result = Result()
    arrays = initial_arrays
    for current_round in range(completed_round + 1, num_rounds + 1):
        train_replies = grid.send_and_receive(
            messages=strategy.configure_train(current_round, arrays, train_config, grid),
            timeout=timeout,
        )
        agg_arrays, agg_train_metrics = strategy.aggregate_train(current_round, train_replies)
        if agg_arrays is not None:
            result.arrays = agg_arrays
            arrays = agg_arrays
        if agg_train_metrics is not None:
            result.train_metrics_clientapp[current_round] = agg_train_metrics
        evaluate_replies = grid.send_and_receive(
            messages=strategy.configure_evaluate(current_round, arrays, evaluate_config, grid),
            timeout=timeout,
        )
        agg_evaluate_metrics = strategy.aggregate_evaluate(current_round, evaluate_replies)
        if agg_evaluate_metrics is not None:
            result.evaluate_metrics_clientapp[current_round] = agg_evaluate_metrics
        if evaluate_fn:
            evaluated = evaluate_fn(current_round, arrays)
            if evaluated is not None:
                result.evaluate_metrics_serverapp[current_round] = evaluated
    return result
