"""Run-scoped dynamic-fast-client policy variants and decision evidence.

This module lives only in the isolated pilot app. The original strategy files
and earlier experiment results are not edited.
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path

from e2fl.strategy.LoRACast import LoRACastCommAware


class DecisionEvidence:
    pilot_policy = "unassigned"

    def _predict_train_time(self, identity_key, num_examples, cfg):
        result = super()._predict_train_time(identity_key, num_examples, cfg)
        if getattr(self, '_pilot_capture_candidates', False):
            limit = float(self._pilot_candidate_window) * float(self.jetson_target_margin)
            self._pilot_candidates.append({
                'identity_key': str(identity_key),
                'samples': int(cfg.get('max_samples_per_round_jetson_orin_nx', 0) or 0),
                'step_cap': int(cfg.get('local_steps_jetson_orin_nx', 0) or 0),
                'predicted_compute_s': float(result),
                'reference_window_s': float(self._pilot_candidate_window),
                'allowed_compute_s': limit,
                'decision': ('missing_prediction' if result <= 0 else
                             'accepted_by_time' if result <= limit else
                             'rejected_by_time'),
            })
        return result

    def _select_jetson_workload_for_window(self, target_rpi_time, train_config):
        self._pilot_capture_candidates = True
        self._pilot_candidate_window = float(target_rpi_time)
        self._pilot_candidates = []
        try:
            result = super()._select_jetson_workload_for_window(target_rpi_time, train_config)
        finally:
            self._pilot_capture_candidates = False
        self._pilot_window = float(target_rpi_time)
        self._pilot_selected = [int(result[0]), int(result[1]), float(result[2])]
        return result

    def update_train_config(self, server_round, train_config):
        before = {
            "completed_rounds": sorted(int(r) for r in self.round_stats),
            "round_stats": copy.deepcopy(self.round_stats),
            "time_scale": copy.deepcopy(self.time_scale),
            "beta_sec_per_token": copy.deepcopy(self.beta_sec_per_token),
            "config_before": dict(train_config.items()),
        }
        self._pilot_window = None
        self._pilot_selected = None
        self._pilot_candidates = []
        selected = super().update_train_config(server_round, train_config)
        record = {
            "event": "allocation_decided_before_train_dispatch",
            "policy": self.pilot_policy,
            "round": int(server_round),
            "server_unix_s": time.time(),
            "server_monotonic_s": time.monotonic(),
            "prior_state": before,
            "pi_reference_compute_s": self._pilot_window,
            "safety_margin": float(self.jetson_target_margin),
            "selected_by_original_search": self._pilot_selected,
            "tested_candidates_in_original_search": self._pilot_candidates,
            "config_after": dict(selected.items()),
            "candidate_grid": {
                "jetson_base_quota": int(self.jetson_base_quota),
                "jetson_max_quota": int(self.jetson_max_quota),
                "jetson_sample_step": int(self.jetson_sample_step),
                "jetson_base_steps": int(self.jetson_base_steps),
                "jetson_max_steps": int(self.jetson_max_steps),
                "jetson_step_increment": int(self.jetson_step_increment),
            },
        }
        path = Path(self.log_dir) / f"pilot_decision_round_{int(server_round):02d}.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(record, default=str, sort_keys=True, indent=2) + "\n")
        os.replace(temp, path)
        return selected


class FrozenJetsonCost:
    """Freeze Jetson cost observations after round 4; Pi remains online."""
    _pilot_freeze_cutoff = 4

    @contextmanager
    def _jetson_snapshot(self):
        previous = getattr(self, "_pilot_jetson_scope", False)
        self._pilot_jetson_scope = True
        try:
            yield
        finally:
            self._pilot_jetson_scope = previous

    def _estimator_round_allowed(self, round_number):
        if getattr(self, "_pilot_jetson_scope", False):
            return int(round_number) <= self._pilot_freeze_cutoff
        return super()._estimator_round_allowed(round_number)

    def _predict_train_time(self, identity_key, num_examples, cfg):
        if str(identity_key).startswith("jetson_orin_nx"):
            with self._jetson_snapshot():
                return super()._predict_train_time(identity_key, num_examples, cfg)
        return super()._predict_train_time(identity_key, num_examples, cfg)

    def _estimate_tokens_per_sample_for_device(self, device_name, train_config):
        if device_name == "jetson_orin_nx":
            with self._jetson_snapshot():
                return super()._estimate_tokens_per_sample_for_device(device_name, train_config)
        return super()._estimate_tokens_per_sample_for_device(device_name, train_config)

    def _estimate_tokens_per_step_for_device(self, device_name, train_config):
        if device_name == "jetson_orin_nx":
            with self._jetson_snapshot():
                return super()._estimate_tokens_per_step_for_device(device_name, train_config)
        return super()._estimate_tokens_per_step_for_device(device_name, train_config)

    def _update_time_scale(self, identity_key, predicted_s, observed_s, server_round):
        if (str(identity_key).startswith("jetson_orin_nx")
                and int(server_round) > self._pilot_freeze_cutoff):
            return
        return super()._update_time_scale(identity_key, predicted_s, observed_s, server_round)


class LastCompletedStepRate:
    """Jetson candidate cost = effective steps × latest valid loop seconds/step."""
    def _predict_train_time(self, identity_key, num_examples, cfg):
        if not str(identity_key).startswith("jetson_orin_nx"):
            return super()._predict_train_time(identity_key, num_examples, cfg)
        observations = []
        for round_number, stats in self.round_stats.items():
            clients = stats.get("clients", {}) if isinstance(stats, dict) else {}
            for entry in clients.values():
                if entry.get("device_name") != "jetson_orin_nx":
                    continue
                steps = float(entry.get("local_steps", 0) or 0)
                duration = float(entry.get("train_time", 0) or 0)
                if steps > 0 and duration > 0:
                    observations.append((int(round_number), duration / steps))
        if not observations:
            return 0.0  # same cold-start J8 fallback as the inherited search
        rate = max(observations)[1]
        samples, epochs = self._resolve_effective_workload(
            identity_key, "jetson_orin_nx", cfg, num_examples)
        if samples <= 0 or epochs <= 0:
            return 0.0
        batch = int(cfg.get("batch_size", 1) or 1)
        full_steps = int(math.ceil(samples / batch) * epochs)
        cap = int(self._resolve_effective_steps(identity_key, "jetson_orin_nx", cfg))
        effective_steps = min(full_steps, cap) if cap > 0 else full_steps
        return float(effective_steps) * rate


class ServerPhaseEvidence:
    """Observe server phases without changing allocation or aggregation."""

    def _same_run_event(self, server_round, event, **fields):
        row = {"round": int(server_round), "event": event,
               "server_monotonic_s": time.monotonic(), "server_unix_s": time.time(),
               **fields}
        path = Path(self.log_dir) / "same_run_server_phases.jsonl"
        with path.open("a", buffering=1) as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def configure_train(self, server_round, arrays, config, grid):
        self._same_run_event(server_round, "server_configure_train_enter")
        result = super().configure_train(server_round, arrays, config, grid)
        self._same_run_event(server_round, "server_train_messages_ready",
                             message_count=len(result) if hasattr(result, '__len__') else None)
        return result

    def aggregate_train(self, server_round, replies):
        self._same_run_event(server_round, "server_all_train_replies_available",
                             reply_count=len(replies) if hasattr(replies, '__len__') else None)
        result = super().aggregate_train(server_round, replies)
        self._same_run_event(server_round, "server_aggregation_end")
        return result


class PilotFrozenCost(ServerPhaseEvidence, DecisionEvidence, FrozenJetsonCost, LoRACastCommAware):
    pilot_policy = "A_frozen_jetson_cost_after_round4"


class PilotLastRate(ServerPhaseEvidence, DecisionEvidence, LastCompletedStepRate, LoRACastCommAware):
    pilot_policy = "B_last_completed_step_rate"


class PilotOnline(ServerPhaseEvidence, DecisionEvidence, LoRACastCommAware):
    pilot_policy = "C_original_online_cost"


class ProbeFixed(DecisionEvidence, LoRACastCommAware):
    """One-round fixed bound probe with observational server timestamps only."""
    pilot_policy = "fixed_probe_no_controller_comparison"

    def _probe_event(self, server_round, event, **fields):
        row = {"round": int(server_round), "event": event,
               "server_monotonic_s": time.monotonic(), "server_unix_s": time.time(),
               **fields}
        path = Path(self.log_dir) / "probe_server_phases.jsonl"
        with path.open("a", buffering=1) as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def configure_train(self, server_round, arrays, config, grid):
        self._probe_event(server_round, "server_configure_train_enter")
        result = super().configure_train(server_round, arrays, config, grid)
        self._probe_event(server_round, "server_train_messages_ready",
                          message_count=len(result) if hasattr(result, '__len__') else None)
        return result

    def aggregate_train(self, server_round, replies):
        replies = list(replies)
        self._probe_event(server_round, "server_all_train_replies_available",
                          reply_count=len(replies),
                          client_source_ids=[int(x.metadata.src_node_id) for x in replies])
        result = super().aggregate_train(server_round, replies)
        self._probe_event(server_round, "server_aggregation_end")
        return result


class ProbePairFixed(ProbeFixed):
    """One-round diagnostic with explicit Nano/NX quotas; no online allocation."""
    pilot_policy = "prespecified_pair_workload_diagnostic"

    def update_train_config(self, server_round, train_config):
        if int(server_round) != 1:
            raise ValueError("ProbePairFixed is limited to one round")
        before = dict(train_config.items())
        # Run the original refresh/default path, then replace only the two
        # device-specific sample/step requests for this fixed-workload probe.
        result = LoRACastCommAware.update_train_config(self, server_round, train_config)
        nano = int(result["probe-nano-quota"])
        nx = int(result["probe-nx-quota"])
        if nano not in (320, 960) or nx not in (480, 960):
            raise ValueError(f"unfrozen diagnostic workload: Nano={nano}, NX={nx}")
        result["round_mode"] = "BOTH"
        result["max_samples_per_round_jetson_orin_nano_super"] = nano
        result["max_samples_per_round_jetson_orin_nx"] = nx
        result["local_epochs_jetson_orin_nano_super"] = 1
        result["local_epochs_jetson_orin_nx"] = 1
        result["local_steps_jetson_orin_nano_super"] = 0
        result["local_steps_jetson_orin_nx"] = 0
        record = {
            "event": "prespecified_pair_assignment_before_dispatch",
            "round": int(server_round),
            "server_unix_s": time.time(),
            "server_monotonic_s": time.monotonic(),
            "config_before": before,
            "nano_samples": nano,
            "nx_samples": nx,
            "config_after": dict(result.items()),
        }
        path = Path(self.log_dir) / f"pair_assignment_round_{server_round:02d}.json"
        path.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
        return result
