"""Isolated Nano+NX physical wiring check for the E=1 exposure selector.

The Nano reference and NX allocator are inherited unchanged from the audited
Nano reference port. This is a topology-specific integration check, not a Pi5
power/latency validation or a new learning policy.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from exposure_controller import Candidate, Policy, choose_workload
from nano_reference_strategy import NANO, NX, NanoReferenceLoRACast
from policy_resume import run_remaining_rounds


class NanoFinalExposure(NanoReferenceLoRACast):
    def __init__(self, *, exposure_max: float | None, nx_eligible_unique: int,
                 nx_minimum_samples: int = 0, nx_maximum_samples: int = 0,
                 nx_maximum_epochs: int = 1, **kwargs):
        if (nx_eligible_unique <= 0 or nx_minimum_samples < 0 or nx_maximum_samples < 0
                or nx_maximum_samples and nx_minimum_samples > nx_maximum_samples):
            raise ValueError("invalid NX dataset/workload policy")
        if exposure_max is not None and (not math.isfinite(exposure_max) or exposure_max < 0):
            raise ValueError("invalid cumulative exposure cap")
        if nx_maximum_epochs != 1:
            raise ValueError("E>1 lacks a validated edge cost profile")
        super().__init__(participation_action="none", **kwargs)
        self.exposure_max = exposure_max
        self.nx_eligible_unique = int(nx_eligible_unique)
        self.nx_minimum_samples = int(nx_minimum_samples)
        self.nx_maximum_samples = int(nx_maximum_samples)
        self.nx_cumulative_samples = 0
        self.nx_seen_ids = set()
        self.nx_unique_visited = 0
        self.exposure_decisions = {}
        self.resume_completed_round = 0

    def start(self, *args, **kwargs):
        if not self.resume_completed_round:
            return super().start(*args, **kwargs)
        if args or "train_config" not in kwargs:
            raise ValueError("round-boundary recovery requires keyword train_config")
        for key in ("reference-nano-samples", "reference-nano-steps",
                    "reference-nano-correction-max-samples",
                    "reference-nano-correction-max-steps",
                    "reference-nano-correction-max-seconds"):
            kwargs["train_config"][key] = self.nano_config[key]
        return run_remaining_rounds(
            self, completed_round=self.resume_completed_round, **kwargs
        )

    def _record_exposure_decision(self, round_number: int, payload: dict) -> None:
        self.exposure_decisions[round_number] = payload
        target = Path(self.log_dir) / f"final_exposure_decision_round_{round_number:03d}.json"
        target.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")

    def update_train_config(self, server_round, train_config):
        config = super().update_train_config(server_round, train_config)
        if config.get("round_mode") != "BOTH":
            raise RuntimeError("unexpected participation rebalancing")
        normal_s = int(config.get(f"max_samples_per_round_{NX}", 0) or 0)
        normal_h = int(config.get(f"local_steps_{NX}", 0) or 0)
        normal_e = int(config.get(f"local_epochs_{NX}", 0) or 0)
        if normal_s < 0 or normal_h < 0 or normal_e != 1:
            raise RuntimeError("unexpected native NX workload semantics")
        minimum = max(int(self.jetson_base_quota), self.nx_minimum_samples)
        maximum = min(int(self.jetson_max_quota),
                      self.nx_maximum_samples or int(self.jetson_max_quota))
        if maximum < minimum:
            raise RuntimeError("user NX ceiling lies below native minimum")
        batch = int(config["batch_size"])
        seq = int(config.get("max_seq_len", 64) or 64)
        normal_work = Candidate(normal_s, 1, normal_h, batch, seq, 0.0).executed()[1]
        cap_total = (math.floor(self.exposure_max * self.nx_eligible_unique + 1e-9)
                     if self.exposure_max is not None else None)
        remaining = cap_total - self.nx_cumulative_samples if cap_total is not None else None
        config["exposure_tracking_enabled"] = True
        if (normal_s <= min(maximum, self.nx_eligible_unique)
                and (remaining is None or remaining >= normal_work)):
            self._record_exposure_decision(server_round, {
                "round": int(server_round), "action": "normal", "normal_s": normal_s,
                "selected_s": normal_s, "selected_h": normal_h,
                "remaining_before": remaining, "eligible_unique": self.nx_eligible_unique,
            })
            return config

        nano_s = int(config.get(f"max_samples_per_round_{NANO}", 0) or 0)
        reference = float(self._predict_train_time(NANO, float(nano_s), config))
        uncalibrated = not math.isfinite(reference) or reference <= 0
        if uncalibrated:
            # The first-round monotonic S proxy has no seconds interpretation.
            reference = float(max(1, normal_s))
        uid_map = self._known_client_device_map()
        nx_uid = next((uid for uid, dev in uid_map.items() if dev == NX), None)

        def candidate(s: int) -> Candidate:
            if uncalibrated:
                predicted = float(s)
            else:
                trial = copy.deepcopy(config)
                self._set_workload(trial, NX, nx_uid, s, normal_h)
                predicted = float(self._predict_train_time(NX, float(s), trial))
                if predicted <= 0 or not math.isfinite(predicted):
                    predicted = math.inf
            return Candidate(s, 1, normal_h, batch, seq, predicted)

        normal = candidate(normal_s)
        alternatives = [candidate(s) for s in range(
            minimum, min(normal_s, maximum, self.nx_eligible_unique) + 1,
            int(self.jetson_sample_step))]
        policy = Policy(maximum, minimum, 1, 0, batch, seq, self.exposure_max)
        decision = choose_workload(
            normal=normal, alternatives=alternatives, policy=policy,
            eligible_unique_samples=self.nx_eligible_unique,
            cumulative_executed=self.nx_cumulative_samples,
            previously_visited_unique=self.nx_unique_visited,
            compute_window_s=reference * float(self.jetson_target_margin),
        )
        selected = decision.selected.samples if decision.selected else 0
        selected_h = normal_h if selected else 0
        self._set_workload(config, NX, nx_uid, selected, selected_h)
        if selected == 0:
            nano_ids = {uid for uid, dev in uid_map.items() if dev == NANO}
            if len(nano_ids) != 1:
                raise RuntimeError("cannot omit NX before Nano destination is known")
            self.protected_client_ids = nano_ids
            config["round_mode"] = "PROTECTED_ONLY"
        self._record_exposure_decision(server_round, {
            "round": int(server_round), "action": decision.action,
            "normal_s": normal_s, "normal_h": normal_h,
            "selected_s": selected, "selected_h": selected_h,
            "expected_processed_samples": decision.actual_processed_samples,
            "remaining_before": remaining, "eligible_unique": self.nx_eligible_unique,
            "native_minimum_s": minimum, "native_sample_step": int(self.jetson_sample_step),
            "candidate_cost_status": ("uncalibrated_monotonic_S_proxy" if uncalibrated
                                      else "native_estimator_prediction"),
            "predicted_residual_window_s": (None if uncalibrated else
                                             decision.predicted_residual_window_s),
        })
        return config

    def aggregate_train(self, server_round, replies):
        replies = list(replies)
        observed = 0
        visited = set()
        for msg in replies:
            if not msg.has_content() or "metrics" not in msg.content:
                continue
            metrics = dict(msg.content["metrics"].items())
            device = self._decode_metric_string(metrics.get("device_name_chars"))
            if device != NX:
                continue
            if int(metrics.get("eligible_unique_samples", -1)) != self.nx_eligible_unique:
                raise RuntimeError("client/server eligible-data size mismatch")
            count = int(metrics.get("num-examples", 0))
            ids = metrics.get("visited_sample_ids", [])
            if len(ids) != count:
                raise RuntimeError("client did not report actual visited samples")
            observed += count
            visited.update(int(x) for x in ids)
        expected = int(self.exposure_decisions[server_round]["selected_s"])
        if observed > expected:
            raise RuntimeError("executed samples exceed selected quota")
        if self.exposure_max is not None and self.nx_cumulative_samples + observed > math.floor(
                self.exposure_max * self.nx_eligible_unique + 1e-9):
            raise RuntimeError("cumulative exposure violated before aggregation")
        arrays, metrics = super().aggregate_train(server_round, replies)
        if arrays is not None:
            self.nx_cumulative_samples += observed
            self.nx_seen_ids.update(visited)
            self.nx_unique_visited = len(self.nx_seen_ids)
        return arrays, metrics
