"""Isolated Nano+NX rebalancing variants; never imports into archived runs.

The original estimator and NX sample-then-step candidate search are inherited.
This adapter supplies a Nano reference and ports the original two-client
participation guard to Nano without pretending it is an RPi5. Its three modes
share the same normal allocation path: none, omit, throttle.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from e2fl.strategy.LoRACast import LoRACastCommAware
from quota_throttled_controller import Candidate, CompletedRound, choose_quota

NANO = "jetson_orin_nano_super"
NX = "jetson_orin_nx"


def token_work(samples: int, steps: int, tokens_per_sample: float, tokens_per_step: float) -> float:
    if samples <= 0:
        return 0.0
    sample_work = samples * tokens_per_sample
    return min(sample_work, steps * tokens_per_step) if steps > 0 else sample_work


def retained_work(completed: list[CompletedRound], window_w: int) -> tuple[float, float, tuple[int, ...]]:
    kept = sorted(completed, key=lambda r: r.round_id)[-max(0, window_w-1):] if window_w > 1 else []
    return sum(r.slow_work for r in kept), sum(r.fast_work for r in kept), tuple(r.round_id for r in kept)


def calibrated_floor(a_s: float, a_f: float, maximum_slow_work: float, rho: float,
                     n_clients: int = 2, safety: float = .95) -> float:
    """Same effective-floor arithmetic as original two-client guard."""
    if min(a_s, a_f, maximum_slow_work) < 0 or n_clients < 1 or not 0 < safety <= 1:
        raise ValueError("invalid work/floor calibration")
    total = a_s + a_f + maximum_slow_work
    attainable = (a_s + maximum_slow_work) / total if total > 0 else 0.0
    return min(max(0.0, rho) / n_clients, safety * attainable, .999)


class NanoReferenceLoRACast(LoRACastCommAware):
    """Explicit Nano-reference port with identical normal path across actions.

    `participation_action` selects no regulation, original-style omission, or
    new positive-quota throttling. The historic RPi5 implementation is untouched.
    """

    def __init__(self, *, participation_action: str = "throttle", nano_config=None, **kwargs):
        if participation_action not in {"none", "omit", "throttle"}:
            raise ValueError("participation_action must be none, omit or throttle")
        super().__init__(**kwargs)
        self.participation_action = participation_action
        self.nano_config = dict(nano_config or {})
        self.last_nano_regulation_round = -1
        self.last_nano_decision = {}

    def start(self, *args, **kwargs):
        """Inject run-frozen Nano knobs into the original ServerApp train config."""
        if "train_config" not in kwargs:
            raise ValueError("Nano variant requires keyword train_config")
        train_config = kwargs["train_config"]
        for key in ("reference-nano-samples", "reference-nano-steps",
                    "reference-nano-correction-max-samples",
                    "reference-nano-correction-max-steps",
                    "reference-nano-correction-max-seconds"):
            if key not in self.nano_config:
                raise ValueError(f"Nano run config missing {key}")
            train_config[key] = self.nano_config[key]
        return super().start(*args, **kwargs)

    @staticmethod
    def _required_int(config, key: str, *, minimum: int = 0) -> int:
        if key not in config:
            raise ValueError(f"Nano reference requires explicit {key}")
        value = int(config[key])
        if value < minimum:
            raise ValueError(f"{key} must be >= {minimum}")
        return value

    @staticmethod
    def _required_float(config, key: str) -> float:
        if key not in config:
            raise ValueError(f"Nano reference requires explicit {key}")
        value = float(config[key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
        return value

    def _set_workload(self, config, device: str, uid: str | None, samples: int, steps: int):
        config[f"max_samples_per_round_{device}"] = int(samples)
        config[f"local_epochs_{device}"] = 1 if samples > 0 else 0
        config[f"local_steps_{device}"] = int(steps)
        if uid:
            suffix = self._safe_config_suffix(uid)
            config[f"max_samples_per_round_uid_{suffix}"] = int(samples)
            config[f"local_epochs_uid_{suffix}"] = 1 if samples > 0 else 0
            config[f"local_steps_uid_{suffix}"] = int(steps)

    def _history(self, server_round: int, slow_uid: str, fast_uid: str) -> list[CompletedRound]:
        completed = []
        for key, stats in self.round_stats.items():
            round_id = int(key)
            if round_id >= server_round:
                raise ValueError("current/future round is present in completed-work history")
            clients = stats.get("clients", {}) if isinstance(stats, dict) else {}
            if not isinstance(clients, dict):
                raise ValueError("malformed client history")
            def actual(uid):
                return sum(float(e.get("tokens", 0) or 0) for e in clients.values()
                           if isinstance(e, dict) and str(e.get("device_uid")) == uid)
            completed.append(CompletedRound(round_id, actual(slow_uid), actual(fast_uid)))
        return completed

    def _record_decision(self, config):
        """Write the selected request beside the original strategy logs."""
        payload = dict(self.last_nano_decision)
        payload["participation_action"] = self.participation_action
        payload["requested"] = {
            device: {
                "samples": int(config[f"max_samples_per_round_{device}"]),
                "steps": int(config[f"local_steps_{device}"]),
            }
            for device in (NANO, NX)
        }
        target = Path(self.log_dir) / f"nano_decision_round_{payload['round']:02d}.json"
        target.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")

    def update_train_config(self, server_round, train_config):
        # The parent normal branch handles defaults, estimator-state refresh,
        # LoRA config and generic UID materialization. With no RPi5, its old
        # guard never triggers. The placeholder Pi-reference quota is replaced
        # below before any Flower message is constructed.
        config = super().update_train_config(server_round, train_config)
        if config.get("round_mode") == "PROTECTED_ONLY":
            raise RuntimeError("unexpected original RPi5 protected-only decision in Nano topology")
        if not self.enable_rpi_only_rounds:
            raise RuntimeError("protected-only Flower message filtering must be enabled")

        nano_s = self._required_int(config, "reference-nano-samples", minimum=1)
        nano_h = self._required_int(config, "reference-nano-steps")
        corr_max_s = self._required_int(config, "reference-nano-correction-max-samples", minimum=nano_s)
        corr_max_h = self._required_int(config, "reference-nano-correction-max-steps")
        corr_max_seconds = self._required_float(config, "reference-nano-correction-max-seconds")
        if nano_h > 0 and corr_max_h < nano_h:
            raise ValueError("correction step cap cannot be smaller than normal Nano cap")

        device_map = self._known_client_device_map()
        if device_map and (set(device_map.values()) - {NANO, NX}):
            raise RuntimeError("Nano variant observed an unexpected device type")
        if server_round > 1 and (list(device_map.values()).count(NANO) != 1 or
                                 list(device_map.values()).count(NX) != 1 or len(device_map) != 2):
            raise RuntimeError("Nano variant requires exactly one identified Nano and NX")
        nano_uid = next((uid for uid, dev in device_map.items() if dev == NANO), None)
        nx_uid = next((uid for uid, dev in device_map.items() if dev == NX), None)

        # Explicitly set both device and UID keys: the inherited normal branch
        # otherwise gives Nano the NX quota and may leave stale UID overrides.
        self._set_workload(config, NANO, nano_uid, nano_s, nano_h)
        config["max_samples_per_round_RPi5"] = 0
        config["local_epochs_RPi5"] = 0
        config["local_steps_RPi5"] = 0
        reference = self._predict_train_time(NANO, float(nano_s), config)
        normal_s, normal_h, normal_seconds = self._select_jetson_workload_for_window(reference, config)
        normal_fallback_reason = None
        base_predicted_seconds = None
        if server_round > self.policy_warmup_rounds:
            if not math.isfinite(reference) or reference <= 0:
                raise RuntimeError("Nano reference prediction unavailable after warm-up")
            if not math.isfinite(normal_seconds):
                raise RuntimeError("NX selection prediction is non-finite")
            if normal_seconds <= 0:
                # The inherited search deliberately returns (base S, H=0, 0)
                # when its minimum candidate exceeds the local-compute window.
                # Distinguish that valid fallback from a missing cost estimate.
                base_s = int(self.jetson_base_quota)
                if (normal_s, normal_h) != (base_s, 0):
                    raise RuntimeError("unexpected zero-time NX selection")
                base_cfg = copy.deepcopy(config)
                base_cfg[f"max_samples_per_round_{NX}"] = base_s
                base_cfg[f"local_epochs_{NX}"] = 1
                base_cfg[f"local_steps_{NX}"] = 0
                base_predicted_seconds = self._predict_train_time(NX, float(base_s), base_cfg)
                if not math.isfinite(base_predicted_seconds) or base_predicted_seconds <= 0:
                    raise RuntimeError("NX base prediction unavailable after warm-up")
                if base_predicted_seconds <= reference * self.jetson_target_margin:
                    raise RuntimeError("NX search returned fallback despite a feasible base candidate")
                normal_fallback_reason = "base_candidate_outside_compute_window"
        self._set_workload(config, NX, nx_uid, normal_s, normal_h)
        config["round_mode"] = "BOTH"
        self.protected_client_ids = set()
        self.last_nano_decision = {"round": server_round, "mode": "BOTH", "reference_device": NANO,
                                   "reference_predicted_seconds": reference, "normal_nx_samples": normal_s,
                                   "normal_nx_steps": normal_h, "normal_nx_predicted_seconds": normal_seconds,
                                   "normal_fallback_reason": normal_fallback_reason,
                                   "base_nx_predicted_seconds": base_predicted_seconds}

        if self.participation_action == "none" or server_round <= self.policy_warmup_rounds:
            self._record_decision(config)
            return config
        if self.last_nano_regulation_round == server_round - 1:
            self.last_nano_decision["reason"] = "forced normal round after regulation"
            self._record_decision(config)
            return config
        assert nano_uid is not None and nx_uid is not None
        completed = self._history(server_round, nano_uid, nx_uid)
        a_s, a_f, retained = retained_work(completed, int(self.window_W))
        if a_s + a_f <= 0:
            self._record_decision(config)
            return config

        nano_tps = float(self._estimate_tokens_per_sample_for_device(NANO, config))
        nano_tph = float(self._estimate_tokens_per_step_for_device(NANO, config))
        fast_tps = float(self._estimate_tokens_per_sample_for_device(NX, config))
        fast_tph = float(self._estimate_tokens_per_step_for_device(NX, config))
        if min(nano_tps, nano_tph, fast_tps, fast_tph) <= 0:
            raise RuntimeError("prior token-work rates unavailable")

        # Same bounded correction principle and effective-floor formula as the
        # original Pi guard, now evaluated with Nano's genuine estimator state.
        correction = []
        for samples in range(nano_s, corr_max_s + 1, max(1, int(self.jetson_sample_step))):
            steps = corr_max_h
            candidate_cfg = copy.deepcopy(config)
            self._set_workload(candidate_cfg, NANO, nano_uid, samples, steps)
            predicted = self._predict_train_time(f"{NANO}:{nano_uid}", float(samples), candidate_cfg)
            work = token_work(samples, steps, nano_tps, nano_tph)
            if predicted > 0 and predicted <= corr_max_seconds and work > 0:
                correction.append(Candidate(samples, steps, work, predicted, True))
        if not correction:
            raise RuntimeError("no Nano correction workload passes original predictor and configured time cap")
        maximum = max(correction, key=lambda c: (c.predicted_work, c.samples))
        safety = float(config.get("participation-feasibility-safety", .95))
        safety = min(max(safety, .50), 1.0)  # original clamping
        theta = calibrated_floor(a_s, a_f, maximum.predicted_work,
                                 float(self.participation_fairness_rho), 2, safety)
        prior_share = a_s / (a_s + a_f)
        self.last_nano_decision.update(retained_rounds=retained, prior_share=prior_share,
                                       theta_eff=theta, maximum_correction_work=maximum.predicted_work)
        if theta <= 0 or prior_share >= theta:
            self._record_decision(config)
            return config

        attainable = (a_s + maximum.predicted_work) / (a_s + a_f + maximum.predicted_work)
        target = min(theta + float(self.correction_target_margin), attainable,
                     max(0.0, float(self.participation_fairness_rho)) / 2, .999)
        needed_slow_work = max(0.0, target * a_f / max(1.0 - target, 1e-9) - a_s)
        chosen_slow = next((c for c in sorted(correction, key=lambda c: (c.predicted_work, c.samples))
                            if c.predicted_work >= needed_slow_work), maximum)
        self._set_workload(config, NANO, nano_uid, chosen_slow.samples, chosen_slow.steps)

        if self.participation_action == "throttle":
            normal_fast_work = token_work(normal_s, normal_h, fast_tps, fast_tph)
            normal = Candidate(normal_s, normal_h, normal_fast_work, normal_seconds, normal_seconds > 0)
            alternatives = []
            for samples in range(int(self.jetson_base_quota), normal_s + 1,
                                 max(1, int(self.jetson_sample_step))):
                for steps in range(1, min(samples, int(self.jetson_max_steps)) + 1,
                                   max(1, int(self.jetson_step_increment))):
                    candidate_cfg = copy.deepcopy(config)
                    self._set_workload(candidate_cfg, NX, nx_uid, samples, steps)
                    predicted = self._predict_train_time(NX, float(samples), candidate_cfg)
                    alternatives.append(Candidate(samples, steps,
                                                  token_work(samples, steps, fast_tps, fast_tph), predicted,
                                                  predicted > 0 and predicted <= reference * self.jetson_target_margin))
            decision = choose_quota(next_round=server_round, completed=completed,
                                    window_w=int(self.window_W), theta_eff=theta,
                                    expected_slow_work=chosen_slow.predicted_work, normal=normal,
                                    alternatives=alternatives, warmup_rounds=int(self.policy_warmup_rounds))
            self.last_nano_decision.update(fast_work_bound=decision.fast_work_bound,
                                           projected_share=decision.projected_share,
                                           reason=decision.reason)
            if decision.mode == "THROTTLED_BOTH":
                selected = decision.candidate
                assert selected is not None
                self._set_workload(config, NX, nx_uid, selected.samples, selected.steps)
                config["round_mode"] = "THROTTLED_BOTH"
                self.last_nano_regulation_round = server_round
                self.last_nano_decision.update(mode="THROTTLED_BOTH",
                                               nx_samples=selected.samples, nx_steps=selected.steps)
                self._record_decision(config)
                return config

        # Original-style hard omission and the positive-candidate fallback use
        # the inherited PROTECTED_ONLY message filter (NX receives no TRAIN).
        self._set_workload(config, NX, nx_uid, 0, 0)
        config["round_mode"] = "PROTECTED_ONLY"
        self.protected_client_ids = {nano_uid}
        self.last_nano_regulation_round = server_round
        self.last_nano_decision.update(mode="PROTECTED_ONLY", reason="omit action or no feasible positive candidate")
        self._record_decision(config)
        return config
