# e2fl/strategy/FedAvgLoRA.py
import csv, os, time, json, math, copy
import numpy as np
from collections.abc import Iterable
from logging import INFO, WARNING
from typing import Callable, Optional
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.result import Result
from flwr.serverapp.strategy.strategy_utils import (
    aggregate_arrayrecords,
    aggregate_metricrecords,
    sample_nodes,
    validate_message_reply_consistency,
)
from flwr.server import Grid
from flwr.common import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MessageType,
    MetricRecord,
    log,
)
# https://github.com/adap/flower/blob/v1.22.0/framework/py/flwr/serverapp/strategy/fedavg.py

class FedAvgLoRA(FedAvg):
    @staticmethod
    def _safe_config_suffix(value: str) -> str:
        """Convert device_uid/device_name into a config-key-safe suffix."""
        return str(value).replace("-", "_").replace(".", "_").replace(":", "_")

    @staticmethod
    def _decode_metric_string(value) -> str:
        """Decode a Flower MetricRecord-safe string encoded as list[int]."""
        if value in ("", None):
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            try:
                return "".join(chr(int(x)) for x in value)
            except Exception:
                return ""
        return ""

    def _default_workload_for_device(self, device_name: str, train_config: ConfigRecord) -> tuple[int, int]:
        """Return the current device-type workload used as fallback."""
        try:
            if device_name == "RPi5":
                samples = int(train_config.get("max_samples_per_round_RPi5", train_config.get("max_samples_per_round", 0)) or 0)
                epochs = int(train_config.get("local_epochs_RPi5", train_config.get("local_epochs", 1)) or 0)
            elif device_name == "jetson_orin_nx":
                samples = int(train_config.get("max_samples_per_round_jetson_orin_nx", train_config.get("max_samples_per_round", 0)) or 0)
                epochs = int(train_config.get("local_epochs_jetson_orin_nx", train_config.get("local_epochs", 1)) or 0)
            elif device_name == "jetson_orin_nano_super":
                samples = int(train_config.get("max_samples_per_round_jetson_orin_nano_super", train_config.get("max_samples_per_round", 0)) or 0)
                epochs = int(train_config.get("local_epochs_jetson_orin_nano_super", train_config.get("local_epochs", 1)) or 0)
            else:
                samples = int(train_config.get("max_samples_per_round", 0) or 0)
                epochs = int(train_config.get("local_epochs", 1) or 0)
        except Exception:
            samples, epochs = 0, 0
        return samples, epochs

    def _split_identity_key(self, identity_key: str) -> tuple[str, str]:
        """Split grouped or per-client prediction key into device name and UID."""
        if identity_key in ("", None):
            return "", ""
        parts = str(identity_key).split(":", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return str(identity_key), ""

    def _resolve_effective_workload(
        self,
        identity_key: str,
        device_name: str,
        cfg: ConfigRecord,
        num_examples: float | None = None,
    ) -> tuple[float, float]:
        """Resolve sample cap and epochs used by a prediction target.

        UID-specific workload overrides have priority over device-type keys.
        This keeps pred_train_time_self aligned with the workload actually sent
        to each client and later logged in the CSV.
        """
        key_device, key_uid = self._split_identity_key(identity_key)
        if device_name in ("", None):
            device_name = key_device

        uid_suffix = self._safe_config_suffix(key_uid) if key_uid not in ("", None) else ""
        sample_value = None
        epoch_value = None

        if uid_suffix:
            sample_value = cfg.get(f"max_samples_per_round_uid_{uid_suffix}")
            epoch_value = cfg.get(f"local_epochs_uid_{uid_suffix}")

        if sample_value in ("", None):
            if device_name == "RPi5":
                sample_value = cfg.get("max_samples_per_round_RPi5", cfg.get("max_samples_per_round", num_examples))
            elif device_name == "jetson_orin_nx":
                sample_value = cfg.get("max_samples_per_round_jetson_orin_nx", cfg.get("max_samples_per_round", num_examples))
            elif device_name == "jetson_orin_nano_super":
                sample_value = cfg.get(
                    "max_samples_per_round_jetson_orin_nano_super",
                    cfg.get("max_samples_per_round_jetson_orin_nx", cfg.get("max_samples_per_round", num_examples)),
                )
            else:
                sample_value = cfg.get("max_samples_per_round", num_examples)

        if epoch_value in ("", None):
            if device_name == "RPi5":
                epoch_value = cfg.get("local_epochs_RPi5", cfg.get("local_epochs", 1))
            elif device_name == "jetson_orin_nx":
                epoch_value = cfg.get("local_epochs_jetson_orin_nx", cfg.get("local_epochs", 1))
            elif device_name == "jetson_orin_nano_super":
                epoch_value = cfg.get(
                    "local_epochs_jetson_orin_nano_super",
                    cfg.get("local_epochs_jetson_orin_nx", cfg.get("local_epochs", 1)),
                )
            else:
                epoch_value = cfg.get("local_epochs", 1)

        explicit_zero_epoch = False
        try:
            epochs = float(epoch_value if epoch_value not in ("", None) else 1.0)
            explicit_zero_epoch = epoch_value not in ("", None) and epochs <= 0.0
        except Exception:
            epochs = 1.0
            explicit_zero_epoch = False

        explicit_zero_sample = False
        try:
            explicit_zero_sample = sample_value not in ("", None) and float(sample_value) <= 0.0
        except Exception:
            explicit_zero_sample = False

        try:
            samples = float(sample_value if sample_value not in ("", None) else (num_examples or 0.0))
        except Exception:
            samples = float(num_examples or 0.0)

        if samples <= 0.0 and num_examples not in ("", None) and not explicit_zero_sample:
            try:
                samples = float(num_examples or 0.0)
            except Exception:
                samples = 0.0
        if epochs <= 0.0 and not explicit_zero_epoch:
            epochs = 1.0
        if explicit_zero_epoch or explicit_zero_sample:
            # Preserve explicit omit/slow-only semantics. A zero workload should
            # remain zero rather than being converted back to one epoch or all
            # available examples by fallback logic.
            samples = 0.0 if explicit_zero_sample else samples
            epochs = 0.0 if explicit_zero_epoch else epochs
        return samples, epochs

    def _resolve_effective_epochs_for_round(
        self,
        identity_key: str,
        device_name: str,
        cfg: ConfigRecord,
    ) -> float:
        """Resolve local epochs from a historical round config."""
        _, epochs = self._resolve_effective_workload(identity_key, device_name, cfg, None)
        return float(epochs)


    def _apply_uid_specific_workloads(self, server_round: int, train_config: ConfigRecord) -> None:
        """Materialize device-level workloads into per-client config keys.

        Explicit UID-specific assignments created by the per-client correction
        policy have priority and are never overwritten here.
        """
        if server_round <= 1 or not self.round_stats:
            return

        prev_rounds = []
        for r in self.round_stats.keys():
            try:
                if int(r) < int(server_round):
                    prev_rounds.append(int(r))
            except Exception:
                pass
        if not prev_rounds:
            return

        latest_round = max(prev_rounds)
        latest_stats = self.round_stats.get(latest_round, {})
        clients = latest_stats.get("clients", {}) if isinstance(latest_stats, dict) else {}
        if not isinstance(clients, dict) or not clients:
            return

        for identity_key, entry in clients.items():
            dev = entry.get("device_name", "")
            uid = entry.get("device_uid", "")
            if uid in ("", None):
                continue

            suffix = self._safe_config_suffix(uid)
            sample_key = f"max_samples_per_round_uid_{suffix}"
            epoch_key = f"local_epochs_uid_{suffix}"
            step_key = (f"local_steps_uid_{suffix}")

            if (
                train_config.get(sample_key) not in ("", None)
                or train_config.get(epoch_key) not in ("", None)
                or train_config.get(step_key) not in ("", None)
            ):
                continue

            samples, epochs = self._default_workload_for_device(dev, train_config)
            samples = max(0, int(samples))
            epochs = max(0, int(epochs))

            if dev == "jetson_orin_nx":
                steps = int(train_config.get("local_steps_jetson_orin_nx", 0,) or 0)
            elif dev == "jetson_orin_nano_super":
                steps = int(train_config.get("local_steps_jetson_orin_nano_super",
                        train_config.get("local_steps_jetson_orin_nx", 0,),) or 0)
            else:
                steps = int(train_config.get("local_steps_RPi5", 0,) or 0)

            train_config[sample_key] = samples
            train_config[epoch_key] = epochs
            train_config[step_key] = steps
            log(
                INFO,
                "[FedAvgLoRA][ROUND %s] UID workload %s (%s/%s): "
                "samples=%s epochs=%s steps=%s "
                "[materialized device assignment]",
                server_round,
                suffix,
                dev,
                uid,
                samples,
                epochs,
                steps,
            )
    def _resolve_effective_steps(
        self,
        identity_key: str,
        device_name: str,
        cfg: ConfigRecord,
    ) -> int:
        """Resolve optimizer-step cap H for a prediction target."""

        _, key_uid = self._split_identity_key(identity_key)

        uid_suffix = (
            self._safe_config_suffix(key_uid)
            if key_uid not in ("", None)
            else ""
        )

        value = None

        if uid_suffix:
            value = cfg.get(
                f"local_steps_uid_{uid_suffix}"
            )

        if value in ("", None):
            if device_name == "RPi5":
                value = cfg.get(
                    "local_steps_RPi5",
                    cfg.get("local_steps", 0),
                )
            elif device_name == "jetson_orin_nx":
                value = cfg.get(
                    "local_steps_jetson_orin_nx",
                    cfg.get("local_steps", 0),
                )
            elif device_name == "jetson_orin_nano_super":
                value = cfg.get(
                    "local_steps_jetson_orin_nano_super",
                    cfg.get(
                        "local_steps_jetson_orin_nx",
                        cfg.get("local_steps", 0),
                    ),
                )
            else:
                value = cfg.get(
                    "local_steps",
                    0,
                )

        try:
            return max(0, int(value or 0))
        except Exception:
            return 0
    def _client_window_token_totals(self) -> dict[str, float]:
        """Return recent W-round token totals per client identity."""
        totals: dict[str, float] = {}
        if not self.round_stats:
            return totals

        rounds = []
        for r in self.round_stats.keys():
            try:
                rounds.append(int(r))
            except Exception:
                pass
        if not rounds:
            return totals

        start_round = max(rounds) - max(1, int(self.window_W)) + 1
        for r in sorted(rounds):
            if r < start_round:
                continue
            stats = self.round_stats.get(r, {})
            clients = stats.get("clients", {}) if isinstance(stats, dict) else {}
            if not isinstance(clients, dict):
                continue
            for client_key, entry in clients.items():
                if not isinstance(entry, dict):
                    continue
                identity = str(entry.get("device_uid") or entry.get("cid") or client_key)
                if not identity:
                    continue
                try:
                    tokens = float(entry.get("tokens", 0.0) or 0.0)
                except Exception:
                    tokens = 0.0
                totals[identity] = totals.get(identity, 0.0) + max(0.0, tokens)

        return totals

    def _known_client_identities(self) -> set[str]:
        """Return client identities seen by the server so far."""
        ids: set[str] = set()
        if not self.round_stats:
            return ids

        for stats in self.round_stats.values():
            clients = stats.get("clients", {}) if isinstance(stats, dict) else {}
            if not isinstance(clients, dict):
                continue
            for client_key, entry in clients.items():
                if not isinstance(entry, dict):
                    continue
                identity = str(entry.get("device_uid") or entry.get("cid") or client_key)
                if identity:
                    ids.add(identity)

        return ids

    def _compute_underrepresented_clients(self) -> tuple[set[str], float, dict[str, float]]:
        """Find clients whose normalized participation is below rho times fair share.

        For N known clients, the fair target is pi_i = 1/N. A client is protected
        when q_i < rho*pi_i. This keeps the threshold independent of the number
        or type of devices.
        """
        ids = self._known_client_identities()
        totals = self._client_window_token_totals()
        ids.update(totals.keys())
        if not ids:
            return set(), 0.0, {}

        n = max(1, len(ids))
        threshold = float(self.participation_fairness_rho) / float(n)
        total_tokens = float(sum(max(0.0, v) for v in totals.values()))

        shares: dict[str, float] = {}
        protected: set[str] = set()
        for identity in ids:
            q = (float(totals.get(identity, 0.0)) / total_tokens) if total_tokens > 0.0 else 0.0
            shares[identity] = q
            if total_tokens > 0.0 and q < threshold:
                protected.add(identity)

        return protected, threshold, shares

    def _recent_device_count(self, device_name: str) -> int:
        """Return the recent number of clients for a device group."""
        if not self.round_stats:
            return 1
        latest_rounds = []
        for r in self.round_stats.keys():
            try:
                latest_rounds.append(int(r))
            except Exception:
                pass
        if not latest_rounds:
            return 1
        latest_round = max(latest_rounds)
        latest_stats = self.round_stats.get(latest_round, {})
        clients = latest_stats.get("clients", {}) if isinstance(latest_stats, dict) else {}
        if not isinstance(clients, dict) or not clients:
            return 1
        count = 0
        for entry in clients.values():
            if isinstance(entry, dict) and entry.get("device_name") == device_name:
                count += 1
        return max(1, count)

    def _estimate_tokens_per_sample_for_device(self, device_name: str, train_config: ConfigRecord) -> float:
        """Estimate realized tokens per sample for a device from recent rounds.

        `max_seq_len` is only a cap. The realized token count depends on the
        tokenizer, prompt template, dataset, truncation/packing behavior, and
        model-specific preprocessing. Use recent observations when available so
        the participation guard reasons in the same token-equivalent units as
        `window_history`.
        """
        values: list[tuple[int, float]] = []
        for r, stats in self.round_stats.items():
            try:
                rr = int(r)
            except Exception:
                rr = 0
            clients = stats.get("clients", {}) if isinstance(stats, dict) else {}
            if not isinstance(clients, dict):
                continue
            for entry in clients.values():
                if not isinstance(entry, dict) or entry.get("device_name") != device_name:
                    continue
                try:
                    tokens = float(entry.get("tokens", 0.0) or 0.0)
                except Exception:
                    tokens = 0.0
                try:
                    samples = float(entry.get("max_samples", entry.get("num_examples", 0.0)) or 0.0)
                except Exception:
                    samples = 0.0
                try:
                    epochs = float(entry.get("local_epochs", 1.0) or 1.0)
                except Exception:
                    epochs = 1.0
                denom = samples * max(epochs, 1.0)
                if tokens > 0.0 and denom > 0.0:
                    values.append((rr, tokens / denom))

        if values:
            values.sort(key=lambda x: x[0], reverse=True)
            recent = [v for _, v in values[: max(1, self.max_beta_history)]]
            return float(np.median(recent))

        try:
            seq_len = float(train_config.get("max_seq_len", self.tokens_per_example) or self.tokens_per_example)
        except Exception:
            seq_len = float(self.tokens_per_example)
        if seq_len <= 0.0:
            seq_len = float(self.tokens_per_example)
        return float(seq_len)
    def _estimate_tokens_per_step_for_device(
        self,
        device_name: str,
        train_config: ConfigRecord,
    ) -> float:
        """Estimate realized tokens per optimizer step."""

        values = []

        for round_key, stats in self.round_stats.items():
            try:
                round_num = int(round_key)
            except Exception:
                round_num = 0

            clients = (
                stats.get("clients", {})
                if isinstance(stats, dict)
                else {}
            )

            if not isinstance(clients, dict):
                continue

            for entry in clients.values():
                if (
                    not isinstance(entry, dict)
                    or entry.get("device_name") != device_name
                ):
                    continue

                try:
                    tokens = float(
                        entry.get("tokens", 0.0) or 0.0
                    )
                    steps = float(
                        entry.get("local_steps", 0.0) or 0.0
                    )
                except Exception:
                    continue

                if tokens > 0.0 and steps > 0.0:
                    values.append(
                        (round_num, tokens / steps)
                    )

        if values:
            values.sort(
                key=lambda item: item[0],
                reverse=True,
            )

            recent = [
                value
                for _, value in values[
                    : max(1, self.max_beta_history)
                ]
            ]

            return float(np.median(recent))

        # Warm-up fallback.
        try:
            batch_size = float(
                train_config.get("batch_size", 1) or 1
            )
        except Exception:
            batch_size = 1.0

        tokens_per_sample = (
            self._estimate_tokens_per_sample_for_device(
                device_name,
                train_config,
            )
        )

        return (
            max(1.0, batch_size)
            * max(1.0, tokens_per_sample)
        )
    def _estimate_group_work_for_next_round(self, device_name: str, train_config: ConfigRecord) -> float:
        """Estimate next-round group contribution in the same units as window_history."""
        samples, epochs = self._default_workload_for_device(device_name, train_config)
        count = self._recent_device_count(device_name)
        tokens_per_sample = self._estimate_tokens_per_sample_for_device(device_name, train_config)
        try:
            return max(0.0, float(samples) * float(epochs) * float(tokens_per_sample) * float(count))
        except Exception:
            return 0.0

    def _effective_slow_group_threshold(self) -> float:
        """Return the slow-group participation threshold for the current run.

        In slow-group mode, rho means the fraction of a fair-share target that
        should be preserved. With N active/known clients, the fair target is 1/N,
        so the slow-group threshold is rho/N. For the common two-client case,
        rho=0.6 gives theta=0.3, matching the previous default.
        """
        try:
            rho = float(self.participation_fairness_rho)
        except Exception:
            rho = 0.0

        if rho <= 0.0:
            return float(self.min_rpi_ratio)

        try:
            n = len(self._known_client_identities())
        except Exception:
            n = 0

        if n <= 0:
            try:
                n = int(self.min_available_nodes)
            except Exception:
                n = 1
        n = max(1, int(n))

        theta = rho / float(n)
        return float(min(max(theta, 0.0), 0.999))
    
    """FedAvg strategy that logs unified client metrics (train + evaluate)."""

    def __init__(
        self,
        fraction_train: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_available_nodes: int = 1,
        log_dir: str = "logs_fedavg",
        model_name: str | None = None,
        latte: bool = False,
        model_info: dict | None = None,
        device_lookup: dict | None = None,
        ROOT: str = "/home/wwjang/EEFL/E2FL",
        device_profiles_path: str | None = None,
        tokens_per_example: float = 256.0,
        enable_rpi_only_rounds: bool = True,
        use_token_ratio: bool = True,
        enable_deadline_guard: bool = False,
        participation_guard_mode: str = "slow-group",
        participation_fairness_rho: float = 0.3,
        **kwargs,
    ):
        # Initialize base FedAvg with only the arguments it expects
        super().__init__(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            min_available_nodes=min_available_nodes,
            **kwargs,
        )

        # For 1J3R sanity runs, require all available clients to reply before
        # aggregation. Otherwise Flower may aggregate with only min_train_nodes
        # replies even though configure_train sampled min_available_nodes clients.
        self.min_train_nodes = max(self.min_train_nodes, self.min_available_nodes)
        self.min_evaluate_nodes = max(self.min_evaluate_nodes, self.min_available_nodes)

        # Flower can still call aggregate_* with partial replies when some sampled
        # clients do not return before the framework-level collection boundary.
        # For sanity experiments, fail fast instead of silently aggregating partial
        # 1J3R rounds and producing misleading CSV rows. Apply the same rule to
        # evaluation so the next training round does not proceed after partial
        # client participation.
        self.require_full_train_replies = True
        self.require_full_evaluate_replies = True

        self.device_lookup = device_lookup if device_lookup is not None else {1: "RPi5", 2: "jetson_orin_nx", 3: "jetson_orin_nano_super"}
        self.log_dir = os.path.join(ROOT, log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        self.model_info = model_info if model_info is not None else {}

        # Profiling-based device performance model (optional)
        self.device_profiles: dict[str, dict] = {}
        self.tokens_per_example: float = float(tokens_per_example)

        # Behavior toggles (can be controlled via run_config / toml)
        self.enable_rpi_only_rounds: bool = bool(enable_rpi_only_rounds)
        self.use_token_ratio: bool = bool(use_token_ratio)
        self.enable_deadline_guard: bool = bool(enable_deadline_guard)

        self.participation_guard_mode = str(participation_guard_mode or "slow-group")
        self.participation_fairness_rho = float(participation_fairness_rho)
        self.protected_client_ids: set[str] = set()

        if device_profiles_path is not None:
            try:
                if os.path.exists(device_profiles_path):
                    with open(device_profiles_path, "r") as pf:
                        self.device_profiles = json.load(pf)
                    log(INFO, "[FedAvgLoRA] Loaded device profiles from %s", device_profiles_path)
                else:
                    log(WARNING, "[FedAvgLoRA] device_profiles_path not found: %s", device_profiles_path)
            except Exception as e:
                log(WARNING, "[FedAvgLoRA] Failed to load device profiles from %s: %s", device_profiles_path, e)

        # === Strategy hyperparams ===
        # Normal rounds use a shorter RPi workload and expand Jetson primarily
        # through the sample cap. Correction rounds repay the RPi token deficit
        # with a bounded RPi-only workload.
        self.window_W = 5
        self.min_rpi_ratio = 0.15
        self.rpi_normal_quota = 256
        self.rpi_min_quota = self.rpi_normal_quota  # backward-compatible alias
        self.rpi_correction_max_quota = 384
        self.rpi_sample_step = 32
        self.max_correction_time_s = 3600.0
        # Correction aims slightly above theta to absorb token-per-sample
        # estimation error and avoid consecutive correction rounds.
        self.correction_target_margin = 0.005
        # A per-client correction round must be followed by one normal round.
        self.last_per_client_correction_round = -1

        self.jetson_base_quota = 64
        self.jetson_max_quota = 4096
        self.jetson_sample_step = 64
        self.jetson_target_margin = 0.95
        # Hierarchical S+H workload control.
        # S is selected first as the coarse knob, then H fills residual slack.
        self.jetson_base_steps = 1
        self.jetson_max_steps = 4096
        self.jetson_step_increment = 1

        # Keep epochs conservative. Sample cap is the primary workload knob.
        self.jetson_base_epochs = 1
        self.jetson_epochs_max = 1

        # Round 1--2 are calibration rounds and do not trigger correction.
        self.policy_warmup_rounds = 2
        self.rpi_only_cooldown = 0
        self.cooldown_remaining = 0

        # === Strategy state ===
        # round별 device별 stats 저장: { round: {"RPi5": {...}, "jetson_orin_nx": {...}, "jetson_orin_nano_super": {...}} }
        self.round_stats = {}
        self.round_configs = {}
        self.expected_train_replies_by_round = {}
        self.expected_evaluate_replies_by_round = {}
        self.global_arrays_before_round = {}
        # ratio 계산용 sliding window (round 번호 순서대로 append)
        self.window_history = []       # 각 원소: {"RPi5": value, "jetson_orin_nx": value, "jetson_orin_nano_super": value}
        # node_id -> device_name mapping used for slow-node preservation rounds
        self.cid_to_device = {}
        self.cid_to_device_uid = {}
        self.slowest_device_name: str | None = None

        # Per-device beta (sec per token) based on loop_time
        self.beta_sec_per_token: dict[str, float] = {}
        # Per-device additional time scaling factor
        self.time_scale: dict[str, float] = {}
        # Per-client observed upload throughput, estimated from previous replies.
        # Keyed by the same identity key as training-time prediction, e.g., "RPi5:rpi5-4".
        self.upload_rate_bps: dict[str, float] = {}

        # Beta estimation parameters
        self.min_beta_round: int = 1        # start estimating from round >= 1
        self.max_beta_history: int = 5      # use up to 5 recent rounds

        # Calibration (time_scale) parameters
        self.calib_alpha: float = 0.3       # EMA weight
        self.calib_clip: tuple[float, float] = (0.5, 2.0)  # clamp ratio

        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.csv_path = os.path.join(self.log_dir, f"fedavg_{timestamp}.csv")
        print(f"=== DEBUG CSV write at {self.csv_path} ===")

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "round", "phase", "cid", "device_name", "device_uid",
                "num_examples", "input_tokens", "supervised_tokens", "loss", "accuracy",
                "comp_time", "init_time", "loop_time",
                "phase_sent", "phase_recv",
                "comm_time", "client_upload_elapsed_time", "comm_sent", "comm_recv",
                "energy", "update_norm",
                "batch_size", "local_epochs", "local_steps",
                "max_samples", "seq_len", "lora_rank", "lora_modules",
                "round_mode", "rpi_max_samples", "rpi_local_epochs",
                "jetson_max_samples", "jetson_local_epochs",
                "upload_throttle_mbps", "upload_throttle_payload_bytes", "upload_throttle_sleep_time",
                "pred_train_time_self", "pred_comm_time_self", "pred_total_time_self",
                "actual_train_time_self", "actual_upload_time_self", "actual_total_time_self",
                "est_beta_self", "est_alpha_self", "est_upload_rate_mbps_self",
                "assigned_workload_proxy",
                "pred_train_time_RPi5", "pred_train_time_jetson_orin_nx",
            ])
            
    '''
    def update_train_config(self, server_round: int, train_config: ConfigRecord) -> ConfigRecord:
        """Round-level policy hook.

        Single source of truth: this function mutates/returns train_config.
        For now, it only ensures default fields exist and can be extended later
        to implement SLO-aware workload allocation.
        """
        # Ensure keys exist (safe defaults)
        if "batch_size" not in train_config:
            train_config["batch_size"] = 32
        if "local_epochs" not in train_config:
            train_config["local_epochs"] = 1
        if "max_samples_per_round" not in train_config:
            train_config["max_samples_per_round"] = ""
        if "max_seq_len" not in train_config:
            train_config["max_seq_len"] = ""
        if "lora_rank" not in train_config:
            train_config["lora_rank"] = ""
        if "lora_modules" not in train_config:
            train_config["lora_modules"] = ""

        # Example placeholder for later: you can change values per round here.
        # (Keep it as no-op for now.)

        # From round 2 onward, refine workload assignment at the individual
        # client level using uid-specific keys. Round 1 remains warm-up.
        self._apply_uid_specific_workloads(server_round, train_config)
        return train_config
    '''


    def _refresh_workload_policy_from_config(self, train_config: ConfigRecord) -> None:
        """Refresh policy knobs supplied through pyproject/run_config."""
        def _get(names, current, caster):
            value = current
            for name in names:
                candidate = train_config.get(name)
                if candidate not in ("", None):
                    value = candidate
                    break
            try:
                return caster(value)
            except Exception:
                return current
    
        self.window_W = max(
            1,
            _get(("window-W", "window_W", "window-w"), self.window_W, int),
        )
        self.participation_fairness_rho = max(
            0.0,
            _get(
                ("participation-fairness-rho", "participation_fairness_rho"),
                self.participation_fairness_rho,
                float,
            ),
        )
        self.rpi_normal_quota = max(
            1,
            _get(
                ("rpi-normal-quota", "rpi_normal_quota", "rpi-min-quota", "rpi_min_quota"),
                self.rpi_normal_quota,
                int,
            ),
        )
        self.rpi_min_quota = self.rpi_normal_quota
        self.rpi_correction_max_quota = max(
            self.rpi_normal_quota,
            _get(
                ("rpi-correction-max-quota", "rpi_correction_max_quota"),
                self.rpi_correction_max_quota,
                int,
            ),
        )
        self.rpi_sample_step = max(
            1,
            _get(("rpi-sample-step", "rpi_sample_step"), self.rpi_sample_step, int),
        )
        self.max_correction_time_s = max(
            1.0,
            _get(
                ("max-correction-time-s", "max_correction_time_s"),
                self.max_correction_time_s,
                float,
            ),
        )
        self.jetson_base_quota = max(
            1,
            _get(
                ("jetson-base-quota", "jetson_base_quota"),
                self.jetson_base_quota,
                int,
            ),
        )
        self.jetson_max_quota = max(
            self.jetson_base_quota,
            _get(
                ("jetson-max-quota", "jetson_max_quota"),
                self.jetson_max_quota,
                int,
            ),
        )
        self.jetson_sample_step = max(
            1,
            _get(
                ("jetson-sample-step", "jetson_sample_step"),
                self.jetson_sample_step,
                int,
            ),
        )
        self.jetson_target_margin = min(
            1.0,
            max(
                0.1,
                _get(
                    ("jetson-target-margin", "jetson_target_margin"),
                    self.jetson_target_margin,
                    float,
                ),
            ),
        )
        self.policy_warmup_rounds = max(
            1,
            _get(
                ("policy-warmup-rounds", "policy_warmup_rounds"),
                self.policy_warmup_rounds,
                int,
            ),
        )
        self.jetson_base_steps = max(
            1,
            _get(
                ("jetson-base-steps", "jetson_base_steps"),
                self.jetson_base_steps,
                int,
            ),
        )

        self.jetson_max_steps = max(
            self.jetson_base_steps,
            _get(
                ("jetson-max-steps", "jetson_max_steps"),
                self.jetson_max_steps,
                int,
            ),
        )

        self.jetson_step_increment = max(
            1,
            _get(
                ("jetson-step-increment", "jetson_step_increment"),
                self.jetson_step_increment,
                int,
            ),
        )
    
    def _recent_group_token_totals_for_correction(
        self,
        slow_device_name: str,
    ) -> tuple[float, float]:
        """Return slow-group and total tokens from the last W-1 rounds."""
        history = self.window_history[-max(0, int(self.window_W) - 1):]
        slow_tokens = 0.0
        total_tokens = 0.0
    
        for entry in history:
            if not isinstance(entry, dict):
                continue
            for device_name, value in entry.items():
                try:
                    tokens = max(0.0, float(value or 0.0))
                except Exception:
                    tokens = 0.0
                total_tokens += tokens
                if device_name == slow_device_name:
                    slow_tokens += tokens
    
        return slow_tokens, total_tokens
    
    def _required_slow_group_tokens_for_target(
        self,
        slow_device_name: str,
    ) -> tuple[float, float, float]:
        """Return required correction tokens and current projected share."""
        theta = self._effective_slow_group_threshold()
        slow_tokens, total_tokens = self._recent_group_token_totals_for_correction(
            slow_device_name
        )
    
        current_share = (
            slow_tokens / total_tokens
            if total_tokens > 0.0
            else 0.5
        )
    
        if total_tokens <= 0.0 or theta <= 0.0 or current_share >= theta:
            return 0.0, theta, current_share
    
        # (slow + x) / (total + x) >= theta
        required = (
            theta * total_tokens - slow_tokens
        ) / max(1.0 - theta, 1e-9)
    
        return max(0.0, required), theta, current_share
    
    def _select_rpi_correction_quota(
        self,
        required_group_tokens: float,
        train_config: ConfigRecord,
    ) -> tuple[int, float]:
        """Convert token debt into a bounded per-RPi sample quota."""
        count = self._recent_device_count("RPi5")
        tokens_per_sample = self._estimate_tokens_per_sample_for_device(
            "RPi5",
            train_config,
        )
        denom = max(
            float(count) * float(tokens_per_sample),
            1e-9,
        )
    
        requested = int(
            math.ceil(
                max(0.0, required_group_tokens) / denom
            )
        )
        requested = max(
            self.rpi_normal_quota,
            requested,
        )
    
        step = max(1, int(self.rpi_sample_step))
        requested = int(
            math.ceil(requested / step) * step
        )
    
        max_quota = max(
            self.rpi_normal_quota,
            int(self.rpi_correction_max_quota),
        )
    
        feasible_max = self.rpi_normal_quota
        for quota in range(
            self.rpi_normal_quota,
            max_quota + 1,
            step,
        ):
            candidate_cfg = copy.deepcopy(train_config)
            candidate_cfg["max_samples_per_round_RPi5"] = int(quota)
            candidate_cfg["local_epochs_RPi5"] = 1
            predicted = self._predict_train_time(
                "RPi5",
                float(quota),
                candidate_cfg,
            )
    
            if predicted <= 0.0 or predicted <= self.max_correction_time_s:
                feasible_max = quota
            else:
                break
    
        selected = min(
            requested,
            feasible_max,
            max_quota,
        )
    
        selected_cfg = copy.deepcopy(train_config)
        selected_cfg["max_samples_per_round_RPi5"] = int(selected)
        selected_cfg["local_epochs_RPi5"] = 1
        predicted_time = self._predict_train_time(
            "RPi5",
            float(selected),
            selected_cfg,
        )
    
        return int(selected), float(predicted_time or 0.0)
    
    def _select_jetson_workload_for_window(
        self,
        target_rpi_time: float,
        train_config: ConfigRecord,
    ) -> tuple[int, int, float]:
        """Select S first, then use H to fill residual slack."""

        fallback_samples = int(self.jetson_base_quota)
        fallback_steps = 0

        if target_rpi_time <= 0.0:
            return (
                fallback_samples,
                fallback_steps,
                0.0,
            )

        target_limit = (
            float(target_rpi_time)
            * float(self.jetson_target_margin)
        )

        best_samples = fallback_samples
        best_steps = 0
        best_time = 0.0
        feasible_found = False

        sample_step = max(
            1,
            int(self.jetson_sample_step),
        )

        best_feasible_samples = fallback_samples
        best_feasible_time = 0.0

        first_infeasible_samples = None

        # Stage 1: coarse search over S.
        for samples in range(
            int(self.jetson_base_quota),
            int(self.jetson_max_quota) + 1,
            sample_step,
        ):
            candidate_cfg = copy.deepcopy(
                train_config
            )

            candidate_cfg[
                "max_samples_per_round_jetson_orin_nx"
            ] = int(samples)

            candidate_cfg[
                "local_epochs_jetson_orin_nx"
            ] = 1

            # H disabled during coarse S search.
            candidate_cfg[
                "local_steps_jetson_orin_nx"
            ] = 0

            predicted = self._predict_train_time(
                "jetson_orin_nx",
                float(samples),
                candidate_cfg,
            )

            if predicted <= 0.0:
                continue

            if predicted <= target_limit:
                feasible_found = True
                best_feasible_samples = int(samples)
                best_feasible_time = float(predicted)
            else:
                first_infeasible_samples = int(samples)
                break

        if not feasible_found:
            return (
                fallback_samples,
                0,
                0.0,
            )

        # S-only candidate already fills the window, or no larger candidate exists.
        if first_infeasible_samples is None:
            return (
                best_feasible_samples,
                0,
                best_feasible_time,
            )

        # Stage 2: fine search over H using selected S.
        candidate_samples = int(
            first_infeasible_samples
        )

        tokens_per_sample = (
            self._estimate_tokens_per_sample_for_device(
                "jetson_orin_nx",
                train_config,
            )
        )

        tokens_per_step = (
            self._estimate_tokens_per_step_for_device(
                "jetson_orin_nx",
                train_config,
            )
        )

        candidate_tokens = (
            float(candidate_samples)
            * float(tokens_per_sample)
        )

        previous_tokens = (
            float(best_feasible_samples)
            * float(tokens_per_sample)
        )

        min_steps = max(
            1,
            int(
                math.ceil(
                    previous_tokens
                    / max(tokens_per_step, 1e-9)
                )
            ),
        )

        full_steps = max(
            min_steps,
            int(
                math.ceil(
                    candidate_tokens
                    / max(tokens_per_step, 1e-9)
                )
            ),
        )

        max_steps = min(
            int(self.jetson_max_steps),
            full_steps,
        )
        if max_steps < min_steps:
            log(
                WARNING,
                "[FedAvgLoRA] Empty H search range: "
                "S_feasible=%d S_candidate=%d "
                "min_steps=%d max_steps=%d",
                int(best_feasible_samples),
                int(candidate_samples),
                int(min_steps),
                int(max_steps),
            )

            return (
                best_feasible_samples,
                0,
                best_feasible_time,
            )
        best_samples = best_feasible_samples
        best_steps = 0
        best_time = best_feasible_time

        for steps in range(
            min_steps,
            max_steps + 1,
            int(self.jetson_step_increment),
        ):
            candidate_cfg = copy.deepcopy(
                train_config
            )

            candidate_cfg[
                "max_samples_per_round_jetson_orin_nx"
            ] = int(candidate_samples)

            candidate_cfg[
                "local_epochs_jetson_orin_nx"
            ] = 1

            candidate_cfg[
                "local_steps_jetson_orin_nx"
            ] = int(steps)

            predicted = self._predict_train_time(
                "jetson_orin_nx",
                float(candidate_samples),
                candidate_cfg,
            )

            if predicted <= 0.0:
                continue

            if predicted <= target_limit:
                best_samples = int(
                    candidate_samples
                )
                best_steps = int(steps)
                best_time = float(predicted)
            else:
                break
        log(
            INFO,
            "[FedAvgLoRA] S+H search: "
            "S_feasible=%d S_candidate=%d "
            "H_range=[%d,%d] target=%.1fs",
            int(best_feasible_samples),
            int(candidate_samples),
            int(min_steps),
            int(max_steps),
            float(target_limit),
        )
        return (
            best_samples,
            best_steps,
            best_time,
        )
    

    def _known_client_device_map(self) -> dict[str, str]:
        """Return the most recently observed device type for each client UID."""
        mapping: dict[str, tuple[int, str]] = {}

        for round_key, stats in self.round_stats.items():
            try:
                round_num = int(round_key)
            except Exception:
                round_num = 0

            clients = stats.get("clients", {}) if isinstance(stats, dict) else {}
            if not isinstance(clients, dict):
                continue

            for client_key, entry in clients.items():
                if not isinstance(entry, dict):
                    continue

                uid = str(
                    entry.get("device_uid")
                    or entry.get("cid")
                    or client_key
                )
                device_name = str(entry.get("device_name") or "")
                if not uid or not device_name:
                    continue

                previous = mapping.get(uid)
                if previous is None or round_num >= previous[0]:
                    mapping[uid] = (round_num, device_name)

        return {
            uid: device_name
            for uid, (_, device_name) in mapping.items()
        }


    def _client_correction_history_totals(
        self,
        ) -> dict[str, float]:
        """Return per-client token totals from the W-1 rounds that remain
        after the upcoming correction round enters the sliding window.
        """
        totals: dict[str, float] = {}

        if not self.round_stats:
            return totals

        rounds = []

        for round_key in self.round_stats.keys():
            try:
                rounds.append(int(round_key))
            except Exception:
                continue

        if not rounds:
            return totals

        keep_count = max(
            0,
            int(self.window_W) - 1,
        )

        selected_rounds = sorted(rounds)[-keep_count:] if keep_count > 0 else []

        for round_number in selected_rounds:
            stats = self.round_stats.get(
                round_number,
                {},
            )

            clients = (
                stats.get("clients", {})
                if isinstance(stats, dict)
                else {}
            )

            if not isinstance(clients, dict):
                continue

            for client_key, entry in clients.items():
                if not isinstance(entry, dict):
                    continue

                uid = str(
                    entry.get("device_uid")
                    or entry.get("cid")
                    or client_key
                )

                if not uid:
                    continue

                try:
                    tokens = max(
                        0.0,
                        float(
                            entry.get("tokens", 0.0)
                            or 0.0
                        ),
                    )
                except Exception:
                    tokens = 0.0

                totals[uid] = (
                    totals.get(uid, 0.0)
                    + tokens
                )

        return totals
    
    def _solve_per_client_correction_debts(
        self,
        train_config: ConfigRecord,
    ) -> tuple[dict[str, float], float, dict[str, float]]:
        """Plan a capability-calibrated per-client correction round.

        The configured participation_fairness_rho is treated as an upper bound,
        not as an always-enforced fixed target.

        After warm-up, LoRACast uses its per-client runtime estimator and the
        correction time/sample limits to estimate the largest workload each RPi
        can feasibly execute in one correction round. It then derives a feasible
        participation threshold from the minimum projected post-correction share.

        Consequently, the same rule can yield a relatively high threshold in a
        1-Jetson/1-RPi deployment and a lower, realizable threshold when several
        slower clients share the system with one much faster client.
        """
        totals = self._client_correction_history_totals()
        device_map = self._known_client_device_map()
        known_ids = set(device_map.keys()) | set(totals.keys())

        if not known_ids:
            return {}, 0.0, {}

        rpi_ids = {
            uid
            for uid in known_ids
            if device_map.get(uid) == "RPi5"
        }

        if not rpi_ids:
            return {}, 0.0, {}

        n_clients = max(
            1,
            len(known_ids),
        )

        total_tokens = float(
            sum(
                max(
                    0.0,
                    float(totals.get(uid, 0.0)),
                )
                for uid in known_ids
            )
        )

        shares = {
            uid: (
                max(
                    0.0,
                    float(totals.get(uid, 0.0)),
                )
                / total_tokens
                if total_tokens > 0.0
                else 0.0
            )
            for uid in known_ids
        }

        if total_tokens <= 0.0:
            return {}, 0.0, shares

        # Configured rho is the maximum normalized participation target.
        #
        # For example:
        #   rho_max = 0.6, N = 2 -> theta_cap = 0.30
        #   rho_max = 0.6, N = 4 -> theta_cap = 0.15
        #
        # The effective target may be lower when the system cannot feasibly
        # reach this cap within one bounded correction round.
        rho_max = max(
            0.0,
            float(self.participation_fairness_rho),
        )

        theta_cap = min(
            rho_max / float(n_clients),
            0.999,
        )

        if theta_cap <= 0.0:
            return {}, 0.0, shares

        # Safety factor absorbs runtime-prediction and tokens/sample error.
        #
        # It can optionally be supplied through the run config:
        #
        # participation-feasibility-safety = 0.95
        #
        try:
            feasibility_safety = float(
                train_config.get(
                    "participation-feasibility-safety",
                    train_config.get(
                        "participation_feasibility_safety",
                        0.95,
                    ),
                )
                or 0.95
            )
        except Exception:
            feasibility_safety = 0.95

        feasibility_safety = min(
            max(feasibility_safety, 0.50),
            1.00,
        )

        tokens_per_sample = max(
            float(
                self._estimate_tokens_per_sample_for_device(
                    "RPi5",
                    train_config,
                )
            ),
            1e-9,
        )

        # Estimate the largest feasible correction workload for every RPi.
        #
        # Passing a very large token request makes the existing selector return
        # the maximum quota allowed jointly by:
        #
        #   1. rpi_correction_max_quota
        #   2. max_correction_time_s
        #   3. the warm-up/runtime prediction model
        #
        # Therefore, no second feasibility model is introduced here.
        feasible_quotas: dict[str, int] = {}
        feasible_additions: dict[str, float] = {}
        feasible_times: dict[str, float] = {}

        for uid in sorted(rpi_ids):
            try:
                max_quota, predicted_time = (
                    self._select_rpi_correction_quota_for_identity(
                        uid,
                        1.0e18,
                        train_config,
                    )
                )
            except (RuntimeError, OverflowError, ValueError) as exc:
                log(
                    WARNING,
                    "[FedAvgLoRA] Failed to derive maximum feasible "
                    "correction quota for %s: %s. "
                    "Falling back to the normal RPi quota.",
                    uid,
                    exc,
                )

                max_quota = int(self.rpi_normal_quota)

                fallback_cfg = copy.deepcopy(train_config)
                suffix = self._safe_config_suffix(uid)

                fallback_cfg[
                    f"max_samples_per_round_uid_{suffix}"
                ] = int(max_quota)
                fallback_cfg[
                    f"local_epochs_uid_{suffix}"
                ] = 1

                predicted_time = self._predict_train_time(
                    f"RPi5:{uid}",
                    float(max_quota),
                    fallback_cfg,
                )

                if predicted_time <= 0.0:
                    predicted_time = self._predict_train_time(
                        "RPi5",
                        float(max_quota),
                        fallback_cfg,
                    )

            feasible_quotas[uid] = int(max_quota)
            feasible_times[uid] = float(
                predicted_time or 0.0
            )
            feasible_additions[uid] = (
                float(max_quota)
                * float(tokens_per_sample)
            )

        maximum_added_tokens = float(
            sum(feasible_additions.values())
        )

        feasible_denominator = (
            total_tokens
            + maximum_added_tokens
        )

        if feasible_denominator <= 0.0:
            return {}, 0.0, shares

        feasible_post_shares = {
            uid: (
                max(
                    0.0,
                    float(totals.get(uid, 0.0)),
                )
                + float(feasible_additions.get(uid, 0.0))
            )
            / feasible_denominator
            for uid in rpi_ids
        }

        minimum_feasible_share = min(
            feasible_post_shares.values(),
            default=0.0,
        )

        # The effective threshold is the smaller of:
        #
        #   1. the user's normalized fairness preference, rho_max / N
        #   2. the share that the slowest RPi can realistically reach within
        #      one bounded correction round
        #
        # A safety factor keeps the target slightly inside the feasible region.
        theta = min(
            theta_cap,
            feasibility_safety * minimum_feasible_share,
        )

        theta = min(
            max(theta, 0.0),
            0.999,
        )

        effective_rho = min(
            rho_max,
            float(n_clients) * theta,
        )

        log(
            INFO,
            "[FedAvgLoRA] capability-calibrated participation: "
            "rho_max=%.4f rho_eff=%.4f N=%d "
            "theta_cap=%.4f theta_eff=%.4f "
            "feasible_min=%.4f safety=%.3f "
            "max_time=%.1fs details=%s",
            float(rho_max),
            float(effective_rho),
            int(n_clients),
            float(theta_cap),
            float(theta),
            float(minimum_feasible_share),
            float(feasibility_safety),
            float(self.max_correction_time_s),
            " | ".join(
                (
                    f"{uid}:share={shares.get(uid, 0.0):.4f},"
                    f"qmax={feasible_post_shares.get(uid, 0.0):.4f},"
                    f"quota={feasible_quotas.get(uid, 0)},"
                    f"pred={feasible_times.get(uid, 0.0):.1f}s"
                )
                for uid in sorted(rpi_ids)
            ),
        )

        if theta <= 0.0:
            return {}, theta, shares

        underrepresented = {
            uid
            for uid in rpi_ids
            if shares.get(uid, 0.0) < theta
        }

        if not underrepresented:
            return {}, theta, shares

        # Aim slightly above the trigger threshold, but never above the
        # estimated feasible post-correction floor.
        correction_target = min(
            theta + float(self.correction_target_margin),
            minimum_feasible_share,
            theta_cap,
            0.999,
        )

        # Every RPi participates in a correction round with at least its
        # normal-round workload. Solving all additions jointly accounts for
        # the common denominator growth caused by the other RPi clients.
        normal_added_tokens = {
            uid: (
                float(self.rpi_normal_quota)
                * float(tokens_per_sample)
            )
            for uid in rpi_ids
        }

        additions = dict(
            normal_added_tokens
        )

        # Joint fixed-point solver:
        #
        #   (history_i + addition_i)
        #   -------------------------  >= correction_target
        #    history_total + sum(additions)
        #
        for _ in range(200):
            denominator_after = (
                total_tokens
                + float(sum(additions.values()))
            )

            updated = {
                uid: max(
                    normal_added_tokens[uid],
                    (
                        correction_target
                        * denominator_after
                        - max(
                            0.0,
                            float(totals.get(uid, 0.0)),
                        )
                    ),
                )
                for uid in rpi_ids
            }

            delta = max(
                abs(
                    updated[uid]
                    - additions[uid]
                )
                for uid in rpi_ids
            )

            additions = updated

            if delta <= 1e-6:
                break

        denominator_after = (
            total_tokens
            + float(sum(additions.values()))
        )

        predicted_shares = {
            uid: (
                max(
                    0.0,
                    float(totals.get(uid, 0.0)),
                )
                + float(additions.get(uid, 0.0))
            )
            / denominator_after
            for uid in rpi_ids
        }

        log(
            INFO,
            "[FedAvgLoRA] planned post-correction shares: "
            "theta_eff=%.4f target=%.4f %s",
            float(theta),
            float(correction_target),
            " | ".join(
                f"{uid}={predicted_shares[uid]:.4f}"
                for uid in sorted(predicted_shares)
            ),
        )

        return additions, theta, shares

    def _select_rpi_correction_quota_for_identity(
        self,
        device_uid: str,
        required_tokens: float,
        train_config: ConfigRecord,
    ) -> tuple[int, float]:
        """Convert one RPi client's token debt into a bounded sample quota."""
        tokens_per_sample = self._estimate_tokens_per_sample_for_device(
            "RPi5",
            train_config,
        )
        tokens_per_sample = max(float(tokens_per_sample), 1e-9)

        requested = int(
            math.ceil(max(0.0, required_tokens) / tokens_per_sample)
        )
        requested = max(self.rpi_normal_quota, requested)

        step = max(1, int(self.rpi_sample_step))
        requested = int(math.ceil(requested / step) * step)

        max_quota = max(
            int(self.rpi_normal_quota),
            int(self.rpi_correction_max_quota),
        )

        suffix = self._safe_config_suffix(device_uid)
        identity_key = f"RPi5:{device_uid}"

        feasible_max = 0
        predicted_at_feasible = 0.0

        for quota in range(
            int(self.rpi_normal_quota),
            max_quota + 1,
            step,
        ):
            candidate_cfg = copy.deepcopy(train_config)
            candidate_cfg[f"max_samples_per_round_uid_{suffix}"] = int(quota)
            candidate_cfg[f"local_epochs_uid_{suffix}"] = 1
            candidate_cfg["max_samples_per_round_RPi5"] = int(quota)
            candidate_cfg["local_epochs_RPi5"] = 1

            predicted = self._predict_train_time(
                identity_key,
                float(quota),
                candidate_cfg,
            )
            if predicted <= 0.0:
                predicted = self._predict_train_time(
                    "RPi5",
                    float(quota),
                    candidate_cfg,
                )

            # Unknown runtime must not be treated as deadline-feasible.
            if predicted <= 0.0:
                break
            if predicted <= self.max_correction_time_s:
                feasible_max = int(quota)
                predicted_at_feasible = float(predicted)
            else:
                break
        if feasible_max <= 0:
            raise RuntimeError(
                f"No correction workload has a valid runtime prediction "
                f"for RPi client {device_uid}."
            )
        selected = min(requested, feasible_max, max_quota)

        selected_cfg = copy.deepcopy(train_config)
        selected_cfg[f"max_samples_per_round_uid_{suffix}"] = int(selected)
        selected_cfg[f"local_epochs_uid_{suffix}"] = 1
        selected_cfg["max_samples_per_round_RPi5"] = int(selected)
        selected_cfg["local_epochs_RPi5"] = 1

        predicted_time = self._predict_train_time(
            identity_key,
            float(selected),
            selected_cfg,
        )
        if predicted_time <= 0.0:
            predicted_time = self._predict_train_time(
                "RPi5",
                float(selected),
                selected_cfg,
            )
        if predicted_time <= 0.0:
            predicted_time = predicted_at_feasible

        return int(selected), float(predicted_time or 0.0)


    def update_train_config(
        self,
        server_round: int,
        train_config: ConfigRecord,
    ) -> ConfigRecord:
        """Apply normal allocation and bounded participation correction.

        Normal rounds shorten the RPi baseline and expand Jetson's workload
        toward the predicted RPi completion window.

        In per-client mode, every known client has target theta=rho/N.
        If any RPi falls below theta, all known RPi clients participate in
        a PROTECTED_ONLY correction round. Their workloads are solved jointly
        against the projected next sliding window, while Jetson is omitted.

        A correction round is followed by one forced normal round.
        """
        if "batch_size" not in train_config:
            train_config["batch_size"] = 32
        if "local_epochs" not in train_config:
            train_config["local_epochs"] = 1
        if "local_steps" not in train_config:
            train_config["local_steps"] = 0
        if "max_samples_per_round" not in train_config:
            train_config["max_samples_per_round"] = ""
        if "max_seq_len" not in train_config:
            train_config["max_seq_len"] = ""
        if "lora_rank" not in train_config:
            train_config["lora_rank"] = ""
        if "lora_modules" not in train_config:
            train_config["lora_modules"] = ""

        self._refresh_workload_policy_from_config(train_config)

        guard_mode = train_config.get(
            "participation-guard-mode",
            train_config.get(
                "participation_guard_mode",
                self.participation_guard_mode,
            ),
        )
        if guard_mode not in ("", None):
            self.participation_guard_mode = str(guard_mode)
        guard_mode = str(self.participation_guard_mode).lower()

        # Remove stale per-UID assignments from the previous round.
        for key in list(train_config.keys()):
            if (
                key.startswith("max_samples_per_round_uid_")
                or key.startswith("local_epochs_uid_")
                or key.startswith("local_steps_uid_")
            ):
                del train_config[key]

        if train_config.get("lora_rank") in ("", None):
            base_rank = (
                train_config.get("lora-r")
                or train_config.get("lora_r")
            )
            if base_rank is None and isinstance(self.model_info, dict):
                base_rank = (
                    self.model_info.get("lora_rank")
                    or self.model_info.get("lora_r")
                    or self.model_info.get("lora-r")
                )
            train_config["lora_rank"] = (
                base_rank if base_rank is not None else ""
            )

        rank = train_config.get("lora_rank", "")
        if rank not in ("", None):
            train_config["lora_rank_RPi5"] = rank
            train_config["lora_rank_jetson_orin_nx"] = rank
            train_config["lora_rank_jetson_orin_nano_super"] = rank

        if (
            not self.enable_rpi_only_rounds
            and not self.use_token_ratio
            and not self.enable_deadline_guard
        ):
            global_max = train_config.get("max_samples_per_round", "")
            global_epochs = int(
                train_config.get("local_epochs", 1) or 1
            )
            train_config["round_mode"] = "BASELINE"
            train_config["max_samples_per_round_RPi5"] = global_max
            train_config["local_epochs_RPi5"] = global_epochs
            train_config["max_samples_per_round_jetson_orin_nx"] = global_max
            train_config["local_epochs_jetson_orin_nx"] = global_epochs
            train_config["max_samples_per_round_jetson_orin_nano_super"] = global_max
            train_config["local_epochs_jetson_orin_nano_super"] = global_epochs
            train_config["local_steps_RPi5"] = 0
            train_config["local_steps_jetson_orin_nx"] = 0
            train_config["local_steps_jetson_orin_nano_super"] = 0

            self.protected_client_ids = set()
            return train_config


        force_normal_after_correction = (
            int(self.last_per_client_correction_round)
            == int(server_round) - 1
        )

        correction_enabled = (
            self.enable_rpi_only_rounds
            and self.use_token_ratio
            and server_round > self.policy_warmup_rounds
            and not force_normal_after_correction
        )

        if force_normal_after_correction:
            log(
                INFO,
                "[FedAvgLoRA][ROUND %s] forcing BOTH "
                "after correction round %s",
                server_round,
                self.last_per_client_correction_round,
            )


        per_client_debts: dict[str, float] = {}
        per_client_theta = 0.0
        per_client_shares: dict[str, float] = {}

        group_required_tokens = 0.0
        group_theta = self._effective_slow_group_threshold()
        group_share = 0.5
        slow_device_name = self._get_slowest_device_name() or "RPi5"

        if correction_enabled and guard_mode == "per-client":
            (
                per_client_debts,
                per_client_theta,
                per_client_shares,
            ) = self._solve_per_client_correction_debts(train_config,)
        elif correction_enabled:
            (
                group_required_tokens,
                group_theta,
                group_share,
            ) = self._required_slow_group_tokens_for_target(
                slow_device_name
            )

        if per_client_debts:
            self.protected_client_ids = set(per_client_debts.keys())
            train_config["round_mode"] = "PROTECTED_ONLY"
            self.last_per_client_correction_round = int(
                server_round
            )

            # Device-level RPi values are fallback only. The actual workloads are
            # written into UID-specific keys below.
            train_config["max_samples_per_round_RPi5"] = int(
                self.rpi_normal_quota
            )
            train_config["local_epochs_RPi5"] = 1
            train_config["max_samples_per_round_jetson_orin_nx"] = 0
            train_config["local_epochs_jetson_orin_nx"] = 0
            train_config["max_samples_per_round_jetson_orin_nano_super"] = 0
            train_config["local_epochs_jetson_orin_nano_super"] = 0
            train_config["local_steps_RPi5"] = 0
            train_config["local_steps_jetson_orin_nx"] = 0
            train_config["local_steps_jetson_orin_nano_super"] = 0

            selected_correction_tokens: dict[str, float] = {}

            tokens_per_sample = max(
                float(
                    self._estimate_tokens_per_sample_for_device(
                        "RPi5",
                        train_config,
                    )
                ),
                1e-9,
            )
            quota_log = []
            for uid, debt_tokens in sorted(per_client_debts.items()):
                quota, predicted_time = (
                    self._select_rpi_correction_quota_for_identity(
                        uid,
                        debt_tokens,
                        train_config,
                    )
                )
                selected_correction_tokens[uid] = (
                    float(quota)
                    * float(tokens_per_sample)
                )
                suffix = self._safe_config_suffix(uid)
                train_config[
                    f"max_samples_per_round_uid_{suffix}"
                ] = int(quota)
                train_config[
                    f"local_epochs_uid_{suffix}"
                ] = 1

                quota_log.append(
                    f"{uid}:share={per_client_shares.get(uid, 0.0):.4f},"
                    f"debt={debt_tokens:.1f},quota={quota},"
                    f"pred={predicted_time:.1f}s"
                )
            history_totals = (
                self._client_correction_history_totals()
            )

            known_ids = (
                set(self._known_client_device_map().keys())
                | set(history_totals.keys())
            )

            history_total_tokens = float(
                sum(
                    max(
                        0.0,
                        float(
                            history_totals.get(uid, 0.0)
                        ),
                    )
                    for uid in known_ids
                )
            )

            selected_total_tokens = float(
                sum(
                    selected_correction_tokens.values()
                )
            )

            post_correction_denominator = (
                history_total_tokens
                + selected_total_tokens
            )

            actual_predicted_shares = {}

            for uid in sorted(per_client_debts):
                numerator = (
                    max(
                        0.0,
                        float(
                            history_totals.get(uid, 0.0)
                        ),
                    )
                    + float(
                        selected_correction_tokens.get(
                            uid,
                            0.0,
                        )
                    )
                )

                actual_predicted_shares[uid] = (
                    numerator
                    / post_correction_denominator
                    if post_correction_denominator > 0.0
                    else 0.0
                )

            minimum_predicted_share = min(
                actual_predicted_shares.values(),
                default=0.0,
            )

            log(
                INFO,
                "[FedAvgLoRA][ROUND %s] ACTUAL_QUOTA_POST_SHARE: "
                "theta=%.4f min=%.4f shares=%s",
                server_round,
                float(per_client_theta),
                float(minimum_predicted_share),
                " | ".join(
                    f"{uid}={share:.4f}"
                    for uid, share in sorted(
                        actual_predicted_shares.items()
                    )
                ),
            )

            if (
                minimum_predicted_share
                + 1e-9
                < float(per_client_theta)
            ):
                log(
                    WARNING,
                    "[FedAvgLoRA][ROUND %s] correction quota "
                    "is insufficient to reach theta: "
                    "predicted_min=%.4f theta=%.4f. "
                    "Quota or correction-time cap is binding.",
                    server_round,
                    float(minimum_predicted_share),
                    float(per_client_theta),
                )
            log(
                INFO,
                "[FedAvgLoRA][ROUND %s] PER_CLIENT_CORRECTION: "
                "theta=%.4f protected=%s max_time=%.1f",
                server_round,
                float(per_client_theta),
                " | ".join(quota_log),
                float(self.max_correction_time_s),
            )

        elif group_required_tokens > 0.0:
            self.protected_client_ids = set()
            rpi_quota, predicted_rpi_time = (
                self._select_rpi_correction_quota(
                    group_required_tokens,
                    train_config,
                )
            )

            train_config["round_mode"] = "RPIONLY"
            train_config["max_samples_per_round_RPi5"] = int(rpi_quota)
            train_config["local_epochs_RPi5"] = 1
            train_config["max_samples_per_round_jetson_orin_nx"] = 0
            train_config["local_epochs_jetson_orin_nx"] = 0
            train_config["max_samples_per_round_jetson_orin_nano_super"] = 0
            train_config["local_epochs_jetson_orin_nano_super"] = 0
            train_config["local_steps_RPi5"] = 0
            train_config["local_steps_jetson_orin_nx"] = 0
            train_config["local_steps_jetson_orin_nano_super"] = 0

            log(
                INFO,
                "[FedAvgLoRA][ROUND %s] GROUP_CORRECTION: "
                "share=%.4f theta=%.4f debt_tokens=%.1f "
                "rpi_quota=%d pred_rpi_time=%.1f max_time=%.1f",
                server_round,
                float(group_share),
                float(group_theta),
                float(group_required_tokens),
                int(rpi_quota),
                float(predicted_rpi_time),
                float(self.max_correction_time_s),
            )

        else:
            self.protected_client_ids = set()
            train_config["round_mode"] = "BOTH"
            train_config["max_samples_per_round_RPi5"] = int(
                self.rpi_normal_quota
            )
            train_config["local_epochs_RPi5"] = 1

            target_rpi_time = self._predict_train_time(
                "RPi5",
                float(self.rpi_normal_quota),
                train_config,
            )
            (
                jetson_quota,
                jetson_steps,
                predicted_jetson_time,
            ) = self._select_jetson_workload_for_window(
                target_rpi_time,
                train_config,
            )

            if server_round == 1 and target_rpi_time <= 0.0:
                jetson_quota = int(self.jetson_base_quota)

            train_config["max_samples_per_round_jetson_orin_nx"] = int(
                jetson_quota
            )
            train_config["local_epochs_jetson_orin_nx"] = 1
            train_config["max_samples_per_round_jetson_orin_nano_super"] = int(
                jetson_quota
            )
            train_config["local_epochs_jetson_orin_nano_super"] = 1

            train_config[
                "local_steps_jetson_orin_nx"
            ] = int(jetson_steps)

            train_config[
                "local_steps_jetson_orin_nano_super"
            ] = int(jetson_steps)

            theta_for_log = (
                per_client_theta
                if guard_mode == "per-client"
                else group_theta
            )
            log(
                INFO,
                "[FedAvgLoRA][ROUND %s] NORMAL: guard=%s "
                "rpi_quota=%d pred_rpi_time=%.1f "
                "jetson_quota=%d jetson_steps=%d "
                "pred_jetson_time=%.1f theta=%.4f",
                server_round,
                guard_mode,
                int(self.rpi_normal_quota),
                float(target_rpi_time or 0.0),
                int(jetson_quota),
                int(jetson_steps),
                float(predicted_jetson_time or 0.0),
                float(theta_for_log),
            )

        self._apply_uid_specific_workloads(
            server_round,
            train_config,
        )
        return train_config

    def configure_train(
            self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
        ) -> Iterable[Message]:
            """Configure the next round of federated training."""
            # Do not configure federated train if fraction_train is 0.
            if self.fraction_train == 0.0:
                return []

            # Apply LoRACast's round-level policy before sampling/logging. Since
            # this method overrides Flower's default configure_train, we must call
            # the policy hook explicitly; otherwise round_mode and device-specific
            # workload keys remain empty/default in the CSV.
            config = self.update_train_config(server_round, copy.deepcopy(config))

            # Sample nodes. For scalability experiments, min_available_nodes is the
            # intended number of concurrently participating clients. Flower's base
            # min_train_nodes can remain at its default (often 2), so we explicitly
            # include min_available_nodes when computing the sample size.
            # `grid.get_node_ids()` can be stale before `sample_nodes` waits for
            # min_available_nodes, so do not clamp sample_size to its current length.
            # Otherwise, a run may wait for 5 clients but still sample only 3.
            available_node_ids = list(grid.get_node_ids())
            num_available_now = len(available_node_ids)
            effective_available = max(num_available_now, self.min_available_nodes)
            num_nodes = int(effective_available * self.fraction_train)
            sample_size = max(num_nodes, self.min_train_nodes, self.min_available_nodes)
            node_ids, num_total = sample_nodes(grid, self.min_available_nodes, sample_size)
            log(
                INFO,
                "configure_train: fraction_train=%.3f min_train_nodes=%s min_available_nodes=%s num_available_now=%s sample_size=%s",
                float(self.fraction_train),
                self.min_train_nodes,
                self.min_available_nodes,
                num_available_now,
                sample_size,
            )
            log(
                INFO,
                "configure_train: Sampled %s nodes (out of %s)",
                len(node_ids),
                num_total,
            )
            # Optionally filter nodes for participation preservation rounds.
            # Historical round_mode value "RPIONLY" is kept for compatibility, but
            # the selected group is the currently slowest observed device group.
            round_mode = config.get("round_mode", "BOTH")
            if self.enable_rpi_only_rounds and round_mode == "PROTECTED_ONLY":
                protected = set(getattr(self, "protected_client_ids", set()))
                filtered_ids = []
                for nid in node_ids:
                    cid_str = str(nid)
                    device_uid = str(self.cid_to_device_uid.get(nid, self.cid_to_device_uid.get(cid_str, cid_str)))
                    if cid_str in protected or device_uid in protected:
                        filtered_ids.append(nid)
                if filtered_ids:
                    node_ids = filtered_ids
                    log(
                        INFO,
                        "[FedAvgLoRA] PROTECTED_ONLY round: keeping %s/%s protected clients: %s",
                        len(filtered_ids),
                        len(node_ids),
                        sorted(str(x) for x in filtered_ids),
                    )
                else:
                    raise RuntimeError(
                        "PROTECTED_ONLY round requested, but none of the protected "
                        "client UIDs matched the sampled Flower node IDs. "
                        f"protected={sorted(self.protected_client_ids)}, "
                        f"sampled={sorted(node_ids)}"
                    )

            if self.enable_rpi_only_rounds and round_mode == "RPIONLY":
                slow_dev = self._get_slowest_device_name()
                filtered_ids = []
                unknown_ids = []
                for nid in node_ids:
                    # Prefer dynamically learned mapping, fall back to static device_lookup
                    dev_name = self.cid_to_device.get(nid)
                    if dev_name is None and self.device_lookup is not None:
                        try:
                            dev_name = self.device_lookup.get(int(nid))
                        except Exception:
                            dev_name = self.device_lookup.get(nid)
                    if dev_name is None:
                        unknown_ids.append(nid)
                    elif slow_dev is not None and dev_name == slow_dev:
                        filtered_ids.append(nid)

                # If the slow device is known and present, keep only that group.
                # If mapping is still incomplete, keep unknowns as a conservative fallback
                # so the round does not accidentally become empty.
                if filtered_ids:
                    node_ids = filtered_ids
                elif unknown_ids:
                    node_ids = unknown_ids
            # Always inject current server round
            config["server-round"] = server_round

            # Track the exact number of train messages sent in this round. This is
            # important for slow-only/rebalancing rounds, where the expected reply
            # count can be smaller than min_available_nodes.
            self.expected_train_replies_by_round[server_round] = len(node_ids)

            # Flower v1.22 calls aggregate_train(server_round, replies), without
            # passing the train ConfigRecord. Keep a round-local snapshot here so
            # aggregate_train can recover the exact config sent to clients.
            self.round_configs[server_round] = copy.deepcopy(config)

            # Construct messages
            record = RecordDict(
                {self.arrayrecord_key: arrays, self.configrecord_key: config}
            )
            return self._construct_messages(record, node_ids, MessageType.TRAIN)
        
    def _check_and_log_replies(
        self, replies: Iterable[Message], is_train: bool, validate: bool = True
        ) -> tuple[list[Message], list[Message]]:
        """Check replies for errors and log them.

        Parameters
        ----------
        replies : Iterable[Message]
            Iterable of reply Messages.
        is_train : bool
            Set to True if the replies are from a training round; False otherwise.
            This impacts logging and validation behavior.
        validate : bool (default: True)
            Whether to validate the reply contents for consistency.

        Returns
        -------
        tuple[list[Message], list[Message]]
            A tuple containing two lists:
            - Messages with valid contents.
            - Messages with errors.
        """
        if not replies:
            return [], []

        # Filter messages that carry content
        valid_replies: list[Message] = []
        error_replies: list[Message] = []
        for msg in replies:
            if msg.has_error():
                error_replies.append(msg)
            else:
                valid_replies.append(msg)

        log(
            INFO,
            "%s: Received %s results and %s failures",
            "aggregate_train" if is_train else "aggregate_evaluate",
            len(valid_replies),
            len(error_replies),
        )

        # Log errors
        for msg in error_replies:
            log(
                INFO,
                "\t> Received error in reply from node %d: %s",
                msg.metadata.src_node_id,
                msg.error.reason,
            )

        # Ensure expected ArrayRecords and MetricRecords are received
        if validate and valid_replies:
            validate_message_reply_consistency(
                replies=[msg.content for msg in valid_replies],
                weighted_by_key=self.weighted_by_key,
                check_arrayrecord=is_train,
            )

        return valid_replies, error_replies

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        print("\n==============================")
        print(f"=== DEBUG aggregate_train (round {server_round}) ===")
        train_config = self.round_configs.get(server_round, ConfigRecord())
        # Round-level training config may change per round.

        print(
            f"[ROUND {server_round}] train_config snapshot: bs={train_config.get('batch_size','')} ep={train_config.get('local_epochs','')} max_samples={train_config.get('max_samples_per_round','')} seq_len={train_config.get('max_seq_len','')}"
        )

        # replies iterator 때문에 리스트화
        replies = list(replies)
        print("Raw replies count:", len(replies))

        # ---------------------------------------------------------
        # 1) Raw replies 내용 덤프 (서버가 실제로 받은 메시지 자체 확인)
        # ---------------------------------------------------------
        for i, msg in enumerate(replies):
            print(f"\n--- DEBUG RAW REPLY [{i}] ---")
            print("has_content:", msg.has_content())
            print("metadata:", msg.metadata)
            print("message_type:", msg.metadata.message_type)

            if msg.has_content():
                print("content keys:", list(msg.content.keys()))

                if "arrays" not in msg.content:
                    print(" !! WARNING: arrays key missing in msg.content")
                else:
                    print("arrays type:", type(msg.content["arrays"]))

                if "metrics" not in msg.content:
                    print(" !! WARNING: metrics key missing in msg.content")
                else:
                    print("metrics type:", type(msg.content["metrics"]))
            else:
                print("NO content available in msg")

        # ---------------------------------------------------------
        # 2) validator 통해 valid / invalid 분리
        # ---------------------------------------------------------
        valid_replies, invalid_replies = self._check_and_log_replies(
            replies, is_train=True
        )

        print("\n=== DEBUG reply validation summary ===")
        print("valid_replies:", len(valid_replies))
        print("invalid_replies:", len(invalid_replies))

        if invalid_replies:
            print("\n--- INVALID REPLIES DETAIL ---")
            for msg in invalid_replies:
                print("metadata:", msg.metadata)
                if msg.has_error():
                    print("reason:", msg.error.reason)

        # ---------------------------------------------------------
        # 3) valid replies 없으면 early return
        # ---------------------------------------------------------
        arrays, metrics = None, None
        if not valid_replies:
            print("!!! aggregate_train: NO VALID REPLIES — skipping aggregation !!!")
            return None, None

        expected_replies = int(
            self.expected_train_replies_by_round.get(
                server_round,
                max(self.min_train_nodes, self.min_available_nodes),
            )
        )
        if self.require_full_train_replies and len(valid_replies) < expected_replies:
            msg = (
                f"aggregate_train received only {len(valid_replies)}/{expected_replies} valid replies "
                f"in round {server_round}. Aborting this sanity run to avoid partial aggregation. "
                "The server-side reply collection window ended before all sampled clients returned. "
                "Increase the Flower strategy.start timeout/collection timeout, or reduce the sanity workload."
            )
            log(WARNING, msg)
            raise RuntimeError(msg)

        # ---------------------------------------------------------
        # 4) 로깅 + 기존 aggregation 실행
        # ---------------------------------------------------------
        print("=== VALID REPLIES FOUND → AGGREGATING ===")
        self._write_metrics("train", train_config, server_round, valid_replies)

        reply_contents = [msg.content for msg in valid_replies]

        # Arrays aggregation
        try:
            arrays = aggregate_arrayrecords(
                reply_contents, self.weighted_by_key
            )
            print("aggregate_arrayrecords SUCCESS")
        except Exception as e:
            print("!!! aggregate_arrayrecords FAILED !!!", e)
            arrays = None

        # Metrics aggregation
        try:
            metrics = self.train_metrics_aggr_fn(
                reply_contents, self.weighted_by_key
            )
            print("train_metrics_aggr_fn SUCCESS")
        except Exception as e:
            print("!!! train_metrics_aggr_fn FAILED !!!", e)
            metrics = None

        return arrays, metrics

    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated evaluation."""
        # Do not configure federated evaluation if fraction_evaluate is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Sample nodes. Match train-side behavior so evaluation sanity logs also
        # cover all connected clients when fraction_evaluate=1.0.
        # `grid.get_node_ids()` can be stale before `sample_nodes` waits for
        # min_available_nodes, so do not clamp sample_size to its current length.
        available_node_ids = list(grid.get_node_ids())
        num_available_now = len(available_node_ids)
        effective_available = max(num_available_now, self.min_available_nodes)
        num_nodes = int(effective_available * self.fraction_evaluate)
        sample_size = max(num_nodes, self.min_evaluate_nodes, self.min_available_nodes)
        node_ids, num_total = sample_nodes(grid, self.min_available_nodes, sample_size)
        log(
            INFO,
            "configure_evaluate: fraction_evaluate=%.3f min_evaluate_nodes=%s min_available_nodes=%s num_available_now=%s sample_size=%s",
            float(self.fraction_evaluate),
            self.min_evaluate_nodes,
            self.min_available_nodes,
            num_available_now,
            sample_size,
        )
        log(
            INFO,
            "configure_evaluate: Sampled %s nodes (out of %s)",
            len(node_ids),
            num_total,
        )

        # Always inject current server round
        config["server-round"] = server_round

        # Track the exact number of evaluate messages sent in this round.
        self.expected_evaluate_replies_by_round[server_round] = len(node_ids)

        # Flower v1.22 calls aggregate_evaluate(server_round, replies), without
        # passing the evaluate ConfigRecord. Store a snapshot for logging.
        self.round_configs[f"eval:{server_round}"] = copy.deepcopy(config)

        # Construct messages
        record = RecordDict(
            {self.arrayrecord_key: arrays, self.configrecord_key: config}
        )
        return self._construct_messages(record, node_ids, MessageType.EVALUATE)
    
    def aggregate_evaluate(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> MetricRecord | None:
        evaluate_config = self.round_configs.get(f"eval:{server_round}", ConfigRecord())
        valid_replies, _ = self._check_and_log_replies(replies, is_train=False)
        metrics = None
        expected_replies = int(
            self.expected_evaluate_replies_by_round.get(
                server_round,
                max(self.min_evaluate_nodes, self.min_available_nodes),
            )
        )
        if self.require_full_evaluate_replies and len(valid_replies) < expected_replies:
            msg = (
                f"aggregate_evaluate received only {len(valid_replies)}/{expected_replies} valid replies "
                f"in round {server_round}. Aborting this sanity run to avoid partial evaluation."
            )
            log(WARNING, msg)
            raise RuntimeError(msg)
        if valid_replies:
            self._write_metrics("evaluate", evaluate_config, server_round, valid_replies)

            reply_contents = [msg.content for msg in valid_replies]
            metrics = self.evaluate_metrics_aggr_fn(
                reply_contents, self.weighted_by_key,
            )
        return metrics

    def _extract_arrays(self, rec):
        """Extracts arrays from various possible ArrayRecord-like objects, including Flower ArrayRecord."""
        if rec is None:
            return None
        # If it's a Flower ArrayRecord, use to_numpy_ndarrays if available
        if isinstance(rec, ArrayRecord):
            try:
                return rec.to_numpy_ndarrays()
            except Exception:
                # Fallback: try extracting .values() and converting to numpy
                try:
                    return [v.numpy() for v in rec.values()]
                except Exception:
                    return None
        # If it's a dict, return as is
        if isinstance(rec, dict):
            return rec
        # If it's a list or tuple, return as is
        if isinstance(rec, (list, tuple)):
            return rec
        # Fallback: try numpy conversion
        try:
            return np.asarray(rec)
        except Exception:
            return None

    def _compute_update_norm(self, server_round: int, client_arrays_rec):
        """해당 라운드에서 client update w_i - w_global 의 L2 norm 계산.

        global_arrays_before_round[round]에 저장된 global 파라미터와
        client reply의 arrays를 비교한다.
        """
        global_rec = self.global_arrays_before_round.get(server_round)
        if global_rec is None or client_arrays_rec is None:
            return ""

        g = self._extract_arrays(global_rec)
        c = self._extract_arrays(client_arrays_rec)

        if g is None or c is None:
            return ""

        sq_sum = 0.0
        try:
            if isinstance(g, dict) and isinstance(c, dict):
                # Dict-style: match by key
                for k, g_val in g.items():
                    if k not in c:
                        continue
                    c_val = c[k]
                    g_arr = np.asarray(g_val)
                    c_arr = np.asarray(c_val)
                    diff = c_arr - g_arr
                    sq_sum += float(np.sum(diff * diff))
            else:
                # Sequence-style: treat as list/tuple/ArrayRecord.arrays
                g_list = list(g)
                c_list = list(c)
                for g_val, c_val in zip(g_list, c_list):
                    g_arr = np.asarray(g_val)
                    c_arr = np.asarray(c_val)
                    diff = c_arr - g_arr
                    sq_sum += float(np.sum(diff * diff))
        except Exception:
            return ""

        if sq_sum <= 0.0:
            return 0.0
        return math.sqrt(sq_sum)

    def _write_metrics(self, phase, train_config, server_round, valid_replies):
        print("=== DEBUG _write_metrics ===")
        print("phase:", phase)
        print("num valid_replies:", len(valid_replies))

        now = time.strftime("%Y-%m-%d_%H-%M-%S")
        # Use wall-clock time here to match client-side timestamps (time.time),
        # assuming NTP keeps server and clients roughly in sync.
        t_end = time.time()
        rows = []

        # --- Initialize round_obs for train phase ---
        round_obs = {} if phase == "train" else None

        for idx, msg in enumerate(valid_replies):
            print(f"\n--- DEBUG msg[{idx}] ---")
            print("has_content:", msg.has_content())

            if not msg.has_content():
                print("  has_content=False → SKIP")
                continue

            content = msg.content
            print("\tcontent keys:", type(content.keys()))
            if "metrics" not in content:
                print("  metrics not found → SKIP")
                continue
                
            metric_record = content["metrics"]     # MetricRecord
            if not hasattr(metric_record, "items"):
                print("  metric_record has no items() → SKIP")
                continue

            m = dict(metric_record.items())        # dict conversion
            input_tokens = m.get("input_tokens", "")
            supervised_tokens = m.get("supervised_tokens", "")
            print("  METRIC KEYS:", list(m.keys()))

            print("metadata:", msg.metadata)
            raw_cid = msg.metadata.src_node_id
            try:
                cid = int(raw_cid)
            except Exception:
                cid = raw_cid
            print("cid extracted:", cid)
            device_uid = self._decode_metric_string(m.get("device_uid_chars", ""))
            if phase == "train":
                loss = m.get("train_loss", "")
                acc = ""
                train_time = m.get("train_time", "")
                train_init_time = m.get("train_init_time", "")
                train_loop_time = m.get("train_loop_time", "")
                e = m.get("train_energy", "")
                # 서버에서 global vs client arrays로 update norm 계산
                arr_key = getattr(self, "arrayrecord_key", "arrays")
                client_arrays_rec = content.get(arr_key, None)
                update_norm = self._compute_update_norm(server_round, client_arrays_rec)
                start_t = m.get("upload_start_time", 0.0)
                comm_t = max(t_end - start_t, 0) if start_t else ""
                client_upload_elapsed_time = m.get("client_upload_elapsed_time", "")
                phase_sent = m.get("train_sent", "")
                phase_recv = m.get("train_recv", "")
                comm_sent = m.get("update_sent", "")
                comm_recv = m.get("update_recv", "")

                # Prefer explicit device_name from metrics if available; fall back to device_code lookup.
                raw_dev_name = self._decode_metric_string(m.get("device_name_chars", ""))
                if raw_dev_name:
                    device_name = str(raw_dev_name)
                else:
                    device_name = self.device_lookup.get(m.get("device_code"), "RPi5")
                if device_uid in ("", None):
                    device_uid = str(cid)
            else:
                loss = m.get("eval_loss", "")
                acc = m.get("eval_accuracy", "")
                t = m.get("eval_time", "")
                e = m.get("eval_energy", "")
                update_norm = m.get("update_norm", "")
                init_t = ""
                loop_t = ""
                start_t = m.get("update_start_time", 0.0)
                comm_t = max(t_end - start_t, 0) if start_t else ""
                client_upload_elapsed_time = ""
                phase_sent = m.get("eval_sent", "")
                phase_recv = m.get("eval_recv", "")
                comm_sent = m.get("upload_sent", "")
                comm_recv = m.get("upload_recv", "")
                raw_dev_name = self._decode_metric_string(m.get("device_name_chars", ""))
                if raw_dev_name:
                    device_name = str(raw_dev_name)
                else:
                    device_name = self.device_lookup.get(m.get("device_code"), "")
                if device_uid in ("", None):
                    device_uid = str(cid)

            # Use sum of init_time and loop_time for train phase, else eval_time
            if phase == "train":
                try:
                    it = float(train_init_time) if train_init_time not in ("", None) else 0.0
                except Exception:
                    it = 0.0
                try:
                    lt = float(train_loop_time) if train_loop_time not in ("", None) else 0.0
                except Exception:
                    lt = 0.0
                comp_time = it + lt
            else:
                comp_time = t

            base_row = [
                now, server_round, phase, cid, device_name, str(device_uid) if phase == "train" else "",
                m.get("num-examples", 0),
                input_tokens, supervised_tokens,
                loss, acc,
                comp_time, train_init_time if phase == "train" else init_t, train_loop_time if phase == "train" else loop_t,
                phase_sent, phase_recv,
                comm_t, client_upload_elapsed_time, comm_sent, comm_recv,
                e, update_norm,
            ]
            # -------------------------------
            # Knob fields (server-side intended)
            # -------------------------------
            if phase == "train":
                # Round-level config (global defaults)
                batch_size = train_config.get("batch_size", "")
                local_epochs = train_config.get("local_epochs", "")
                max_samples = train_config.get("max_samples_per_round", "")
                seq_len = train_config.get("max_seq_len", "")
                # Start from client-reported lora_rank if available
                lora_rank = m.get("lora_rank", None)
                lora_modules = train_config.get("lora_modules", "")

                # Fallback to round-level config only if metric is missing/empty
                if lora_rank in ("", None):
                    cfg_rank = train_config.get("lora_rank")
                    if cfg_rank not in ("", None):
                        lora_rank = cfg_rank

                # Device-specific override for epochs/max_samples/lora_rank so that
                # CSV reflects the *actual* workload each client used. UID-specific
                # keys have priority over device-type keys.
                uid_suffix = self._safe_config_suffix(device_uid) if device_uid not in ("", None) else ""
                uid_local_epochs = train_config.get(f"local_epochs_uid_{uid_suffix}") if uid_suffix else None
                uid_max_samples = train_config.get(f"max_samples_per_round_uid_{uid_suffix}") if uid_suffix else None

                if uid_local_epochs not in ("", None):
                    local_epochs = uid_local_epochs
                if uid_max_samples not in ("", None):
                    max_samples = uid_max_samples

                if device_name == "RPi5":
                    if uid_local_epochs in ("", None):
                        local_epochs = train_config.get("local_epochs_RPi5", local_epochs)
                    if uid_max_samples in ("", None):
                        max_samples = train_config.get("max_samples_per_round_RPi5", max_samples)
                    dev_rank = train_config.get("lora_rank_RPi5")
                    if dev_rank not in ("", None):
                        lora_rank = dev_rank
                elif device_name == "jetson_orin_nx":
                    if uid_local_epochs in ("", None):
                        local_epochs = train_config.get("local_epochs_jetson_orin_nx", local_epochs)
                    if uid_max_samples in ("", None):
                        max_samples = train_config.get("max_samples_per_round_jetson_orin_nx", max_samples)
                    dev_rank = train_config.get("lora_rank_jetson_orin_nx")
                    if dev_rank not in ("", None):
                        lora_rank = dev_rank
                elif device_name == "jetson_orin_nano_super":
                    if uid_local_epochs in ("", None):
                        local_epochs = train_config.get(
                            "local_epochs_jetson_orin_nano_super",
                            train_config.get("local_epochs_jetson_orin_nx", local_epochs),
                        )
                    if uid_max_samples in ("", None):
                        max_samples = train_config.get(
                            "max_samples_per_round_jetson_orin_nano_super",
                            train_config.get("max_samples_per_round_jetson_orin_nx", max_samples),
                        )
                    dev_rank = train_config.get(
                        "lora_rank_jetson_orin_nano_super",
                        train_config.get("lora_rank_jetson_orin_nx"),
                    )
                    if dev_rank not in ("", None):
                        lora_rank = dev_rank

                # Fallback: If lora_rank still empty, try to get from model_info
                if lora_rank in ("", None):
                    try:
                        mi = getattr(self, "model_info", None)
                        if isinstance(mi, dict):
                            # Try per-device first
                            dev_info = mi.get(device_name, {})
                            if isinstance(dev_info, dict):
                                lora_rank = dev_info.get("lora_rank", lora_rank)
                            # If still empty, try global keys
                            if lora_rank in ("", None):
                                for k in ("lora_rank", "lora_r", "lora-r"):
                                    val = mi.get(k, None)
                                    if val not in ("", None):
                                        lora_rank = val
                                        break
                    except Exception:
                        pass

                # Round-level workload knobs for each device type (for reference)
                round_mode = train_config.get("round_mode", "")
                rpi_max_samples = train_config.get("max_samples_per_round_RPi5", "")
                rpi_local_epochs = train_config.get("local_epochs_RPi5", "")
                jetson_max_samples = train_config.get("max_samples_per_round_jetson_orin_nx", "")
                jetson_local_epochs = train_config.get("local_epochs_jetson_orin_nx", "")

                # Optional client-side upload throttling metrics for communication-emulation experiments
                upload_throttle_mbps = m.get("upload_throttle_mbps", 0.0)
                upload_throttle_payload_bytes = m.get("upload_throttle_payload_bytes", 0)
                upload_throttle_sleep_time = m.get("upload_throttle_sleep_time", 0.0)

                # Per-client prediction/debug fields. These are computed before
                # committing the current round to round_stats, so they reflect
                # the server's pre-update estimate for this reply.
                try:
                    ex_self = float(m.get("num-examples", 0) or 0)
                except Exception:
                    ex_self = 0.0

                uid_for_pred = device_uid if device_uid not in ("", None) else str(cid)
                identity_key_self = f"{device_name}:{uid_for_pred}"

                pred_train_time_self = self._predict_train_time(identity_key_self, ex_self, train_config)
                if pred_train_time_self <= 0.0:
                    pred_train_time_self = self._predict_train_time(device_name, ex_self, train_config)

                try:
                    payload_bytes_self = float(upload_throttle_payload_bytes or 0.0)
                except Exception:
                    payload_bytes_self = 0.0
                try:
                    throttle_mbps_self = float(upload_throttle_mbps or 0.0)
                except Exception:
                    throttle_mbps_self = 0.0

                pred_comm_time_self = self._predict_comm_time(identity_key_self, device_name, payload_bytes_self, throttle_mbps_self)
                pred_total_time_self = float(pred_train_time_self or 0.0) + float(pred_comm_time_self or 0.0)

                try:
                    actual_train_time_self = float(train_loop_time if train_loop_time not in ("", None) else 0.0)
                except Exception:
                    actual_train_time_self = 0.0
                try:
                    actual_upload_time_self = float(client_upload_elapsed_time if client_upload_elapsed_time not in ("", None) else 0.0)
                except Exception:
                    actual_upload_time_self = 0.0
                try:
                    actual_total_time_self = float(comp_time or 0.0) + actual_upload_time_self
                except Exception:
                    actual_total_time_self = actual_train_time_self + actual_upload_time_self

                est_beta_self = self.beta_sec_per_token.get(identity_key_self)
                if est_beta_self is None:
                    est_beta_self = self.beta_sec_per_token.get(device_name, "")
                est_alpha_self = self.time_scale.get(identity_key_self)
                if est_alpha_self is None:
                    est_alpha_self = self.time_scale.get(device_name, 1.0)
                est_upload_rate_bps_self = self._estimate_upload_rate_from_history(identity_key_self, device_name)
                est_upload_rate_mbps_self = (est_upload_rate_bps_self / 1e6) if est_upload_rate_bps_self and est_upload_rate_bps_self > 0 else ""

                try:
                    ep_self = float(local_epochs if local_epochs not in ("", None) else 1.0)
                except Exception:
                    ep_self = 1.0
                try:
                    seq_self = float(seq_len if seq_len not in ("", None) else 0.0)
                except Exception:
                    seq_self = 0.0
                try:
                    input_tokens_self = float(input_tokens if input_tokens not in ("", None) else 0.0)
                except Exception:
                    input_tokens_self = 0.0
                if input_tokens_self > 0.0:
                    assigned_workload_proxy = input_tokens_self
                elif seq_self > 0.0:
                    assigned_workload_proxy = ex_self * ep_self * seq_self
                else:
                    assigned_workload_proxy = ex_self * ep_self * float(self.tokens_per_example)

                # Estimate local_steps per client using its effective epochs
                local_steps = ""
                try:
                    ne = float(m.get("num-examples", 0) or 0)
                    bs = float(batch_size) if batch_size not in ("", None) else None
                    ep = float(local_epochs) if local_epochs not in ("", None) else None
                    if bs and ep and bs > 0 and ep > 0:
                        local_steps = m.get("local_steps", "",)

                        if local_steps in ("", None):
                            uid_steps = (train_config.get(f"local_steps_uid_{uid_suffix}") if uid_suffix else None)

                            if uid_steps not in ("", None):
                                local_steps = uid_steps

                            elif device_name == "RPi5":
                                local_steps = train_config.get("local_steps_RPi5", train_config.get("local_steps", 0,),)

                            elif device_name == "jetson_orin_nx":
                                local_steps = train_config.get("local_steps_jetson_orin_nx", train_config.get("local_steps", 0,),)
                            
                            elif device_name == "jetson_orin_nano_super":
                                local_steps = train_config.get(
                                    "local_steps_jetson_orin_nano_super",
                                    train_config.get("local_steps_jetson_orin_nx", train_config.get("local_steps", 0,),),)

                            else:
                                local_steps = train_config.get("local_steps", 0,)

                        # Backward-compatible fallback for runs that do not explicitly configure H.
                        if local_steps in ("", None, 0, "0"):
                            try:
                                ne = float(m.get("num-examples", 0) or 0)
                                bs = (float(batch_size) if batch_size not in ("", None) else None)
                                ep = (float(local_epochs) if local_epochs not in ("", None) else None)
                                if bs is not None and ep is not None and bs > 0 and ep > 0:
                                    local_steps = int(math.ceil(ne / bs) * ep)
                                else:
                                    local_steps = ""
                            except Exception:

                                local_steps = ""
                except Exception:
                    local_steps = ""

                knob_row = [
                    batch_size, local_epochs, local_steps,
                    max_samples, seq_len, lora_rank, lora_modules,
                    round_mode, rpi_max_samples, rpi_local_epochs,
                    jetson_max_samples, jetson_local_epochs,
                    upload_throttle_mbps, upload_throttle_payload_bytes, upload_throttle_sleep_time,
                    pred_train_time_self, pred_comm_time_self, pred_total_time_self,
                    actual_train_time_self, actual_upload_time_self, actual_total_time_self,
                    est_beta_self, est_alpha_self, est_upload_rate_mbps_self,
                    assigned_workload_proxy,
                ]
            else:
                # For non-train phases, keep knob columns empty (same length as above)
                knob_row = ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""]

            rows.append(base_row + knob_row)

            # === 여기서 전략 state 업데이트 (round_obs에만 적재) ===
            if phase == "train" and device_name is not None:
                ex = float(m.get("num-examples", 0) or 0)
                t_train = float(train_loop_time if train_loop_time not in ("", None) else 0.0)
                if update_norm in ("", None):
                    un = 0.0
                else:
                    try:
                        un = float(update_norm)
                    except Exception:
                        un = 0.0

                # Track runtime cid -> device metadata for preservation rounds.
                self.cid_to_device[cid] = device_name
                self.cid_to_device[str(cid)] = device_name
                self.cid_to_device_uid[cid] = str(device_uid)
                self.cid_to_device_uid[str(cid)] = str(device_uid)

                # Use a per-session composite key. Flower MetricRecord does not allow
                # string-valued metrics, so we intentionally avoid device_uid here.
                client_key = f"{device_name}:{device_uid}"

                # Prefer exact token counts reported by the client; fall back to a coarse
                # example-based approximation only if token metrics are unavailable.
                try:
                    input_tokens_metric = float(m.get("input_tokens", 0) or 0)
                except Exception:
                    input_tokens_metric = 0.0
                try:
                    supervised_tokens_metric = float(m.get("supervised_tokens", 0) or 0)
                except Exception:
                    supervised_tokens_metric = 0.0

                # Keep the old approximation as a fallback for backward compatibility.
                try:
                    ep_dev = float(local_epochs if local_epochs not in ("", None) else 1.0)
                except Exception:
                    ep_dev = 1.0

                if input_tokens_metric > 0.0:
                    tokens = input_tokens_metric
                else:
                    try:
                        tokens = float(assigned_workload_proxy or 0.0)
                    except Exception:
                        tokens = ex * ep_dev * float(self.tokens_per_example)

                if round_obs is not None:
                    round_obs[client_key] = {
                        "cid": cid,
                        "identity_key": client_key,
                        "device_name": device_name,
                        "device_uid": device_uid,
                        "num_examples": ex,
                        "local_steps": float(local_steps if local_steps not in ("", None) else 0.0),
                        "input_tokens": input_tokens_metric,
                        "supervised_tokens": supervised_tokens_metric,
                        "train_time": t_train,
                        "tokens": tokens,
                        "update_norm": un,
                        "upload_time": float(client_upload_elapsed_time or 0.0) if client_upload_elapsed_time not in ("", None) else 0.0,
                        "payload_bytes": float(m.get("upload_throttle_payload_bytes", 0) or 0.0),
                        "local_epochs": float(local_epochs if local_epochs not in ("", None) else 1.0),
                        "max_samples": float(max_samples if max_samples not in ("", None) else ex),
                    }

        # --- Compute round-level predicted train times from *previous* rounds only ---
        pred_rpi = ""
        pred_jet = ""
        if phase == "train":
            # 1) Snapshot current train_config for this round (used later by _estimate_beta_from_history)
            self.round_configs[server_round] = copy.deepcopy(train_config)

            # 2) Collect current round observations from round_obs (may be empty)
            # round_obs is keyed by "<device_name>:<cid>" so same-type replicas are preserved.
            current_rs = round_obs or {}

            # Aggregate per-client observations back into device-type totals for policy logic.
            rpi_ex = 0.0
            jet_ex = 0.0
            nano_ex = 0.0
            rpi_obs_t = 0.0
            jet_obs_t = 0.0
            nano_obs_t = 0.0
            rpi_max_t = 0.0
            jet_max_t = 0.0
            nano_max_t = 0.0
            rpi_tokens = 0.0
            jet_tokens = 0.0
            nano_tokens = 0.0
            rpi_updates = 0.0
            jet_updates = 0.0
            nano_updates = 0.0
            rpi_count = 0
            jet_count = 0
            nano_count = 0

            for entry in current_rs.values():
                dev = entry.get("device_name")
                ex = float(entry.get("num_examples", 0.0) or 0.0)
                tt = float(entry.get("train_time", 0.0) or 0.0)
                tok = float(entry.get("tokens", 0.0) or 0.0)
                upd = float(entry.get("update_norm", 0.0) or 0.0)
                if dev == "RPi5":
                    rpi_ex += ex
                    rpi_obs_t += tt
                    rpi_max_t = max(rpi_max_t, tt)
                    rpi_tokens += tok
                    rpi_updates += upd
                    rpi_count += 1
                elif dev == "jetson_orin_nx":
                    jet_ex += ex
                    jet_obs_t += tt
                    jet_max_t = max(jet_max_t, tt)
                    jet_tokens += tok
                    jet_updates += upd
                    jet_count += 1
                elif dev == "jetson_orin_nano_super":
                    nano_ex += ex
                    nano_obs_t += tt
                    nano_max_t = max(nano_max_t, tt)
                    nano_tokens += tok
                    nano_updates += upd
                    nano_count += 1

            # 3) Predict train time for this round using ONLY previous round_stats
            #    (round_stats has not been updated with current_rs yet)
            pred_rpi = self._predict_train_time("RPi5", rpi_ex, train_config)
            pred_jet = self._predict_train_time("jetson_orin_nx", jet_ex, train_config)

            # 4) Observed train_time from this round (for logging + calibration).
            # Synchronous FL round completion is determined by the slowest client in
            # each device group, so use the max per-device time for policy
            # calibration. Keep averages only for debug logging.
            avg_rpi = (rpi_obs_t / rpi_count) if rpi_count > 0 else 0.0
            avg_jet = (jet_obs_t / jet_count) if jet_count > 0 else 0.0
            avg_nano = (nano_obs_t / nano_count) if nano_count > 0 else 0.0
            obs_rpi = rpi_max_t if rpi_count > 0 else 0.0
            obs_jet = jet_max_t if jet_count > 0 else 0.0
            obs_nano = nano_max_t if nano_count > 0 else 0.0

            # --- Logging predicted vs observed train_time for debugging ---
            try:
                log(
                    INFO,
                    "[FedAvgLoRA][ROUND %s] train_time (s): pred RPi5=%.3f, pred Jetson=%.3f | obs_max RPi5=%.3f, obs_max Jetson=%.3f, obs_max Nano=%.3f | obs_avg RPi5=%.3f, obs_avg Jetson=%.3f, obs_avg Nano=%.3f",
                    server_round,
                    float(pred_rpi) if pred_rpi not in ("", None) else -1.0,
                    float(pred_jet) if pred_jet not in ("", None) else -1.0,
                    float(obs_rpi),
                    float(obs_jet),
                    float(obs_nano),
                    float(avg_rpi),
                    float(avg_jet),
                    float(avg_nano),
                )
            except Exception:
                pass

            # --- Online calibration of per-device time scale ---
            try:
                rnd = server_round
                if rnd >= self.min_beta_round:
                    self._update_time_scale("RPi5", pred_rpi, obs_rpi)
                    self._update_time_scale("jetson_orin_nx", pred_jet, obs_jet)
                    pred_nano = self._predict_train_time("jetson_orin_nano_super", nano_ex, train_config)
                    self._update_time_scale("jetson_orin_nano_super", pred_nano, obs_nano)
            except Exception:
                # Calibration is best-effort; failures should not break training.
                pass

            # Also calibrate per-client composite identity keys within the current run.
            try:
                rnd = server_round
                if rnd >= self.min_beta_round:
                    for entry in current_rs.values():
                        identity_key = entry.get("identity_key")
                        ex_i = float(entry.get("num_examples", 0.0) or 0.0)
                        obs_i = float(entry.get("train_time", 0.0) or 0.0)
                        if identity_key in ("", None):
                            continue
                        pred_i = self._predict_train_time(identity_key, ex_i, train_config)
                        self._update_time_scale(identity_key, pred_i, obs_i)
            except Exception:
                pass

            # 5) 이제서야 round_stats와 window_history에 이번 라운드 관측치를 commit
            if current_rs:
                aggregated_rs = {
                    "RPi5": {
                        "num_examples": rpi_ex,
                        "num_examples_per_client": (rpi_ex / rpi_count) if rpi_count > 0 else 0.0,
                        "train_time": rpi_max_t if rpi_count > 0 else 0.0,
                        "avg_train_time": (rpi_obs_t / rpi_count) if rpi_count > 0 else 0.0,
                        "tokens": rpi_tokens,
                        "tokens_per_client": (rpi_tokens / rpi_count) if rpi_count > 0 else 0.0,
                        "update_norm": (rpi_updates / rpi_count) if rpi_count > 0 else 0.0,
                        "upload_time": 0.0,
                        "payload_bytes": 0.0,
                    },
                    "jetson_orin_nx": {
                        "num_examples": jet_ex,
                        "num_examples_per_client": (jet_ex / jet_count) if jet_count > 0 else 0.0,
                        "train_time": jet_max_t if jet_count > 0 else 0.0,
                        "avg_train_time": (jet_obs_t / jet_count) if jet_count > 0 else 0.0,
                        "tokens": jet_tokens,
                        "tokens_per_client": (jet_tokens / jet_count) if jet_count > 0 else 0.0,
                        "update_norm": (jet_updates / jet_count) if jet_count > 0 else 0.0,
                        "upload_time": 0.0,
                        "payload_bytes": 0.0,
                    },
                    "jetson_orin_nano_super": {
                        "num_examples": nano_ex,
                        "num_examples_per_client": (nano_ex / nano_count) if nano_count > 0 else 0.0,
                        "train_time": nano_max_t if nano_count > 0 else 0.0,
                        "avg_train_time": (nano_obs_t / nano_count) if nano_count > 0 else 0.0,
                        "tokens": nano_tokens,
                        "tokens_per_client": (nano_tokens / nano_count) if nano_count > 0 else 0.0,
                        "update_norm": (nano_updates / nano_count) if nano_count > 0 else 0.0,
                        "upload_time": 0.0,
                        "payload_bytes": 0.0,
                    },
                    # Keep raw per-client observations for debugging and analysis.
                    "clients": current_rs,
                }
                inferred_slowest = self._infer_slowest_device_name(aggregated_rs)
                if inferred_slowest not in ("", None):
                    self.slowest_device_name = inferred_slowest
                self.round_stats[server_round] = aggregated_rs

                if self.use_token_ratio:
                    rpi_val = rpi_tokens
                    jet_val = jet_tokens
                    nano_val = nano_tokens
                else:
                    rpi_val = rpi_ex
                    jet_val = jet_ex
                    nano_val = nano_ex

                self.window_history.append({
                    "RPi5": rpi_val,
                    "jetson_orin_nx": jet_val,
                    "jetson_orin_nano_super": nano_val,
                })
                if len(self.window_history) > self.window_W:
                    self.window_history.pop(0)

        # Extend each row with the prediction columns
        extended_rows = []
        for row in rows:
            if phase == "train":
                extended_rows.append(row + [pred_rpi, pred_jet])
            else:
                # For non-train phases, keep the two prediction columns empty
                extended_rows.append(row + ["", ""])

        with open(self.csv_path, "a", newline="") as f:
            print("\n=== DEBUG rows before CSV write ===")
            print("rows count:", len(extended_rows))
            if len(extended_rows) > 0:
                print("example row[0]:", extended_rows[0])
            else:
                print("NO rows -> CSV will be empty!")

            csv.writer(f).writerows(extended_rows)

    def _client_window_token_totals(self) -> dict[str, float]:
        """Return recent W-round token totals per client identity."""
        totals: dict[str, float] = {}
        if not self.round_stats:
            return totals

        rounds = []
        for r in self.round_stats.keys():
            try:
                rounds.append(int(r))
            except Exception:
                pass

        if not rounds:
            return totals

        start_round = max(rounds) - max(1, int(self.window_W)) + 1

        for r in sorted(rounds):
            if r < start_round:
                continue

            stats = self.round_stats.get(r, {})
            clients = stats.get("clients", {}) if isinstance(stats, dict) else {}
            if not isinstance(clients, dict):
                continue

            for client_key, entry in clients.items():
                if not isinstance(entry, dict):
                    continue

                identity = str(entry.get("device_uid") or entry.get("cid") or client_key)
                if not identity:
                    continue

                try:
                    tokens = float(entry.get("tokens", 0.0) or 0.0)
                except Exception:
                    tokens = 0.0

                totals[identity] = totals.get(identity, 0.0) + max(0.0, tokens)

        return totals

    def _known_client_identities(self) -> set[str]:
        """Return client identities seen by the server so far."""
        ids: set[str] = set()
        if not self.round_stats:
            return ids

        for stats in self.round_stats.values():
            clients = stats.get("clients", {}) if isinstance(stats, dict) else {}
            if not isinstance(clients, dict):
                continue

            for client_key, entry in clients.items():
                if not isinstance(entry, dict):
                    continue

                identity = str(entry.get("device_uid") or entry.get("cid") or client_key)
                if identity:
                    ids.add(identity)

        return ids

    def _compute_underrepresented_clients(self) -> tuple[set[str], float, dict[str, float]]:
        """Find clients whose normalized participation is below rho times fair share.

        For N known clients, the fair target is pi_i = 1/N. A client is protected
        when q_i < rho*pi_i. This keeps the threshold independent of the number
        or type of devices.
        """
        ids = self._known_client_identities()
        totals = self._client_window_token_totals()
        ids.update(totals.keys())

        if not ids:
            return set(), 0.0, {}

        n = max(1, len(ids))
        threshold = float(self.participation_fairness_rho) / float(n)
        total_tokens = float(sum(max(0.0, v) for v in totals.values()))

        shares: dict[str, float] = {}
        protected: set[str] = set()

        for identity in ids:
            q = (float(totals.get(identity, 0.0)) / total_tokens) if total_tokens > 0.0 else 0.0
            shares[identity] = q
            if total_tokens > 0.0 and q < threshold:
                protected.add(identity)

        return protected, threshold, shares

    def _get_slowest_device_name(self) -> str | None:
        """Return the currently slowest observed device group.

        Falls back to RPi5 before any observations are available to preserve the
        original cold-start behavior.
        """
        if self.slowest_device_name not in ("", None):
            return self.slowest_device_name
        inferred = self._infer_slowest_device_name_from_history()
        if inferred not in ("", None):
            self.slowest_device_name = inferred
            return inferred
        return "RPi5"

    def _infer_slowest_device_name(self, round_stats: dict) -> str | None:
        """Infer the slowest device group from one round's observed train times."""
        candidates = []
        for dev_name in ("RPi5", "jetson_orin_nx", "jetson_orin_nano_super"):
            entry = round_stats.get(dev_name, {}) if isinstance(round_stats, dict) else {}
            try:
                t_train = float(entry.get("train_time", 0.0) or 0.0)
            except Exception:
                t_train = 0.0
            if t_train > 0.0:
                candidates.append((t_train, dev_name))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]

    def _infer_slowest_device_name_from_history(self) -> str | None:
        """Infer the slowest device group from recent committed round_stats."""
        if not self.round_stats:
            return None
        latest_rounds = sorted(self.round_stats.keys(), reverse=True)[: self.max_beta_history]
        totals: dict[str, list[float]] = {}
        for rnd in latest_rounds:
            rs = self.round_stats.get(rnd, {})
            for dev_name in ("RPi5", "jetson_orin_nx", "jetson_orin_nano_super"):
                entry = rs.get(dev_name, {}) if isinstance(rs, dict) else {}
                try:
                    t_train = float(entry.get("train_time", 0.0) or 0.0)
                except Exception:
                    t_train = 0.0
                if t_train > 0.0:
                    totals.setdefault(dev_name, []).append(t_train)
        if not totals:
            return None
        avg_times = [(sum(vals) / len(vals), dev_name) for dev_name, vals in totals.items() if vals]
        if not avg_times:
            return None
        avg_times.sort(reverse=True)
        return avg_times[0][1]

    def _compute_slowest_device_ratio(self) -> float:
        """최근 window_W 라운드에서 가장 느린 디바이스 그룹의 기여 비율."""
        if not self.window_history:
            return 0.5  # 초기에는 중립

        slow_dev = self._get_slowest_device_name()
        if slow_dev in ("", None):
            return 0.5

        slow_sum = sum(entry.get(slow_dev, 0.0) for entry in self.window_history)
        total = 0.0
        for entry in self.window_history:
            for value in entry.values():
                try:
                    total += float(value or 0.0)
                except Exception:
                    pass
        if total <= 0:
            return 0.5
        return slow_sum / total

    def _get_device_profile(self, device_name: str) -> dict | None:
        """Return profiling entry for given device name, with simple aliasing.

        The estimator JSON may use slightly different keys (e.g., "jetson-orin" vs
        "jetson_orin_nx"), so we normalize by removing underscores and lowering.
        """
        if not self.device_profiles:
            return None

        # Direct match
        if device_name in self.device_profiles:
            return self.device_profiles[device_name]

        norm = device_name.lower().replace("_", "")
        for key, prof in self.device_profiles.items():
            k_norm = str(key).lower().replace("_", "")
            if norm == k_norm or norm in k_norm or k_norm in norm:
                return prof
        return None

    def _estimate_beta_from_history(self, identity_key: str, cfg: ConfigRecord) -> float | None:
        """Estimate sec/token beta from recent train loop_time history.

        The input key may be either:
        - a stable per-client identity (preferred), e.g., device_uid
        - a grouped device-type key such as "RPi5" or "jetson_orin_nx"

        If per-client observations are available under round_stats[round]["clients"], they are
        preferred. Otherwise, the function falls back to grouped device-type aggregates.
        """
        samples: list[dict] = []

        for rnd, per_dev in self.round_stats.items():
            try:
                r = int(rnd)
            except Exception:
                continue

            if r < self.min_beta_round:
                continue

            dev_entry = None

            # 1) Prefer raw per-client observations keyed by stable identity
            clients = per_dev.get("clients", {}) if isinstance(per_dev, dict) else {}
            if isinstance(clients, dict) and identity_key in clients:
                dev_entry = clients.get(identity_key)

            # 2) Fallback to grouped aggregate entries (legacy / grouped mode)
            if dev_entry is None and isinstance(per_dev, dict):
                dev_entry = per_dev.get(identity_key)

            if not dev_entry:
                continue

            tokens = float(dev_entry.get("tokens_per_client", dev_entry.get("tokens", 0.0)) or 0.0)
            t_sec = float(dev_entry.get("train_time", 0.0) or 0.0)
            if tokens <= 0.0 or t_sec <= 0.0:
                continue

            samples.append({"round": r, "tokens": tokens, "train_time": t_sec})

        if not samples:
            return None

        samples.sort(key=lambda e: e["round"], reverse=True)
        samples = samples[: self.max_beta_history]

        num = 0.0
        den = 0.0
        for e in samples:
            tokens = float(e["tokens"])
            t_sec = float(e["train_time"])
            num += tokens * t_sec
            den += tokens * tokens

        if den <= 0.0:
            return None

        beta_sec_per_token = num / den
        if beta_sec_per_token <= 0.0:
            return None

        self.beta_sec_per_token[identity_key] = beta_sec_per_token
        return beta_sec_per_token

    def _estimate_tokens_per_example(self, identity_key: str, device_name: str) -> float:
        """Estimate tokens per example per local epoch from recent observations.

        round_stats stores `tokens` as the total observed input tokens for the
        local training loop, i.e., it already includes local epochs. For workload
        prediction, convert it back to tokens/example/epoch and use it to map a
        candidate sample cap and epoch count to an estimated token workload.
        """
        samples = []
        for rnd, per_dev in self.round_stats.items():
            try:
                r = int(rnd)
            except Exception:
                continue
            if r < self.min_beta_round:
                continue

            dev_entry = None
            clients = per_dev.get("clients", {}) if isinstance(per_dev, dict) else {}
            if isinstance(clients, dict) and identity_key in clients:
                dev_entry = clients.get(identity_key)
            if dev_entry is None and isinstance(per_dev, dict):
                dev_entry = per_dev.get(identity_key)
            if dev_entry is None and device_name != identity_key and isinstance(per_dev, dict):
                dev_entry = per_dev.get(device_name)
            if not dev_entry:
                continue

            try:
                tokens = float(dev_entry.get("tokens", 0.0) or 0.0)
                examples = float(dev_entry.get("num_examples", 0.0) or 0.0)
            except Exception:
                continue
            if tokens <= 0.0 or examples <= 0.0:
                continue

            if "local_epochs" in dev_entry:
                try:
                    ep = float(dev_entry.get("local_epochs", 1.0) or 1.0)
                except Exception:
                    ep = 1.0
            else:
                cfg = self.round_configs.get(rnd, ConfigRecord())
                try:
                    ep = self._resolve_effective_epochs_for_round(identity_key, device_name, cfg)
                except Exception:
                    ep = 1.0

            if ep <= 0.0:
                ep = 1.0
            samples.append((r, tokens / max(examples * ep, 1e-9)))

        if not samples:
            return float(self.tokens_per_example)

        samples.sort(key=lambda x: x[0], reverse=True)
        recent = [v for _, v in samples[: self.max_beta_history] if v > 0.0]
        if not recent:
            return float(self.tokens_per_example)
        return float(sum(recent) / len(recent))

    def _estimate_upload_rate_from_history(self, identity_key: str, device_name: str) -> float | None:
        """Estimate upload throughput in bits/sec from recent client observations."""
        samples: list[tuple[int, float]] = []

        for rnd, per_dev in self.round_stats.items():
            try:
                r = int(rnd)
            except Exception:
                continue

            clients = per_dev.get("clients", {}) if isinstance(per_dev, dict) else {}
            dev_entry = None

            if isinstance(clients, dict) and identity_key in clients:
                dev_entry = clients.get(identity_key)

            if dev_entry is None and isinstance(per_dev, dict):
                dev_entry = per_dev.get(device_name)

            if not dev_entry:
                continue

            try:
                payload_bytes = float(dev_entry.get("payload_bytes", 0.0) or 0.0)
                upload_time = float(dev_entry.get("upload_time", 0.0) or 0.0)
            except Exception:
                continue

            if payload_bytes <= 0.0 or upload_time <= 0.0:
                continue

            samples.append((r, payload_bytes * 8.0 / upload_time))

        if not samples:
            return self.upload_rate_bps.get(identity_key) or self.upload_rate_bps.get(device_name)

        samples.sort(key=lambda x: x[0], reverse=True)
        recent = [v for _, v in samples[: self.max_beta_history] if v > 0.0]
        if not recent:
            return None

        rate = float(sum(recent) / len(recent))
        self.upload_rate_bps[identity_key] = rate
        return rate

    def _predict_comm_time(
        self,
        identity_key: str,
        device_name: str,
        payload_bytes: float,
        throttle_mbps: float = 0.0,
    ) -> float:
        """Predict client-side upload delay for the returned LoRA payload."""
        try:
            payload_bytes = float(payload_bytes or 0.0)
        except Exception:
            payload_bytes = 0.0

        if payload_bytes <= 0.0:
            return 0.0

        try:
            throttle_mbps = float(throttle_mbps or 0.0)
        except Exception:
            throttle_mbps = 0.0

        if throttle_mbps > 0.0:
            return payload_bytes * 8.0 / (throttle_mbps * 1e6)

        rate_bps = self._estimate_upload_rate_from_history(identity_key, device_name)
        if rate_bps is None or rate_bps <= 0.0:
            return 0.0

        return payload_bytes * 8.0 / rate_bps

    def _predict_train_time(self, identity_key: str, num_examples: float, cfg: ConfigRecord) -> float:
        """Predict loop-dominant training time in seconds.

        `identity_key` can be a grouped device key such as "RPi5" or a
        per-client key such as "RPi5:rpi5-4". Per-client UID workload overrides
        must take priority over device-type defaults.
        """
        if identity_key in ("", None):
            return 0.0

        key_device, _ = self._split_identity_key(identity_key)
        device_name = key_device if key_device not in ("", None) else str(identity_key)
        effective_examples, effective_epochs = self._resolve_effective_workload(
            identity_key, device_name, cfg, num_examples
        )
        if effective_examples <= 0.0 or effective_epochs <= 0.0:
            return 0.0

        beta = self._estimate_beta_from_history(identity_key, cfg)
        if beta is None and device_name != identity_key:
            beta = self._estimate_beta_from_history(device_name, cfg)

        if beta is None:
            prof = self._get_device_profile(device_name)
            if prof is not None:
                for k in (
                    "beta_sec_per_token",
                    "sec_per_token",
                    "loop_sec_per_token",
                    "train_sec_per_token",
                ):
                    try:
                        val = float(prof.get(k, 0.0) or 0.0)
                    except Exception:
                        val = 0.0
                    if val > 0.0:
                        beta = val
                        break

        if beta is None or beta <= 0.0:
            return 0.0

        tokens_per_example = self._estimate_tokens_per_example(identity_key, device_name)
        sample_limited_tokens = (
            float(effective_examples) * float(effective_epochs) * float(tokens_per_example)
        )
        effective_steps = self._resolve_effective_steps(identity_key, device_name, cfg,)
        if effective_steps > 0:
            tokens_per_step = (self._estimate_tokens_per_step_for_device(device_name, cfg,))
            step_limited_tokens = (float(effective_steps) * float(tokens_per_step))
            token_workload = min(sample_limited_tokens, step_limited_tokens,)
        else:
            token_workload = sample_limited_tokens
        if token_workload <= 0.0:
            return 0.0

        scale = self.time_scale.get(identity_key)
        if scale is None and device_name != identity_key:
            scale = self.time_scale.get(device_name)
        if scale is None:
            scale = 1.0
        try:
            scale = float(scale)
        except Exception:
            scale = 1.0
        if scale <= 0.0:
            scale = 1.0

        return float(beta) * float(token_workload) * float(scale)
