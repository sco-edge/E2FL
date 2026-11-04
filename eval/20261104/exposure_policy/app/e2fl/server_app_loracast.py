"""LoRACast Flower ServerApp for federated LLM fine-tuning."""
import os as _isolated_os
from pathlib import Path as _IsolatedPath
_ISOLATED_APP_ROOT = _IsolatedPath(__file__).resolve().parents[1]
_requested_root = _isolated_os.environ.get("LORACAST_WORKSPACE_ROOT")
if _requested_root and _IsolatedPath(_requested_root).resolve() != _ISOLATED_APP_ROOT:
    raise RuntimeError("isolated Flower app refuses an external workspace root")
_isolated_os.environ["LORACAST_WORKSPACE_ROOT"] = str(_ISOLATED_APP_ROOT)
from log import WrlsEnv
from datetime import datetime
import subprocess, os, logging, time, re, json, hashlib, shutil, random
from pathlib import Path

import torch
import numpy as np
import flwr as fl
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from e2fl.task_llm import train, is_lora_model
from e2fl.runtime.llm_model import get_model, is_llm_model_name
from e2fl.round_cooldown import wait_after_round
from e2fl.utils import cfg, lora_cfg_from
from peft import get_peft_model_state_dict, set_peft_model_state_dict

# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.debug, logging.info, logging.warning, logging.error, logging.critical
class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

home_dir = os.path.expanduser("~")
_LOCAL_PATH = os.environ.get("LORACAST_WORKSPACE_ROOT", f"{home_dir}/EEFL/E2FL")

# 전역 로거 구성 (서버 전용)
E2FL_DEBUG = os.environ.get("E2FL_DEBUG", "0") == "1"
logger = logging.getLogger("e2fl_server")
if not logger.handlers:
    logger.setLevel(logging.DEBUG if E2FL_DEBUG else logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)

# 안전하게 Flower 파일 로거 구성 (옵션)
try:
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    fl.common.logger.configure(identifier="e2fl_server", filename=_LOCAL_PATH+f"/eval/fl_server_{current_time}.txt")
except Exception:
    logger.debug("fl.common.logger.configure unavailable or failed; continuing without fl file logger")


device_lookup = {
    1: "RPi5",
    2: "jetson_orin_nx",
}


def _config_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no"):
        return False
    raise ValueError(f"invalid Boolean run config value: {value!r}")

def ensure_time_sync(
    ntp_server: str | None = None,
    max_offset_ms: float = 50.0,
):
    """
    간단한 NTP 오프셋 체크 + 필요하면 ntpdate 시도.
    - ntplib 없으면 그냥 워닝만 찍고 스킵.
    - root 권한 아니면 ntpdate는 안 돌리고 워닝만.
    """
    global logger

    ntp_server = ntp_server or os.environ.get("NTP_SERVER", "time.google.com")

    try:
        import ntplib
    except ImportError:
        logger.warning("[TimeSync] ntplib not installed; skipping NTP check")
        return

    try:
        client = ntplib.NTPClient()
        res = client.request(ntp_server, version=3, timeout=3)
        offset_ms = res.offset * 1000.0
        logger.info(f"[TimeSync] NTP offset to {ntp_server}: {offset_ms:.2f} ms")

        if abs(offset_ms) > max_offset_ms:
            logger.warning(
                f"[TimeSync] Large offset ({offset_ms:.2f} ms) detected."
            )
            # root이면 한 번 ntpdate 시도
            try:
                if hasattr(os, "geteuid") and os.geteuid() == 0:
                    logger.info(
                        f"[TimeSync] Trying to correct time via `ntpdate -u {ntp_server}`"
                    )
                    subprocess.run(
                        ["ntpdate", "-u", ntp_server],
                        check=False,
                    )
                else:
                    logger.warning(
                        "[TimeSync] Not running as root; "
                        "please configure system NTP (chrony/systemd-timesyncd) manually."
                    )
            except Exception as e2:
                logger.warning(f"[TimeSync] Failed to run ntpdate: {e2}")
    except Exception as e:
        logger.warning(f"[TimeSync] NTP check/sync failed: {e}")


def create_strategy(context: Context):
    """Create a selected strategy with shared experiment logging and checks."""
    from e2fl.strategy.LoRACast import LoRACastCommAware
    from e2fl.strategy.FinalExposureLoRACast import FinalExposureLoRACast
    from e2fl.strategy.DynamicSizingPilot import PilotFrozenCost, PilotLastRate, PilotOnline, ProbeFixed, ProbePairFixed
    from e2fl.strategy.CrossClientSchedule import (
        ScheduledFixed, ScheduledFrozenCost, ScheduledLastRate, ScheduledOnline,
    )
    from e2fl.strategy.FedAvgLoRA import FedAvgLoRA
    from e2fl.strategy.PyramidFL import PyramidFLLoRA_NoDropout

    fraction_train = float(context.run_config["fraction-train"])
    min_clients = int(context.run_config["min-clients"])
    model_name = str(context.run_config.get("model", "unknown_model"))
    experiment_id = str(cfg(context, "experiment-id", "") or "").strip()
    if experiment_id and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", experiment_id):
        raise ValueError("experiment-id must use only letters, digits, dots, underscores, and hyphens")
    experiment_root = os.path.abspath(os.path.expanduser(str(
        cfg(context, "experiment-output-root", "") or
        os.path.join(_LOCAL_PATH, "eval", "experiments")
    )))
    log_dir = os.path.join(experiment_root, experiment_id, "server") if experiment_id else "eval/logs_loracast"
    rebalancing_value = context.run_config.get("rebalancing-enabled")
    rebalancing_enabled = (
        None if rebalancing_value is None or str(rebalancing_value).lower() == "auto"
        else _config_bool(rebalancing_value)
    )
    if rebalancing_enabled and not (
        _config_bool(cfg(context, "enable-rpi-only-rounds", True))
        and _config_bool(cfg(context, "use-token-ratio", True))
    ):
        raise ValueError("rebalancing-enabled requires enable-rpi-only-rounds and use-token-ratio")
    device_profiles_path = context.run_config.get("device-profiles-path")
    if device_profiles_path and not os.path.isabs(device_profiles_path):
        device_profiles_path = os.path.join(_LOCAL_PATH, device_profiles_path)
    manifest_path = str(cfg(context, "partition-manifest-path", "") or "")
    if manifest_path and not os.path.isabs(manifest_path):
        manifest_path = os.path.join(_LOCAL_PATH, manifest_path)
    manifest_sha256 = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest() if manifest_path else ""
    manifest_client_count = int(json.loads(Path(manifest_path).read_text())["num_clients"]) if manifest_path else 0

    strategy_name = str(cfg(context, "strategy", "LoRACastCommAware"))
    strategies = {
        "LoRACastCommAware": LoRACastCommAware,
        "FinalExposureLoRACast": FinalExposureLoRACast,
        "PilotFrozenCost": PilotFrozenCost,
        "PilotLastRate": PilotLastRate,
        "PilotOnline": PilotOnline,
        "ProbeFixed": ProbeFixed,
        "ProbePairFixed": ProbePairFixed,
        "ScheduledFixed": ScheduledFixed,
        "ScheduledFrozenCost": ScheduledFrozenCost,
        "ScheduledLastRate": ScheduledLastRate,
        "ScheduledOnline": ScheduledOnline,
        "FedAvgLoRA": FedAvgLoRA,
        "PyramidFLLoRA_NoDropout": PyramidFLLoRA_NoDropout,
    }
    if strategy_name in ("PyramidFLLoRA_MatchedBounds", "PyramidFLLoRA_IIDMatched32"):
        from e2fl.strategy.PyramidFLMatchedBounds import PyramidFLLoRA_MatchedBounds
        strategies[strategy_name] = PyramidFLLoRA_MatchedBounds
    if strategy_name not in strategies:
        raise ValueError(f"unsupported LoRACast experiment strategy: {strategy_name}")
    strategy_kwargs = dict(
        fraction_train=fraction_train,
        fraction_evaluate=1.0,
        min_available_nodes=min_clients,
        log_dir=log_dir,
        model_name=model_name,
        latte=False,
        model_info=None,
        device_lookup=device_lookup,
        ROOT=_LOCAL_PATH,
        device_profiles_path=device_profiles_path,
        tokens_per_example=float(cfg(context, "tokens-per-example", 256)),
        enable_rpi_only_rounds=_config_bool(
            cfg(context, "enable-rpi-only-rounds", True)
        ),
        use_token_ratio=_config_bool(cfg(context, "use-token-ratio", True)),
        enable_deadline_guard=bool(
            cfg(context, "enable-deadline-guard", False)
        ),
        participation_guard_mode=str(
            cfg(context, "participation-guard-mode", "per-client")
        ),
        participation_fairness_rho=float(
            cfg(context, "participation-fairness-rho", 0.3)
        ),
        aggregation_mode=str(cfg(context, "aggregation-mode", "processed_samples")),
        experiment_id=experiment_id,
        rebalancing_enabled=rebalancing_enabled,
        expected_manifest_sha256=manifest_sha256,
        expected_manifest_client_count=manifest_client_count,
        estimator_freeze_after_round=int(cfg(context, "estimator-freeze-after-round", 0)),
        residual_calibration_enabled=_config_bool(
            cfg(context, "residual-calibration-enabled", False)
        ),
    )
    if strategy_name in ("LoRACastCommAware", "FinalExposureLoRACast", "PilotFrozenCost", "PilotLastRate", "PilotOnline", "ProbeFixed", "ProbePairFixed",
                         "ScheduledFixed", "ScheduledFrozenCost", "ScheduledLastRate", "ScheduledOnline"):
        strategy_kwargs.update(
            comm_ema_alpha=float(cfg(context, "comm-ema-alpha", 0.3)),
            comm_safety_factor=float(cfg(context, "comm-safety-factor", 1.0)),
            use_comm_prediction=_config_bool(cfg(context, "use-comm-prediction", True)),
            static_allocation_after_round=int(cfg(context, "static-allocation-after-round", 0)),
        )
    if strategy_name == "FinalExposureLoRACast":
        if str(cfg(context,"sample-selection-mode","prefix")) != "exposure_aware":
            raise ValueError("FinalExposureLoRACast requires the measured-visit exposure_aware sampler")
        if str(cfg(context,"aggregation-mode","processed_samples")) != "processed_samples":
            raise ValueError("FinalExposureLoRACast requires processed-sample aggregation")
        exposure_text=str(cfg(context,"exposure-max","none")).strip().lower()
        strategy_kwargs.update(
            exposure_max=None if exposure_text in ("none","disabled","") else float(exposure_text),
            nx_eligible_unique=int(cfg(context,"nx-eligible-unique",0)),
            nx_minimum_samples=int(cfg(context,"nx-minimum-samples",0)),
            nx_maximum_samples=int(cfg(context,"nx-maximum-samples",0)),
            nx_maximum_epochs=int(cfg(context,"nx-maximum-epochs",1)),
        )
    elif strategy_name == "PyramidFLLoRA_MatchedBounds":
        # This branch is used only by the new matched-bounds experiment ID.
        # The original PyramidFL-style strategy remains on its old defaults.
        scale = float(cfg(context, "pyramid-max-scale", 15.0))
        pi_cap = int(cfg(context, "matched-pi-max-quota", 4))
        jetson_cap = int(cfg(context, "matched-jetson-max-quota", 60))
        if scale != 15.0 or pi_cap != 4 or jetson_cap != 60:
            raise ValueError("matched PyramidFL bounds must be scale=15, Pi=4, Jetson=60")
        strategy_kwargs.update(base_quota=4, min_quota=4,
                               max_scale=scale, pi_max_quota=pi_cap,
                               jetson_max_quota=jetson_cap)
    elif strategy_name == "PyramidFLLoRA_IIDMatched32":
        # Separate IID cap-32 experiment; existing strategy paths are unchanged.
        scale = float(cfg(context, "pyramid-max-scale", 0))
        pi_cap = int(cfg(context, "matched-pi-max-quota", 0))
        jetson_cap = int(cfg(context, "matched-jetson-max-quota", 0))
        if (scale, pi_cap, jetson_cap) != (8.0, 4, 32):
            raise ValueError("IID matched PyramidFL bounds must be scale=8, Pi=4, Jetson=32")
        strategy_kwargs.update(base_quota=4, min_quota=4,
                               max_scale=scale, pi_max_quota=pi_cap,
                               jetson_max_quota=jetson_cap)
    return strategies[strategy_name](**strategy_kwargs)

# Create ServerApp (after Flower 1.21.0)
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    ensure_time_sync()
    dataset_name = context.run_config["dataset"]
    bs_train     = int(cfg(context, "batch_size", 32))
    bs_eval      = int(cfg(context, "eval-batch-size", max(1, bs_train)))
    lr_max       = float(cfg(context, "learning-rate-max", 5e-5))
    lr_min       = float(cfg(context, "learning-rate-min", 1e-6))
    wd           = float(cfg(context, "weight-decay", 5e-4))
    resume       = bool(cfg(context, "resume", True))
    # Flower launches ServerApp from SuperLink, not from the `flwr run` shell.
    # Run config is the transport that reaches this process; pass the value to
    # the strategy constructor in this process before it opens result files.
    policy_resume_round = int(cfg(context, "policy-resume-after-round", 0))
    if policy_resume_round and (resume or not 0 < policy_resume_round < int(cfg(context, "num-server-rounds", 1))):
        raise ValueError("policy recovery requires resume=false and a completed nonfinal round")
    os.environ["LORACAST_POLICY_RESUME_ROUND"] = str(policy_resume_round)
    quantization = int(cfg(context, "quantization", 4))
    grad_ckpt      = int(cfg(context, "gradient-checkpointing", False))
    lora_cfg     = lora_cfg_from(context)
    num_rounds = int(cfg(context, "num-server-rounds", 1))
    save_every_round = int(cfg(context, "save-every-round", 1))
    experiment_id = str(cfg(context, "experiment-id", "") or "").strip()
    if experiment_id and save_every_round != 1:
        raise ValueError("experiment runs require save-every-round=1 for time-to-quality evaluation")
    round_cooldown_enabled = _config_bool(cfg(context, "round-cooldown-enabled", False))
    if round_cooldown_enabled and not experiment_id:
        raise ValueError("round cooldown requires experiment-id")
    if experiment_id:
        experiment_seed = int(cfg(context, "experiment-seed", cfg(context, "sample-seed", 0)))
        random.seed(experiment_seed)
        np.random.seed(experiment_seed % (2**32))
        torch.manual_seed(experiment_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(experiment_seed)
        logger.info("[Server] Experiment initialization seed: %d", experiment_seed)
    local_epochs         = int(cfg(context, "local-epochs", 1))
    max_samples_per_round = int(cfg(context, "max-samples-per-round", 0))  # 0이면 제한 없음
    max_seq_len          = int(cfg(context, "max-seq-len", 256))
    server_timeout       = float(cfg(context, "server-timeout", 14400))

    # Initialize the LLM + LoRA model.  Vision/tutorial model code is
    # intentionally outside the LoRACast registration build.
    model_name = str(context.run_config["model"])
    if not is_llm_model_name(model_name):
        raise ValueError(
            f"LoRACast registration ServerApp supports LLM models only: {model_name!r}"
        )

    global_model = get_model(
        model_name,
        lora_cfg=lora_cfg,
        gradient_checkpointing=bool(grad_ckpt),
        quantization=quantization,
    )

    # Create checkpoint filename using model name, dataset, and LoRA/full mode
    safe_model = model_name.replace("/", "_").replace(":", "_")
    safe_dataset = dataset_name.replace("/", "_")
    mode_tag = "adapter" if is_lora_model(global_model) else "full"

    # Use configurable checkpoint root if provided, otherwise default to _LOCAL_PATH/eval
    experiment_root = os.path.abspath(os.path.expanduser(str(
        cfg(context, "experiment-output-root", "") or
        os.path.join(_LOCAL_PATH, "eval", "experiments")
    )))
    ckpt_root = (
        os.path.join(experiment_root, experiment_id, "checkpoints")
        if experiment_id else cfg(context, "checkpoint-path", f"{_LOCAL_PATH}/eval")
    )
    ckpt_dir = os.path.expanduser(ckpt_root)
    if not os.path.isabs(ckpt_dir):
        ckpt_dir = os.path.join(_LOCAL_PATH, ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    if experiment_id and not policy_resume_round:
        committed = Path(experiment_root) / experiment_id / "server" / "resume_state"
        if committed.is_dir() and any(committed.glob("round_*.json")):
            raise RuntimeError("existing completed rounds require explicit policy recovery config")
    if experiment_id and cfg(context, "partition-manifest-path", ""):
        manifest_path = Path(str(cfg(context, "partition-manifest-path", "")))
        if not manifest_path.is_absolute():
            manifest_path = Path(_LOCAL_PATH) / manifest_path
        manifest_snapshot = Path(experiment_root) / experiment_id / "server" / "partition_manifest.json"
        manifest_snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(manifest_path, manifest_snapshot)
    ckpt_name = f"{safe_model}_{safe_dataset}_{mode_tag}.pt"
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)

    if policy_resume_round:
        if not experiment_id:
            raise ValueError("policy recovery requires an experiment ID")
        policy_checkpoint = os.path.join(
            ckpt_dir, f"peft_{safe_model}_{policy_resume_round}.pt"
        )
        state = torch.load(policy_checkpoint, map_location="cpu")
        if not is_lora_model(global_model):
            raise RuntimeError("policy recovery requires the LoRA adapter checkpoint")
        set_peft_model_state_dict(global_model, state)
        arrays = ArrayRecord(get_peft_model_state_dict(global_model))
        logger.info("[Server] Policy recovery from completed round %s checkpoint %s",
                    policy_resume_round, policy_checkpoint)
    elif resume and os.path.exists(ckpt_path):
        logger.info(f"[Server] Resuming from checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        if is_lora_model(global_model):
            set_peft_model_state_dict(global_model, state)
            arrays = ArrayRecord(get_peft_model_state_dict(global_model))
        else:
            global_model.load_state_dict(state, strict=False)
            arrays = ArrayRecord(global_model.state_dict())
    else:
        if resume:
            logger.info(f"[Server] Checkpoint not found at {ckpt_path}. Starting from scratch.")
        arrays = ArrayRecord(
            get_peft_model_state_dict(global_model) if is_lora_model(global_model)
            else global_model.state_dict()
        )

    # Opt-in observation only; unchanged Grid/strategy when disabled.
    if _config_bool(cfg(context, "log_reply_arrivals", False)):
        from reply_arrival_instrumentation import install
        trace_path = Path(experiment_root) / experiment_id / "server" / "reply_arrivals.jsonl"
        install(grid, trace_path, experiment_id)

    # Instantiate strategy
    strategy = create_strategy(context)
    resume_snapshot = None
    if hasattr(strategy, "resume_completed_round"):
        from policy_resume import load_completed_round, save_completed_round
        resume_snapshot = lambda completed, checkpoint: save_completed_round(
            strategy, completed, checkpoint
        )
        if policy_resume_round:
            load_completed_round(strategy, policy_resume_round, policy_checkpoint)
    elif policy_resume_round:
        raise RuntimeError("policy recovery is unavailable for this strategy")
    train_conf = {
        # Standard training configuration
        "lr": lr_max,
        "weight_decay": wd,
        "batch_size": bs_train,
        "local_epochs": local_epochs,
        "max_samples_per_round": max_samples_per_round,
        "probe-nano-quota": int(cfg(context, "probe-nano-quota", 0)),
        "probe-nx-quota": int(cfg(context, "probe-nx-quota", 0)),
        "max_seq_len": max_seq_len,

        # Participation policy
        "enable-rpi-only-rounds": _config_bool(
            cfg(context, "enable-rpi-only-rounds", True,)
        ),
        "use-token-ratio": _config_bool(
            cfg(context, "use-token-ratio", True,)
        ),
        "enable-deadline-guard": bool(
            cfg(context, "enable-deadline-guard", False,)
        ),
        "participation-guard-mode": str(
            cfg(context, "participation-guard-mode", "per-client",)
        ),
        "participation-fairness-rho": float(
            cfg(context, "participation-fairness-rho", 0.3,)
        ),
        # Sliding-window policy
        "window-W": int(
            cfg(context, "window-W", 5,)
        ),
        "policy-warmup-rounds": int(
            cfg(context, "policy-warmup-rounds", 2,)
        ),

        # Normal RPi workload
        "rpi-normal-quota": int(
            cfg(context, "rpi-normal-quota", 128,)
        ),
        "rpi-sample-step": int(
            cfg(context, "rpi-sample-step", 32,)
        ),
        # Per-client correction workload
        "rpi-correction-max-quota": int(
            cfg(context, "rpi-correction-max-quota", 384,)
        ),
        "max-correction-time-s": float(
            cfg(context, "max-correction-time-s", 3600.0,)
        ),
        # Jetson workload expansion
        "jetson-base-quota": int(
            cfg(context, "jetson-base-quota", 64,)
        ),
        "jetson-max-quota": int(
            cfg(context, "jetson-max-quota", 4096,)
        ),
        "jetson-sample-step": int(
            cfg(context, "jetson-sample-step", 64,)
        ),
        "jetson-max-steps": int(
            cfg(context, "jetson-max-steps", 4096,)
        ),
        "jetson-step-increment": int(
            cfg(context, "jetson-step-increment", 1,)
        ),
        "jetson-target-margin": float(
            cfg(context, "jetson-target-margin", 0.95,)
        ),
    }

    # Start strategy, run for 'num_rounds'
    logger.info(f"[Server] Starting strategy with timeout={server_timeout:.1f}s")
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord(train_conf),
        num_rounds=num_rounds,
        timeout=server_timeout,
        evaluate_fn=make_global_evaluate(model_name, dataset_name,
                                         bs_eval, num_rounds, save_every_round,
                                         lora_cfg, grad_ckpt, quantization,
                                         checkpoint_dir=ckpt_dir if experiment_id else None,
                                         cooldown_gate_dir=(
                                             os.path.join(experiment_root, experiment_id,
                                                          "server", "round_cooldown")
                                             if round_cooldown_enabled else ""
                                         ),
                                         resume_snapshot=resume_snapshot,
                                         explicit_round_numbers=bool(policy_resume_round)),
    )
    logger.info("Saving final model to disk...")
    try:
        # Always save the final federated state (adapter-only for LoRA, full state dict otherwise)
        final_state = result.arrays.to_torch_state_dict()
        torch.save(final_state, ckpt_path)
        logger.info(f"[Server] Final model checkpoint saved to {ckpt_path}")
    except Exception as e:
        logger.error(f"[Server] Failed to save final checkpoint to {ckpt_path}: {e}")

def make_global_evaluate(
    model_name: str,
    dataset_name: str,
    eval_bs: int,
    total_round: int = 1,
    save_every_round: int = 1,
    lora_cfg: dict | None = None,
    grad_ckpt: bool | None = None,
    quantization: int | None = None,
    checkpoint_dir: str | None = None,
    cooldown_gate_dir: str = "",
    resume_snapshot=None,
    explicit_round_numbers: bool = False,
):
    """Return a server-side hook that checkpoints LoRA adapters.

    LoRACast's registration build does not perform centralized dataset
    evaluation.  Validation used in the research pipeline can remain an
    external evaluation step.
    """
    del dataset_name, eval_bs
    if not is_llm_model_name(model_name):
        raise ValueError(f"LoRACast requires an LLM model: {model_name!r}")

    safe_name = model_name.replace("/", "_").replace(":", "_")
    eval_dir = checkpoint_dir or f"{_LOCAL_PATH}/eval"
    os.makedirs(eval_dir, exist_ok=True)
    round_offset = None

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        nonlocal round_offset

        if checkpoint_dir and "original-c-same-run-" in checkpoint_dir:
            event_path = Path(checkpoint_dir).parent / "server" / "same_run_server_phases.jsonl"
            event_path.parent.mkdir(parents=True, exist_ok=True)
            with event_path.open("a", buffering=1) as handle:
                handle.write(json.dumps({"round": int(server_round),
                    "event": "server_checkpoint_hook_enter",
                    "server_monotonic_s": time.monotonic(),
                    "server_unix_s": time.time()}) + "\n")

        if server_round == 0:
            if checkpoint_dir:
                checkpoint = os.path.join(eval_dir, f"peft_{safe_name}_0.pt")
                torch.save(arrays.to_torch_state_dict(), checkpoint)
                logger.info("[Server] Saved initial LoRA adapter checkpoint: %s", checkpoint)
            return MetricRecord()

        if server_round != total_round and server_round % save_every_round != 0:
            return MetricRecord()

        if round_offset is None and not explicit_round_numbers:
            existing_max = 0
            prefix = f"peft_{safe_name}_"
            for filename in os.listdir(eval_dir):
                if not (filename.startswith(prefix) and filename.endswith(".pt")):
                    continue
                suffix = filename[len(prefix):-3]
                try:
                    existing_max = max(existing_max, int(suffix))
                except ValueError:
                    continue
            round_offset = existing_max

        logical_round = int(server_round) if explicit_round_numbers else int(round_offset or 0) + int(server_round)
        model = get_model(
            model_name,
            lora_cfg=lora_cfg,
            gradient_checkpointing=bool(grad_ckpt),
            quantization=quantization,
        )
        set_peft_model_state_dict(model, arrays.to_torch_state_dict())
        adapter_state = get_peft_model_state_dict(model)
        checkpoint = os.path.join(
            eval_dir,
            f"peft_{safe_name}_{logical_round}.pt",
        )
        torch.save(adapter_state, checkpoint)
        logger.info("[Server] Saved LoRA adapter checkpoint: %s", checkpoint)
        if resume_snapshot is not None:
            resume_snapshot(logical_round, checkpoint)
        if checkpoint_dir and "original-c-same-run-" in checkpoint_dir:
            event_path = Path(checkpoint_dir).parent / "server" / "same_run_server_phases.jsonl"
            with event_path.open("a", buffering=1) as handle:
                handle.write(json.dumps({"round": int(server_round),
                    "event": "server_checkpoint_saved",
                    "server_monotonic_s": time.monotonic(),
                    "server_unix_s": time.time()}) + "\n")
        wait_after_round(server_round, total_round, cooldown_gate_dir)
        return MetricRecord()

    return global_evaluate
