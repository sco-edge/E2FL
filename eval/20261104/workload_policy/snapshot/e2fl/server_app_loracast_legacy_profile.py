"""LoRACast Flower ServerApp for federated LLM fine-tuning."""
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
    if experiment_id and device_profiles_path and not os.path.isabs(device_profiles_path):
        device_profiles_path = os.path.join(_LOCAL_PATH, device_profiles_path)
    manifest_path = str(cfg(context, "partition-manifest-path", "") or "")
    if manifest_path and not os.path.isabs(manifest_path):
        manifest_path = os.path.join(_LOCAL_PATH, manifest_path)
    manifest_sha256 = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest() if manifest_path else ""
    manifest_client_count = int(json.loads(Path(manifest_path).read_text())["num_clients"]) if manifest_path else 0

    strategy_name = str(cfg(context, "strategy", "LoRACastCommAware"))
    strategies = {
        "LoRACastCommAware": LoRACastCommAware,
        "FedAvgLoRA": FedAvgLoRA,
        "PyramidFLLoRA_NoDropout": PyramidFLLoRA_NoDropout,
    }
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
    if strategy_name == "LoRACastCommAware":
        strategy_kwargs.update(
            comm_ema_alpha=float(cfg(context, "comm-ema-alpha", 0.3)),
            comm_safety_factor=float(cfg(context, "comm-safety-factor", 1.0)),
            use_comm_prediction=_config_bool(cfg(context, "use-comm-prediction", True)),
            static_allocation_after_round=int(cfg(context, "static-allocation-after-round", 0)),
        )
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
    os.makedirs(ckpt_dir, exist_ok=True)
    if experiment_id and cfg(context, "partition-manifest-path", ""):
        manifest_path = Path(str(cfg(context, "partition-manifest-path", "")))
        if not manifest_path.is_absolute():
            manifest_path = Path(_LOCAL_PATH) / manifest_path
        manifest_snapshot = Path(experiment_root) / experiment_id / "server" / "partition_manifest.json"
        manifest_snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(manifest_path, manifest_snapshot)
    ckpt_name = f"{safe_model}_{safe_dataset}_{mode_tag}.pt"
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)

    if resume and os.path.exists(ckpt_path):
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

    # Instantiate strategy
    strategy = create_strategy(context)
    train_conf = {
        # Standard training configuration
        "lr": lr_max,
        "weight_decay": wd,
        "batch_size": bs_train,
        "local_epochs": local_epochs,
        "max_samples_per_round": max_samples_per_round,
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
                                         )),
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

        if server_round == 0:
            if checkpoint_dir:
                checkpoint = os.path.join(eval_dir, f"peft_{safe_name}_0.pt")
                torch.save(arrays.to_torch_state_dict(), checkpoint)
                logger.info("[Server] Saved initial LoRA adapter checkpoint: %s", checkpoint)
            return MetricRecord()

        if server_round != total_round and server_round % save_every_round != 0:
            return MetricRecord()

        if round_offset is None:
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

        logical_round = int(round_offset or 0) + int(server_round)
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
        wait_after_round(server_round, total_round, cooldown_gate_dir)
        return MetricRecord()

    return global_evaluate
