"""E2FL: A Flower / PyTorch app."""
from log import WrlsEnv
from datetime import datetime
import subprocess, os, logging, time, socket, pickle
import paramiko, yaml, re, argparse, json
from typing import List, Tuple

import torch
import flwr as fl
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from e2fl.task_llm import test, train, is_lora_model
from e2fl.models import get_model
from e2fl.dataset import load_data, get_num_classes, is_llm_model_name
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
_LOCAL_PATH = f"{home_dir}/EEFL/E2FL"

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

'''
strategy

https://flower.ai/docs/baselines/how-to-use-baselines.html
https://github.com/adap/flower/tree/v1.17.0/src/py/flwr/server/strategy
#https://github.com/adap/flower/blob/main/framework/docs/source/tutorial-series-use-a-federated-learning-strategy-pytorch.ipynb
'''

def create_strategy(context: Context):
    """Create a strategy based on the configuration in context."""
    strategy_name = context.run_config.get("strategy", "FedAvg")  # 기본값은 FedAvg

    fraction_train = context.run_config["fraction-train"]
    min_clients = context.run_config["min-clients"]

    if strategy_name == "FedAvg":
        from flwr.serverapp.strategy import FedAvg
        return FedAvg(
            fraction_train=fraction_train,
            fraction_evaluate=1.0,
            min_available_nodes=min_clients,
        )
    elif strategy_name == "FedProx":
        from flwr.serverapp.strategy import FedProx
        return FedProx(
            fraction_train=fraction_train,
            fraction_evaluate=1.0,
            min_available_nodes=min_clients,
            mu=context.run_config.get("mu", 0.1),  # FedProx-specific parameter
        )
    elif strategy_name == "QFedAvg":
        from flwr.serverapp.strategy import QFedAvg
        return QFedAvg(
            fraction_train=fraction_train,
            fraction_evaluate=1.0,
            min_available_nodes=min_clients,
            q=context.run_config.get("q", 0.1),  # QFedAvg-specific parameter
        )
    elif strategy_name == "FedAdam":
        from flwr.serverapp.strategy import FedAdam
        return FedAdam(
            fraction_train=fraction_train,
            fraction_evaluate=1.0,
            min_available_nodes=min_clients,
            eta=context.run_config.get("eta", 0.01),  # FedAdam-specific parameter
            beta_1=context.run_config.get("beta_1", 0.9),
            beta_2=context.run_config.get("beta_2", 0.999),
            epsilon=context.run_config.get("epsilon", 1e-8),
        )
    elif strategy_name == "FedYogi":
        from flwr.serverapp.strategy import FedYogi
        return FedYogi(
            fraction_train=fraction_train,
            fraction_evaluate=1.0,
            min_available_nodes=min_clients,
            eta=context.run_config.get("eta", 0.01),  # FedYogi-specific parameter
            beta_1=context.run_config.get("beta_1", 0.9),
            beta_2=context.run_config.get("beta_2", 0.999),
            epsilon=context.run_config.get("epsilon", 1e-8),
        )
    elif strategy_name == "FedOpt":
        from flwr.serverapp.strategy import FedOpt
        return FedOpt(
            fraction_train=fraction_train,
            fraction_evaluate=1.0,
            min_available_nodes=min_clients,
            eta=context.run_config.get("eta", 0.01),  # FedOpt-specific parameter
            beta_1=context.run_config.get("beta_1", 0.9),
            beta_2=context.run_config.get("beta_2", 0.999),
            epsilon=context.run_config.get("epsilon", 1e-8),
        )
    elif strategy_name == "FedAvgLLM":
        from e2fl.strategy.FedAvgLLM import FedAvgLLM

        model_name = context.run_config.get("model", "unknown_model")
        if context.run_config.get("enable-latte", True):
            latte_mode = context.run_config.get("latte-mode", "coarse")
            
            profile_root = os.path.join(f"{_LOCAL_PATH}/predictor/profile")
            model_profile_path = f"{profile_root}/{model_name}_workload_{latte_mode}.json"

            with open(model_profile_path, "r") as f:
                workload = json.load(f)

            if "layers" not in workload:
                raise ValueError("Invalid algo_selection JSON: missing 'layers' section")

            model_info = {
                "algo_selection": workload["algo_selection"],
                "C_key": workload["C_key"],
                "C_non": workload["C_non"],
                "num_epochs": int(cfg(context, "local-epochs", 1)),
                "batch_size": int(cfg(context, "batch_size", 32)),
            }
        else:
            model_info = None
        
        latte_mode = context.run_config.get("latte-mode", "coarse")
        if latte_mode == "fine":
            model_info = None # Fine-grained mode does not use model_info

        return FedAvgLLM(
            fraction_train=fraction_train,
            fraction_evaluate=1.0,
            min_available_nodes=min_clients,
            log_dir="eval/logs_fedavg",
            model_name=model_name,
            latte=context.run_config.get("enable-latte", True),
            model_info=model_info,
            device_lookup=device_lookup,
            ROOT=_LOCAL_PATH,
        )
    elif strategy_name == "FedAvgLoRA":
        from e2fl.strategy.FedAvgLoRA import FedAvgLoRA

        model_name = context.run_config.get("model", "unknown_model")

        # Optional toggles for strategy behavior
        enable_rpi_only_rounds = bool(context.run_config.get("enable-rpi-only-rounds", True))
        use_token_ratio = bool(context.run_config.get("use-token-ratio", True))
        enable_deadline_guard = bool(context.run_config.get("enable-deadline-guard", False))

        # Optional: device profiling path and tokens-per-example (for cost prediction)
        device_profiles_path = context.run_config.get("device-profiles-path", None)
        tokens_per_example = float(cfg(context, "tokens-per-example", 256))

        return FedAvgLoRA(
            fraction_train=fraction_train,
            fraction_evaluate=1.0,
            min_available_nodes=min_clients,
            log_dir="eval/logs_fedavg",
            model_name=model_name,
            latte=False,
            model_info=None,
            device_lookup=device_lookup,
            ROOT=_LOCAL_PATH,
            device_profiles_path=device_profiles_path,
            tokens_per_example=tokens_per_example,
            enable_rpi_only_rounds=enable_rpi_only_rounds,
            use_token_ratio=use_token_ratio,
            enable_deadline_guard=enable_deadline_guard,
        )
    elif strategy_name == "EEFL":
        from e2fl.strategy.EEFL import EEFL
        return EEFL(
            fraction_fit=fraction_train,       # ex. 0.3
            fraction_evaluate=1.0,
            min_fit_clients=min_clients,       # ex. 2
            min_evaluate_clients=min_clients,
            min_available_clients=min_clients,
            base_H=4,
            delta_H=1,
            alpha=0.3,
            Lambda_th=5e-4, #{1e-4, 5e-4, 1e-3}
        )
    elif strategy_name == "E2FL":
        from e2fl.strategy.E2FL import E2FL
        return E2FL(
            fraction_train=fraction_train,
            fraction_evaluate=1.0,
            min_available_nodes=min_clients,
        )
    else:
        raise ValueError(f"Unsupported strategy: {strategy_name}")

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
    local_epochs         = int(cfg(context, "local-epochs", 1))
    max_samples_per_round = int(cfg(context, "max-samples-per-round", 0))  # 0이면 제한 없음
    max_seq_len          = int(cfg(context, "max-seq-len", 256))

    # Initialize model parameters
    model_name = context.run_config["model"]
    # LLM (e.g., Gemma 3 270M) does not need num_classes; pass a dummy value
    if is_llm_model_name(model_name):
        num_classes = 0
    else:
        num_classes = get_num_classes(dataset_name)

    # Load global model
    global_model = get_model(model_name, num_classes, dataset_name, lora_cfg, grad_ckpt, quantization)

    # Create checkpoint filename using model name, dataset, and LoRA/full mode
    safe_model = model_name.replace("/", "_").replace(":", "_")
    safe_dataset = dataset_name.replace("/", "_")
    mode_tag = "adapter" if is_lora_model(global_model) else "full"

    # Use configurable checkpoint root if provided, otherwise default to _LOCAL_PATH/eval
    ckpt_root = cfg(context, "checkpoint-path", f"{_LOCAL_PATH}/eval")
    ckpt_dir = os.path.expanduser(ckpt_root)
    os.makedirs(ckpt_dir, exist_ok=True)
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
        "lr": lr_max,
        "weight_decay": wd,
        "batch_size": bs_train,
        "local_epochs": local_epochs,
        "max_samples_per_round": max_samples_per_round,
        "max_seq_len": max_seq_len,
    }

    # Start strategy, run for 'num_rounds'
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord(train_conf),
        num_rounds=int(cfg(context, "num-server-rounds", 1)),
        evaluate_fn=make_global_evaluate(model_name, dataset_name, 
                                         bs_eval, num_rounds, save_every_round, 
                                         lora_cfg, grad_ckpt, quantization),
    )
    logger.info("Saving final model to disk...")
    try:
        # Always save the final federated state (adapter-only for LoRA, full state dict otherwise)
        final_state = result.arrays.to_torch_state_dict()
        torch.save(final_state, ckpt_path)
        logger.info(f"[Server] Final model checkpoint saved to {ckpt_path}")
    except Exception as e:
        logger.error(f"[Server] Failed to save final checkpoint to {ckpt_path}: {e}")

def make_global_evaluate(model_name: str, dataset_name: str, eval_bs: int, 
                         total_round: int = 1, save_every_round: int = 1,
                         lora_cfg: dict | None = None, grad_ckpt: bool | None = None, quantization: int | None = None):
    """Factory returning an evaluate function bound to the given model/dataset.

    The returned function has the signature expected by Flower's strategy
    (server_round: int, arrays: ArrayRecord) -> MetricRecord.
    """
    is_llm = is_llm_model_name(model_name)
    if is_llm:
        num_classes = 0
    else:
        num_classes = get_num_classes(dataset_name)
    
    safe_name = model_name.replace("/", "_").replace(":", "_")
    eval_dir = f"{_LOCAL_PATH}/eval"
    os.makedirs(eval_dir, exist_ok=True)

    round_offset = None


    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        nonlocal round_offset
        if is_llm:
            logger.info("[Server] LLM detected → skipping global evaluation dataset loading")

            if server_round != 0 and (
                server_round == total_round or server_round % save_every_round == 0
            ):
                # Lazily compute round_offset by scanning existing per-round checkpoints
                if round_offset is None:
                    existing_max = 0
                    for fname in os.listdir(eval_dir):
                        if not fname.startswith(f"peft_{safe_name}_") or not fname.endswith(".pt"):
                            continue
                        try:
                            num_str = fname.split("_")[-1].split(".")[0]
                            num = int(num_str)
                            if num > existing_max:
                                existing_max = num
                        except ValueError:
                            continue
                    round_offset = existing_max

                logical_round = (round_offset or 0) + server_round

                model = get_model(model_name, num_classes, dataset_name, lora_cfg, grad_ckpt, quantization)
                set_peft_model_state_dict(model, arrays.to_torch_state_dict())

                adapter_sd = get_peft_model_state_dict(model)
                ckpt_file = os.path.join(eval_dir, f"peft_{safe_name}_{logical_round}.pt")
                torch.save(adapter_sd, ckpt_file)
            return MetricRecord()
        else:             
            # Build model with the same architecture used to initialize training
            logger.info(f"[Server] starts global evaluation.")
            model = get_model(model_name, num_classes, dataset_name, lora_cfg, grad_ckpt, quantization)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model.to(device)

            # Load entire test set by using load_data with num_partitions=1 (centralized)
            # load_data returns (trainloader, valloader) in this codebase; use valloader as testset
            _, test_dataloader = load_data(
                dataset_name=dataset_name,
                partition_id=0,
                num_partitions=1,
                batch_size=eval_bs,
            )

            # Evaluate the global model on the test set using the project's test() function
            test_loss, test_acc = test(model, test_dataloader, device)
            return MetricRecord({"accuracy": test_acc, "loss": test_loss})

    return global_evaluate
