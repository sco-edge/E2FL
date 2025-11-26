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
from e2fl.task import test, train, is_lora_model
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
    elif strategy_name == "FedAvgLog":
        from e2fl.strategy.FedAvgLog import FedAvgLog

        if context.run_config.get("enable-latte", True):
            model_name = context.run_config.get("model", "unknown_model")
            
            profile_root = os.path.join(f"{_LOCAL_PATH}/predictor/profile")
            algo_path = f"{profile_root}/{model_name}_algo_selection.json"
            model_profile_path = f"{profile_root}/{model_name}_workload.json"
            with open(model_profile_path, "r") as f:
                workload = json.load(f)
            with open(algo_path, "r") as f:
                algo_sel = json.load(f)
            model_info = {
                "C_key": workload["C_key"],
                "C_non": workload["C_non"],
                "algo_selection": algo_sel["layers"]
            }
        else:
            algo_selection, model_info = None, None
            

        return FedAvgLog(
            fraction_train=fraction_train,
            fraction_evaluate=1.0,
            min_available_nodes=min_clients,
            log_dir="eval/logs_fedavg",
            model_name=model_name,
            latte=context.run_config.get("enable-latte", True),
            algo_selection=algo_sel["layers"],
            model_info=model_info,
            device_lookup=device_lookup,
            ROOT=_LOCAL_PATH,
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
    num_classes = get_num_classes(dataset_name)
    num_rounds = int(cfg(context, "num-server-rounds", 1))
    save_every_round = int(cfg(context, "save-every-round", 1))

    # Initialize model parameters
    model_name = context.run_config["model"]
    
    # Load global model
    global_model = get_model(model_name, num_classes, dataset_name, lora_cfg, grad_ckpt, quantization)

    # Create checkpoint filename using model name, dataset, and LoRA/full mode
    model_tag = model_name.replace("/", "_")
    mode_tag = "adapter" if is_lora_model(global_model) else "full"

    ckpt_name = f"{model_tag}_{dataset_name}_{mode_tag}.pt"
    ckpt_path = os.path.join(_LOCAL_PATH+"/eval", ckpt_name)

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
    if is_lora_model(global_model):
        torch.save(get_peft_model_state_dict(global_model), ckpt_path)
    else:
        torch.save(result.arrays.to_torch_state_dict(), ckpt_path)

def make_global_evaluate(model_name: str, dataset_name: str, eval_bs: int, 
                         total_round: int = 1, save_every_round: int = 1,
                         lora_cfg: dict | None = None, grad_ckpt: bool | None = None, quantization: int | None = None):
    """Factory returning an evaluate function bound to the given model/dataset.

    The returned function has the signature expected by Flower's strategy
    (server_round: int, arrays: ArrayRecord) -> MetricRecord.
    """
    num_classes = get_num_classes(dataset_name)

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        if is_llm_model_name(model_name):
            logger.info("[Server] LLM detected → skipping global evaluation dataset loading")

            if server_round != 0 and (
                server_round == total_round or server_round % save_every_round == 0
            ):
                model = get_model(model_name, num_classes, dataset_name, lora_cfg, grad_ckpt, quantization)
                set_peft_model_state_dict(model, arrays.to_torch_state_dict())

                #model.save_pretrained(f"{_LOCAL_PATH}/eval/peft_{model_name}_{server_round}")
                adapter_sd = get_peft_model_state_dict(model)
                safe_name = model_name.replace("/", "_")
                torch.save(adapter_sd, f"{_LOCAL_PATH}/eval/peft_{safe_name}_{server_round}.pt")
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
