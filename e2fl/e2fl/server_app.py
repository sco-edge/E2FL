"""E2FL: A Flower / PyTorch app."""
from log import WrlsEnv
from datetime import datetime
import subprocess, os, logging, time, socket, pickle
import paramiko, yaml
import re
import argparse
from typing import List, Tuple

import torch
import flwr as fl
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from e2fl.task import test, train, is_lora_model
from e2fl.models import get_model
from e2fl.dataset import load_data, get_num_classes
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

_LOCAL_PATH = "./eval/"

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
    fl.common.logger.configure(identifier="e2fl_server", filename=_LOCAL_PATH+f"fl_server_{current_time}.txt")
except Exception:
    logger.debug("fl.common.logger.configure unavailable or failed; continuing without fl file logger")

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
        return FedAvgLog(
            fraction_train=fraction_train,
            fraction_evaluate=1.0,
            min_available_nodes=min_clients,
            log_dir=_LOCAL_PATH+"logs_fedavg",
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
    lr           = float(cfg(context, "learning-rate", 1e-2))
    wd           = float(cfg(context, "weight-decay", 5e-4))
    resume       = bool(cfg(context, "resume", True))
    ckpt_path    = cfg(context, "checkpoint-path", None)
    seq_len      = int(cfg(context, "seq-len", 0)) or None
    lora_cfg     = lora_cfg_from(context)

    # Initialize model parameters
    model_name = context.run_config["model"]
    num_classes = get_num_classes(dataset_name)
    
    # Load global model
    global_model = get_model(model_name, num_classes, dataset_name, lora_cfg, seq_len)

    ckpt_path = _LOCAL_PATH + (
        "final_adapter.pt" if is_lora_model(global_model) else "final_model.pt"
    )
    
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
    evaluate_fn = make_global_evaluate(model_name, dataset_name, bs_eval, lora_cfg, seq_len)

    train_conf = {
        "lr": lr,
        "weight_decay": wd,
        "batch_size": bs_train,
        "seq_len": seq_len or 0,
        "lr_scheduler": cfg(context, "lr-scheduler", "none"),
        "warmup_steps": int(cfg(context, "warmup-steps", 0)),
        "lora": lora_cfg,
    }

    # Start strategy, run for 'num_rounds'
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord(train_conf),
        num_rounds=int(cfg(context, "num-server-rounds", 1)),
        evaluate_fn=evaluate_fn,
    )
    logger.info("Saving final model to disk...")
    if is_lora_model(global_model):
        torch.save(get_peft_model_state_dict(global_model), _LOCAL_PATH+"final_adapter.pt")
    else:
        torch.save(result.arrays.to_torch_state_dict(), _LOCAL_PATH+"final_model.pt")

def make_global_evaluate(model_name: str, dataset_name: str,
                         eval_bs: int, lora_cfg: dict | None, seq_len: int | None):
    """Factory returning an evaluate function bound to the given model/dataset.

    The returned function has the signature expected by Flower's strategy
    (server_round: int, arrays: ArrayRecord) -> MetricRecord.
    """
    num_classes = get_num_classes(dataset_name)

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        # Build model with the same architecture used to initialize training
        model = get_model(model_name, num_classes, dataset_name, lora_cfg, seq_len)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        if is_lora_model(model):
            logger.info(f"[Server] Loading LoRA adapter weights for round {server_round}")
            lora_arrays = arrays.as_numpy()
            sd_keys = list(get_peft_model_state_dict(model).keys())
            adapter_sd = {k: torch.tensor(w).cpu() for k, w in zip(sd_keys, lora_arrays)}
            set_peft_model_state_dict(model, adapter_sd)
        else:
            model.load_state_dict(arrays.to_torch_state_dict())

        # Load entire test set by using load_data with num_partitions=1 (centralized)
        # load_data returns (trainloader, valloader) in this codebase; use valloader as testset
        _, test_dataloader = load_data(
            dataset_name=dataset_name,
            partition_id=0,
            num_partitions=1,
            batch_size=eval_bs,   
            mode="eval",
            seq_len=seq_len,
        )

        # Evaluate the global model on the test set using the project's test() function
        test_loss, test_acc = test(model, test_dataloader, device)

        # Return the evaluation metrics
        return MetricRecord({"accuracy": test_acc, "loss": test_loss})

    return global_evaluate

'''
logger = logging.getLogger("test")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
fl.common.logger.configure(identifier="myFlowerExperiment", filename=f"fl_log_server_{current_time}.txt")
'''
