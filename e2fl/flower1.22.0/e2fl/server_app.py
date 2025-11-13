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
from flwr.common import ndarrays_to_parameters
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from e2fl.task import Net, get_weights, load_data, set_weights, test, train, get_num_classes, get_model

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
    else:
        raise ValueError(f"Unsupported strategy: {strategy_name}")

# Create ServerApp (after Flower 1.21.0)
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    num_rounds: int = context.run_config["num-server-rounds"]
    dataset_name = context.run_config["dataset"]
    fraction_train: float = context.run_config["fraction-train"]

    # Initialize model parameters
    model_name = context.run_config["model"]
    num_classes = get_num_classes(dataset_name)
    
    # Load global model
    global_model = get_model(model_name, num_classes, dataset_name)
    arrays = ArrayRecord(global_model.state_dict())

    # Instantiate strategy
    strategy = create_strategy(context)
    evaluate_fn = make_global_evaluate(model_name, dataset_name)

    # Start strategy, run for 'num_rounds'
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": 0.01}),
        num_rounds=num_rounds,
        evaluate_fn=evaluate_fn,
    )
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")

def make_global_evaluate(model_name: str, dataset_name: str):
    """Factory returning an evaluate function bound to the given model/dataset.

    The returned function has the signature expected by Flower's strategy
    (server_round: int, arrays: ArrayRecord) -> MetricRecord.
    """
    num_classes = get_num_classes(dataset_name)

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        # Build model with the same architecture used to initialize training
        model = get_model(model_name, num_classes, dataset_name)
        model.load_state_dict(arrays.to_torch_state_dict())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Load entire test set by using load_data with num_partitions=1 (centralized)
        # load_data returns (trainloader, valloader) in this codebase; use valloader as testset
        _, test_dataloader = load_data(
            dataset_name=dataset_name,
            partition_id=0,
            num_partitions=1,
            batch_size=128,
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
