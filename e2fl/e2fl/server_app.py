"""E2FL: A Flower / PyTorch app."""
from log import WrlsEnv
from datetime import datetime
import subprocess, os, logging, time, socket, pickle
import paramiko, yaml
import re
import argparse
from typing import List, Tuple

import flwr as fl
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from e2fl.task import Net, get_weights, load_data, set_weights, test, train, get_num_classes, get_model, get_dataset

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

def create_strategy(context: Context, parameters):
    """Create a strategy based on the configuration in context."""
    strategy_name = context.run_config.get("strategy", "FedAvg")  # 기본값은 FedAvg

    fraction_fit = context.run_config["fraction-fit"]
    min_clients = context.run_config["min-clients"]

    if strategy_name == "FedAvg":
        from flwr.server.strategy import FedAvg
        return FedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            min_available_clients=min_clients,
            initial_parameters=parameters,
        )
    elif strategy_name == "FedProx":
        from flwr.server.strategy import FedProx
        return FedProx(
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            min_available_clients=min_clients,
            initial_parameters=parameters,
            mu=context.run_config.get("mu", 0.1),  # FedProx-specific parameter
        )
    elif strategy_name == "QFedAvg":
        from flwr.server.strategy import QFedAvg
        return QFedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            min_available_clients=min_clients,
            initial_parameters=parameters,
            q=context.run_config.get("q", 0.1),  # QFedAvg-specific parameter
        )
    elif strategy_name == "FedAdam":
        from flwr.server.strategy import FedAdam
        return FedAdam(
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            min_available_clients=min_clients,
            initial_parameters=parameters,
            eta=context.run_config.get("eta", 0.01),  # FedAdam-specific parameter
            beta_1=context.run_config.get("beta_1", 0.9),
            beta_2=context.run_config.get("beta_2", 0.999),
            epsilon=context.run_config.get("epsilon", 1e-8),
        )
    elif strategy_name == "FedYogi":
        from flwr.server.strategy import FedYogi
        return FedYogi(
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            min_available_clients=min_clients,
            initial_parameters=parameters,
            eta=context.run_config.get("eta", 0.01),  # FedYogi-specific parameter
            beta_1=context.run_config.get("beta_1", 0.9),
            beta_2=context.run_config.get("beta_2", 0.999),
            epsilon=context.run_config.get("epsilon", 1e-8),
        )
    elif strategy_name == "FedOpt":
        from flwr.server.strategy import FedOpt
        return FedOpt(
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            min_available_clients=min_clients,
            initial_parameters=parameters,
            eta=context.run_config.get("eta", 0.01),  # FedOpt-specific parameter
            beta_1=context.run_config.get("beta_1", 0.9),
            beta_2=context.run_config.get("beta_2", 0.999),
            epsilon=context.run_config.get("epsilon", 1e-8),
        )
    else:
        raise ValueError(f"Unsupported strategy: {strategy_name}")


def server_fn(context: Context):
    """Server function to create server components."""
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    dataset_name = context.node_config["dataset"]

    # Initialize model parameters
    model_name = context.node_config["model"]
    num_classes = get_num_classes(dataset_name)
    net = get_model(model_name, num_classes)
    ndarrays = get_weights(net)
    parameters = ndarrays_to_parameters(ndarrays)

    # Create strategy using the helper function
    strategy = create_strategy(context, parameters)

    # Create server configuration
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

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
# Create ServerApp
app = ServerApp(server_fn=server_fn)
