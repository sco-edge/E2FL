from log import WrlsEnv
from datetime import datetime
import subprocess, os, logging, time, socket, pickle
import paramiko, yaml
import re
import argparse
from typing import List, Tuple

import argparse
from typing import List, Tuple
import flwr as fl
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from E2FL.task import Net, get_weights


_UPTIME_RPI3B = 500

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
    
# FL; define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def server_fn(context: Context):
    # Configure the server
    num_rounds = context.run_config["num-server-rounds"]  # Default to 5 if the key is not present
    fraction_fit = context.run_config["fraction-fit"]

    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)
    
    # Define the strategy
    strategy = FedAvg(
        fraction_fit = fraction_fit,
        fraction_evaluate = context.run_config["fraction-evaluate"],
        min_available_clients = context.run_config["min-clients"],
        evaluate_metrics_aggregation_fn = weighted_average,
        initial_parameters = parameters,
        #on_fit_config_fn=lambda rnd: {"round": rnd},
    )
    config = ServerConfig(num_rounds = num_rounds)
    
    return ServerAppComponents(strategy=strategy, config=config)

if __name__ == "__main__":
    root_path = os.path.abspath(os.getcwd())+'/'
    node_A_name = 'RPi3B+'
    node_A_mode = "PyMonsoon"
    client_ssh_id = 'pi'
    ssh_port = 22

    # Set up logger
    logger = logging.getLogger("test")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    fl.common.logger.configure(identifier="myFlowerExperiment", filename=f"fl_log_server_{current_time}.txt")

    # Read the YAML config file.
    with open(root_path+'config.yaml', 'r') as file:
        config = yaml.safe_load(file)  # Read YAML from the file and convert the structure of the python's dictionary.

    # Prepare a bucket to store the results.
    measurements_dict = []

    WiFi_rates = [1] #[1, 2, 5.5, 11, 6, 9, 12, 18, 24, 36, 48, 54]

    # Start the FL server.
    try:
        app = ServerApp(server_fn=server_fn)
        logger.info("Start FL server.")
        # Wait for server to start fl properly.
        time.sleep(5)
    except Exception as e:
        logger.error(f"FL is failed: {e}")
        exit(1)
