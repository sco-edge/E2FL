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

from core.task import Net, get_weights



parser = argparse.ArgumentParser(description="Flower Embedded devices")
parser.add_argument(
    "--server_address",
    type=str,
    default="0.0.0.0:8080",
    help=f"gRPC server address (deafault '0.0.0.0:8080')",
)
parser.add_argument(
    "--rounds",
    type=int,
    default=5,
    help="Number of rounds of federated learning (default: 5)",
)
parser.add_argument(
    "--sample_fraction",
    type=float,
    default=1.0,
    help="Fraction of available clients used for fit/evaluate (default: 1.0)",
)
parser.add_argument(
    "--min_num_clients",
    type=int,
    default=2,
    help="Minimum number of available clients required for sampling (default: 2)",
)
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

def get_ip_address():
    try:
        hostname = socket.gethostname()
        ip_addr = socket.gethostbyname(hostname)
        return ip_addr
    except socket.error as e:
        print(f'Unable to get IP address: {e}')
        return None

def change_WiFi_interface(interf = 'wlan0', channel = 11, rate = '11M', txpower = 15):
    # change Wi-Fi interface 
    result = subprocess.run([f"iwconfig {interf} channel {channel} rate {rate} txpower {txpower}"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    return result

def change_WiFi_interface_client(client_ssh, interf = 'wlan0', channel = 11, rate = '11M', txpower = 15):
    # change Wi-Fi interface 
    stdin, stdout, stderr = client_ssh.send(f"iwconfig {interf} channel {channel} rate {rate} txpower {txpower}") # exec_command
    time.sleep(2)
    
    # Receive the output
    output = client_ssh.recv(65535).decode() # 65535 is the maximum bytes that can read by recv() method.

    if 'error' in output.lower() or 'command not found' in output.lower():
            logger.error("Error detected in the command output.")
            return False
    else:
        logger.info("iwconfig executed successfully.")
        logger.info(output)
    
    return True

def get_client_SSH(client_ip, wait_time):
    # Wait for boot up
    print(f"Wait {wait_time} seconds for the edge device to boot.")

    # Set up SSH service
    client_SSH = paramiko.SSHClient()
    client_SSH.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # Add the host key automatically AutoAddPolicy()
    mykey = paramiko.RSAKey.from_private_key_file(private_key_path)

    # ssh -i {rsa} {USER}@{IP_ADDRESS}
    try_count = 0
    start_time = time.time()
    while 1:
        print(f'{try_count} try ...')
        try:
            client_SSH.connect(hostname = client_ip, port = ssh_port, username = client_ssh_id, passphrase="", pkey=mykey, look_for_keys=False)
            break
        except:
            try_count = try_count + 1
            pass
        time.sleep(10)
        if time.time() - start_time > _UPTIME_RPI3B:
            try:
                client_SSH.connect(hostname = client_ip, port = ssh_port, username = client_ssh_id, passphrase="", pkey=mykey, look_for_keys=False)
            except Exception as e:
                logger.error("SSH is failed: ", e)
                logger.error(private_key_path)
                exit(1)

    if client_SSH.get_transport() is not None and client_SSH.get_transport().is_active():
        logger.info('An SSH connection for the client is established.')
        # Enable Keep Alive
        client_SSH.get_transport().set_keepalive(30)
        logger.debug("Set the client_SSH's keepalive option.")
    else:
        logger.debug("client_SSH is closed. exit()")
        exit()

    client_shell = client_SSH.invoke_shell()

    return client_shell

# FL; define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def server_fn(context: Context):
    # Configure the server
    num_rounds = context.run_config["num-server-rounds"]

    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)
    
    # Define the strategy
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=context.run_config["fraction-evaluate"],
        min_available_clients=context.run_config["min-clients"],
        evaluate_metrics_aggregation_fn = weighted_average,
        initial_parameters=parameters,
        #on_fit_config_fn=lambda rnd: {"round": rnd},
    )
    config = ServerConfig(num_rounds=round)
    
    ServerAppComponents(strategy=strategy, config=config)

if __name__ == "__main__":
    # default parameters
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
        logger.info("Start FL server.")
        app = ServerApp(server_fn=server_fn)
        # Wait for server to start fl properly.
        time.sleep(5)
    except Exception as e:
        logger.error('FL is failed: ', e)
        exit(1)
