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

def fl_server(server_address, num_clients=4, sample_frac=1.0, round=3):
    num_rounds = context.run_config["num-sever-rounds"]

if __name__ == "__main__":
    # default parameters
    root_path = os.path.abspath(os.getcwd())+'/'
    node_A_name = 'RPi3B+'
    node_A_mode = "PyMonsoon"
    client_ssh_id = 'pi'
    ssh_port = 22
    args = parser.parse_args()
    print(args)

    # FL parameters
    FL_num_clients = 4
    FL_sample_frac = 1.0
    FL_round = 3

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

    # Get IP addresses.
    server_ip = config['server']['host']  # Extract 'host' key's value from the 'server' key
    client_ip1 = config['Jetson']['host']
    client_ip2 = config['RPi5_1']['host']
    client_ip3 = config['RPi5_2']['host']
    client_ip4 = config['RPi5_3']['host']
    private_key_path = root_path + config['RPi3B+']['ssh_key']
    client_interf = config['RPi3B+']['interface']

    if server_ip or client_ip1:
        print(f"The IP address of the server is: {server_ip}")
        print(f"The IP address of the client is: {client_ip1}, {client_ip2}, {client_ip3}, {client_ip4}")
        print(f"The ID of the client is: {client_ssh_id}")
    else:
        print("IP address could not be determined")
        exit(1)


    # set client_SSH
    #client_shells = []
    #for c_ip in [client_ip1, client_ip2, client_ip3, client_ip4]:
    #    client_shells.append(get_client_SSH(client_ip = c_ip, wait_time = _UPTIME_RPI3B))


    # Prepare a bucket to store the results.
    measurements_dict = []

    WiFi_rates = [1] #[1, 2, 5.5, 11, 6, 9, 12, 18, 24, 36, 48, 54]

    # Start the FL server.
    try:
        fl_server(server_address=args.server_address, \
                num_clients=args.min_num_clients, sample_frac=args.sample_fraction, round=args.rounds)
        logger.info("Start FL server.")
        # Wait for server to start fl properly.
        time.sleep(5)
    except Exception as e:
        logger.error('FL is failed: ', e)
        exit(1)


    # Log the end time.
    #logger.info([f'Wi-Fi end(rate: {rate})',time.time()])

    # measurements_dict.append({'rate': rate, 'time': time_records, 'power': samples})

    # Close the SSH connection.
    #for client_SSH in client_shells:
    #    client_SSH.close()


    # Save the data.
    #current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    #filename = f"data_{current_time}.pickle"
    #with open(filename, 'wb') as handle:
    #    pickle.dump(measurements_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #logger.info(f"The measurement data is saved as {filename}.")

    '''

    import grpc
    import logging
    import time

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    class LoggingInterceptor(grpc.ServerInterceptor):
        def intercept_service(self, continuation, handler_call_details):
            method = handler_call_details.method
            def log_and_continue(request, context):
                start_time = time.time()
                response = continuation(request, context)
                end_time = time.time()
                logger.info(f"Method: {method}, Start: {start_time}, End: {end_time}, Duration: {end_time - start_time}")
                return response
            return log_and_continue
    def serve():
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), interceptors=[LoggingInterceptor()])
        
        server.add_insecure_port('[::]:8080')
        server.start()
        server.wait_for_termination()
    '''