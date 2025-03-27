"""E2FL: A Flower / PyTorch app."""
from log import WrlsEnv
from power import _power_monitor_interface
from power.powermon import get_power_monitor

from datetime import datetime
from collections import OrderedDict
import argparse
import psutil
import warnings
import threading
import subprocess, os, logging, time, socket, pickle

import torch

import flwr as fl
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from e2fl.task import Net, get_weights, load_data, set_weights, test, train

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

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs

        interfaces = get_network_interface()
        if 'eth0' in interfaces:
            self.interface = 'eth0' if validate_network_interface('eth0') else "wlan0"
        elif 'wlp1s0' in interfaces:
            self.interface = 'wlp1s0' if validate_network_interface('wlp1s0') else "wlan0"
        else:
            self.interface = 'wlan0'

        self.start_net = self.get_network_usage(self.interface)
        self.end_net = None
    
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def get_network_usage(self):
        """Get the current network usage for the specified interface."""
        net_io = psutil.net_io_counters(pernic=True)
        if self.interface in net_io:
            return {"bytes_sent": net_io[self.interface].bytes_sent, "bytes_recv": net_io[self.interface].bytes_recv}
        else:
            raise ValueError(f"Interface {self.interface} not found.")

    def fit(self, parameters, config):
        logger.info(f"[{time.time()}] Client sampled for fit()")

        # 네트워크 사용량 기록
        self.end_net = self.get_network_usage()
        net_usage_sent = self.end_net["bytes_sent"] - self.start_net["bytes_sent"]
        net_usage_recv = self.end_net["bytes_recv"] - self.start_net["bytes_recv"]
        logger.info(f"Network usage during fit: [sent: {net_usage_sent}, recv: {net_usage_recv}]")

        set_weights(self.net, parameters)

        logger.info(f"[{time.time()}] Starting training...")
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
        )
        logger.info(f"[{time.time()}] Training completed.")

        # 학습 후 네트워크 상태 업데이트
        self.start_net = self.get_network_usage()

        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        logger.info(f"[{time.time()}] Client sampled for evaluate()")
        set_weights(self.net, parameters)

        logger.info(f"[{time.time()}] Starting evaluation...")
        loss, accuracy = test(self.net, self.valloader, self.device)
        logger.info(f"[{time.time()}] Evaluation completed with accuracy: {accuracy}")

        return loss, len(self.valloader.dataset), {"accuracy": accuracy}

def validate_network_interface(interface):
    """
    Validate if the given network interface exists on the system.
    :param interface: Network interface name to validate.
    :return: True if the interface exists, False otherwise.
    """
    if interface in psutil.net_if_addrs():
        return True
    else:
        logging.error(f"Invalid network interface: {interface}")
        return False

def get_network_interface():
    interfaces = psutil.net_if_addrs().keys()
    return list[interfaces]

def get_network_usage(interf):
    net_io = psutil.net_io_counters(pernic=True)
    #net_io = psutil.net_io_counters(pernic=True)
    return {"bytes_sent": net_io[interf].bytes_sent, "bytes_recv": net_io[interf].bytes_recv}

def client_fn(context: Context):
    # Load model and data
    net = Net()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    # Measure power consumption if a power monitor is initialized
    if power_monitor:
        logger.info("Starting power monitoring...")
        power_monitor.start(freq=1)  # Start monitoring with 1-second intervals
        time.sleep(10)  # Example duration for monitoring

    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()

interfaces = get_network_interface()
if 'eth0' in interfaces:
    wlan_interf = 'eth0' if validate_network_interface('eth0') else "wlan0"
    device_name = 'jetson'
    power = "INA3221"
elif 'wlp1s0' in interfaces:
    wlan_interf = 'wlp1s0' if validate_network_interface('wlp1s0') else "wlan0"
    device_name = 'RPi5'
    power = "PMIC"
else:
    wlan_interf = 'wlan0'
    device_name = 'RPi3'
    power = "None"


logger = logging.getLogger("test")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)

pid = psutil.Process().ppid()
logger.info(f"[{time.time()}] PPID: {pid}")
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
logging.basicConfig(filename=f"fl_info_{current_time}_{device_name}.txt")
fl.common.logger.configure(identifier="myFlowerExperiment", filename=f"fl_log_{current_time}.txt")
logger.info([f'[{time.time()}] Client Start!'])


logger.info(f"Using network interface: {wlan_interf}")
start_net = get_network_usage(wlan_interf)

power_monitor = None
if power != "None":
    power_monitor = get_power_monitor(power, device_name=socket.gethostname())

# Flower ClientApp
app = ClientApp(
    client_fn,
)

end_time = time.time()
logger.info([f'[{time.time()}] Communication end: {end_time}'])

# Log the network IO
end_net = get_network_usage(wlan_interf)
net_usage_sent = end_net["bytes_sent"] - start_net["bytes_sent"]
net_usage_recv = end_net["bytes_recv"] - start_net["bytes_recv"]
logger.info([f'[{time.time()}] Evaluation phase ({wlan_interf}): [sent: {net_usage_sent}, recv: {net_usage_recv}]'])

if power_monitor:
    elapsed_time, data_size = power_monitor.stop()
    if elapsed_time is not None:
        logger.info(f"Measured power consumption: Duration={elapsed_time}s, Data size={data_size} samples.")
    else:
        logger.warning("Power monitoring failed or returned no data.")