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
import atexit, csv
import socket

import torch

import flwr as fl
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
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

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs

        interfaces = self.get_network_interface()
        if 'wlp1s0' in interfaces:
            self.interface = 'wlp1s0' if self.validate_network_interface('wlp1s0') else "wlan0"
            self.device_name = 'RPi5'
            self.power = "PMIC"
        elif 'wlan0' in interfaces:
            self.interface = 'wlan0' #if self.validate_network_interface('eth0') else "wlan0"
            self.device_name = 'jetson'
            self.power = "INA3221"
        else:
            self.interface = 'wlan0'
            self.device_name = 'RPi3'
            self.power = "None"
        self.fl_csv_fname = f"fl_{datetime.now().strftime('%Y%m%d')}_{self.device_name}_{self.get_last_octet_from_ip()}.csv"

        self.start_net = self.get_network_usage()
        self.end_net = None
    
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        

        with open(self.fl_csv_fname, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)

            now = datetime.now()
            timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f')
            self.end_net = self.get_network_usage()
            net_usage_sent = self.end_net["bytes_sent"] - self.start_net["bytes_sent"]
            net_usage_recv = self.end_net["bytes_recv"] - self.start_net["bytes_recv"]

            # timestamp, FL state, sent, receive
            writer.writerow([f"{timestamp}", "init", f"{net_usage_sent}", f"{net_usage_recv}"])

        '''
        # Return Client instance
        # Measure power consumption if a power monitor is initialized
        self.power_monitor = None
        if power != "None":
            self.power_monitor = get_power_monitor(self.power, device_name=socket.gethostname())
        if self.power_monitor:
            logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}] Starting power monitoring...")
            self.power_monitor.start(freq=0.01)  # Start monitoring with 1-second intervals
            time.sleep(0.5)  # Example duration for monitoring
        '''
    
    def get_last_octet_from_ip(self):
        # 현재 로컬 IP 주소의 마지막 옥텟 추출
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        last_octet = int(ip_address.strip().split(".")[-1])
        
        return last_octet

    def get_network_interface(self):
        interfaces = psutil.net_if_addrs().keys()
        return list(interfaces)

    def get_network_usage(self):
        """Get the current network usage for the specified interface."""
        net_io = psutil.net_io_counters(pernic=True)
        if self.interface in net_io:
            return {"bytes_sent": net_io[self.interface].bytes_sent, "bytes_recv": net_io[self.interface].bytes_recv}
        else:
            raise ValueError(f"Interface {self.interface} not found.")
    
    def validate_network_interface(self, interface):
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

    def fit(self, parameters, config):
        logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}] Client sampled for fit()")

        # 네트워크 사용량 기록
        self.end_net = self.get_network_usage()
        net_usage_sent = self.end_net["bytes_sent"] - self.start_net["bytes_sent"]
        net_usage_recv = self.end_net["bytes_recv"] - self.start_net["bytes_recv"]
        logger.info(f"Network usage during fit: [sent: {net_usage_sent}, recv: {net_usage_recv}]")


        with open(self.fl_csv_fname, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)

            now = datetime.now()
            timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f')
            self.end_net = self.get_network_usage()
            net_usage_sent = self.end_net["bytes_sent"] - self.start_net["bytes_sent"]
            net_usage_recv = self.end_net["bytes_recv"] - self.start_net["bytes_recv"]

            # timestamp, FL state, sent, receive
            writer.writerow([f"{timestamp}", "fit_init", f"{net_usage_sent}", f"{net_usage_recv}"])


        set_weights(self.net, parameters)

        logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}] Starting training...")
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
        )
        logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}] Training completed.")


        with open(self.fl_csv_fname, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)

            now = datetime.now()
            timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f')
            self.end_net = self.get_network_usage()
            net_usage_sent = self.end_net["bytes_sent"] - self.start_net["bytes_sent"]
            net_usage_recv = self.end_net["bytes_recv"] - self.start_net["bytes_recv"]

            # timestamp, FL state, sent, receive
            writer.writerow([f"{timestamp}", "fit_end", f"{net_usage_sent}", f"{net_usage_recv}"])


        # 학습 후 네트워크 상태 업데이트
        self.start_net = self.get_network_usage()
        
        '''
        if self.power_monitor:
            elapsed_time, data_size = self.power_monitor.stop()
            if elapsed_time is not None:
                logger.info(f"Measured power consumption: Duration={elapsed_time}s, Data size={data_size} samples.")
                self.power_monitor.save(f"power_{self.device_name}_{datetime.now().strftime('%Y-%m-%d_%H%M%S.%f')}.csv")
                self.power_monitor.close()
            else:
                logger.warning("Power monitoring failed or returned no data.")
        '''
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}] Client sampled for evaluate()")


        with open(self.fl_csv_fname, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)

            now = datetime.now()
            timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f')
            self.end_net = self.get_network_usage()
            net_usage_sent = self.end_net["bytes_sent"] - self.start_net["bytes_sent"]
            net_usage_recv = self.end_net["bytes_recv"] - self.start_net["bytes_recv"]

            # timestamp, FL state, sent, receive
            writer.writerow([f"{timestamp}", "eval_init", f"{net_usage_sent}", f"{net_usage_recv}"])


        set_weights(self.net, parameters)

        logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}] Starting evaluation...")
        loss, accuracy = test(self.net, self.valloader, self.device)
        logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}] Evaluation completed with accuracy: {accuracy}")


        with open(self.fl_csv_fname, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)

            now = datetime.now()
            timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f')
            self.end_net = self.get_network_usage()
            net_usage_sent = self.end_net["bytes_sent"] - self.start_net["bytes_sent"]
            net_usage_recv = self.end_net["bytes_recv"] - self.start_net["bytes_recv"]

            # timestamp, FL state, sent, receive
            writer.writerow([f"{timestamp}", "eval_end", f"{net_usage_sent}", f"{net_usage_recv}"])


        '''
        if self.power_monitor:
            elapsed_time, data_size = self.power_monitor.stop()
            if elapsed_time is not None:
                logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}] Measured power consumption: Duration={elapsed_time}s, Data size={data_size} samples.")
                self.power_monitor.save(f"power_{self.device_name}_{datetime.now().strftime('%Y-%m-%d_%H%M%S.%f')}.csv")
                self.power_monitor.close()
            else:
                logger.warning("Power monitoring failed or returned no data.")
        '''
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
    return list(interfaces)

def get_network_usage(interf):
    net_io = psutil.net_io_counters(pernic=True)
    #net_io = psutil.net_io_counters(pernic=True)
    return {"bytes_sent": net_io[interf].bytes_sent, "bytes_recv": net_io[interf].bytes_recv}

def cleanup_power_monitor(power_monitor, start_net):
    logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}] Stopping power monitor via atexit...")
    if power_monitor:
        elapsed_time, data_size = power_monitor.stop()
        if elapsed_time is not None:
            logger.info(f"Measured power consumption: Duration={elapsed_time}s, Data size={data_size} samples.")
            power_monitor.save(f"power_{device_name}_{datetime.now().strftime('%Y-%m-%d_%H%M%S.%f')}.csv")
            power_monitor.close()
        else:
            logger.warning("Power monitoring failed or returned no data.")
    
    end_time = time.time()
    logger.info([f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}] Communication end: {end_time}"])

    # Log the network IO
    end_net = get_network_usage(wlan_interf)
    net_usage_sent = end_net["bytes_sent"] - start_net["bytes_sent"]
    net_usage_recv = end_net["bytes_recv"] - start_net["bytes_recv"]
    logger.info([f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}] Evaluation phase ({wlan_interf}): [sent: {net_usage_sent}, recv: {net_usage_recv}]"])


def get_last_octet_from_ip():
    # 현재 로컬 IP 주소의 마지막 옥텟 추출
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    last_octet = int(ip_address.strip().split(".")[-1])
    
    return last_octet

def get_partition_id_from_ip():
    last_octet = get_last_octet_from_ip()
    
    # 사전 정의된 매핑 테이블
    ip_to_partition = {
        34: 0,
        40: 1,
        47: 2,
        54: 3,
        55: 4,
    }

    # 매핑이 없는 경우 예외 처리
    if last_octet not in ip_to_partition:
        raise ValueError(f"Unknown IP last octet: {last_octet}, no partition-id mapping found.")

    return ip_to_partition[last_octet], len(ip_to_partition)

def client_fn(context: Context):
    # config
    model_name = context.run_config["model"]
    dataset_name = context.run_config["dataset"]
    batch_size = context.run_config.get("batch-size", 32)
    local_epochs = context.run_config["local-epochs"]

    # 자동 partition 설정
    partition_id, num_partitions = get_partition_id_from_ip()

    # load dataset and model
    num_classes = get_num_classes(dataset_name)
    net = get_model(model_name, num_classes)
    trainloader, valloader = load_data(
        dataset_name=dataset_name,
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=batch_size
    )

    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()

interfaces = get_network_interface()
if 'wlp1s0' in interfaces:
    wlan_interf = 'wlp1s0' if validate_network_interface('wlp1s0') else "wlan0"
    device_name = 'RPi5'
    power = "PMIC"
elif 'wlan0' in interfaces:
    wlan_interf = 'wlan0' #if validate_network_interface('eth0') else "wlan0"
    device_name = 'jetson'
    power = "INA3221"
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
logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}] PPID: {pid}")
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
logging.basicConfig(filename=f"fl_info_{current_time}_{device_name}_{get_last_octet_from_ip()}.txt")
fl.common.logger.configure(identifier="myFlowerExperiment", filename=f"fl_log_{current_time}_{device_name}_{get_last_octet_from_ip()}.txt")
logger.info([f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}] Client Start!"])


logger.info(f"Using network interface: {wlan_interf}")
start_net = get_network_usage(wlan_interf)

# Return Client instance
# Measure power consumption if a power monitor is initialized
power_monitor = None
if power != "None":
    power_monitor = get_power_monitor(power, device_name=socket.gethostname())
if power_monitor:
    logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}] Starting power monitoring...")
    power_monitor.start(freq=0.01)  # Start monitoring with 1-second intervals
    time.sleep(0.5)  # Example duration for monitoring
    atexit.register(cleanup_power_monitor, power_monitor, start_net)

# Flower ClientApp
app = ClientApp(client_fn)

time.sleep(0.5)



'''
if power_monitor:
    elapsed_time, data_size = power_monitor.stop()
    if elapsed_time is not None:
        logger.info(f"Measured power consumption: Duration={elapsed_time}s, Data size={data_size} samples.")
    else:
        logger.warning("Power monitoring failed or returned no data.")
'''