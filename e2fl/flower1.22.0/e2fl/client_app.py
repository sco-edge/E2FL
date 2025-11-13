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
from flwr.clientapp import ClientApp
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from e2fl.task import Net, get_weights, load_data, set_weights, test, train, get_num_classes, get_model

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

def get_last_octet_from_ip():
    # 현재 로컬 IP 주소의 마지막 옥텟 추출
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    last_octet = int(ip_address.strip().split(".")[-1])
    return last_octet

def get_network_interface():
    interfaces = psutil.net_if_addrs().keys()
    return list(interfaces)

def get_network_usage(interface):
    """Get the current network usage for the specified interface."""
    net_io = psutil.net_io_counters(pernic=True)
    if interface in net_io:
        return {"bytes_sent": net_io[interface].bytes_sent, "bytes_recv": net_io[interface].bytes_recv}
    else:
        raise ValueError(f"Interface {interface} not found.")

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

def cleanup_power_monitor(power_monitor, interface, device_name, start_net):
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
    end_net = get_network_usage(interface)
    net_usage_sent = end_net["bytes_sent"] - start_net["bytes_sent"]
    net_usage_recv = end_net["bytes_recv"] - start_net["bytes_recv"]
    logger.info([f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}] Evaluation phase ({interface}): [sent: {net_usage_sent}, recv: {net_usage_recv}]"])

def get_partition_id_from_ip():
    last_octet = get_last_octet_from_ip()
    ip_to_partition = {
        53: 0, # Jetson Orin NX
        47: 1, # Pi5_1
        23: 2, # Pi5_2
        55: 3, # Pi5_3
        56: 4, # Pi5_4
        54: 5  # Pi5_5
    }
    if last_octet not in ip_to_partition:
        raise ValueError(f"Unknown IP last octet: {last_octet}, no partition-id mapping found.")
    return ip_to_partition[last_octet], len(ip_to_partition)

interfaces = get_network_interface()
if 'wlp1s0' in interfaces:
    interfaces = 'wlp1s0' if validate_network_interface('wlp1s0') else "wlan0"
    device_name, power = 'RPi5', "PMIC"
elif 'wlP1p1s0' in interfaces:
    interfaces = 'wlP1p1s0' #if validate_network_interface('eth0') else "wlan0"
    device_name, power = 'jetson', "INA3221"
else:
    interfaces, device_name, power = 'wlan0', 'RPi3', "None"

# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.debug, logging.info, logging.warning, logging.error, logging.critical
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

# Flower ClientApp
app = ClientApp()

@app.train()
def train(msg: Message, context: Context) -> Message:
    # initialize context
    if "net" not in context.state:
        # load dataset and model
        model_name = context.run_config["model"]
        batch_size = context.run_config.get("batch-size", 32)
        dataset_name = context.run_config["dataset"]

        num_classes = get_num_classes(dataset_name)
        partition_id, num_partitions = get_partition_id_from_ip()
        context.state["trainloader"], context.state["valloader"] = load_data(
            dataset_name=dataset_name,
            partition_id=partition_id,
            num_partitions=num_partitions,
            batch_size=batch_size
        )
        context.state["net"] = get_model(model_name, num_classes)
        context.state["local_epochs"] = context.run_config["local-epochs"]

        interface = get_network_interface()
        if 'wlp1s0' in interface:
            context.state["interfaces"] = 'wlp1s0' if validate_network_interface('wlp1s0') else "wlan0"
            context.state["device_name"] = 'RPi5'
            context.state["power"] = "PMIC"
        elif 'wlP1p1s0' in interface:
            context.state["interfaces"] = 'wlP1p1s0' #if validate_network_interface('eth0') else "wlan0"
            context.state["device_name"] = 'jetson'
            context.state["power"] = "INA3221"
        else:
            context.state["interfaces"] = 'wlan0'
            context.state["device_name"] = 'RPi3'
            context.state["power"] = "None"
        logger.info(f"Using network interface: {context.state['interfaces']}")

        context.state["fl_csv_fname"] = f"fl_{datetime.now().strftime('%Y%m%d')}_{context.state['device_name']}_{get_last_octet_from_ip()}.csv"

        start_net = get_network_usage(context.state["interfaces"])
        context.state["end_net"] = None
    
        context.state["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        context.state["net"].to(context.state["device"])
        
        now = datetime.now()
        timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f')
        context.state["end_net"] = get_network_usage(context.state["interfaces"])
        net_usage_sent = context.state["end_net"]["bytes_sent"] - context.state["start_net"]["bytes_sent"]
        net_usage_recv = context.state["end_net"]["bytes_recv"] - context.state["start_net"]["bytes_recv"]
        context.state["start_net"] = context.state["end_net"]
        with open(context.state["fl_csv_fname"], 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([f"{timestamp}", "init", f"{net_usage_sent}", f"{net_usage_recv}"])
        
        # Run power monitor if configured
        if context.run_config.get("power-monitor", False):
            context.state["power_monitor"] = get_power_monitor(context.state["power"], device_name=socket.gethostname())
        if context.state["power_monitor"]:
            logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}] Starting power monitoring...")
            context.state["power_monitor"].start(freq=0.01)  # Start monitoring with 1-second intervals
            time.sleep(0.5)  # Example duration for monitoring
            atexit.register(cleanup_power_monitor, context.state["power_monitor"], context.state["start_net"])


    """Train the model on local data"""
    logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}] Client sampled for training()")

    # record network usage before training
    logger.info(f"Network usage during training: [sent: {net_usage_sent}, recv: {net_usage_recv}]")

    now = datetime.now()
    timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f')
    context.state["end_net"] = get_network_usage(context.state["interfaces"])
    net_usage_sent = context.state["end_net"]["bytes_sent"] - context.state["start_net"]["bytes_sent"]
    net_usage_recv = context.state["end_net"]["bytes_recv"] - context.state["start_net"]["bytes_recv"]
    context.state["start_net"] = context.state["end_net"]
    with open(context.state["fl_csv_fname"], 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f"{timestamp}", "fit_init", f"{net_usage_sent}", f"{net_usage_recv}"])


    # parameters to model
    arrays = msg.content["arrays"].as_numpy()
    set_weights(context.state["net"], arrays)

    # run training
    logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}] Starting training...")
    train_loss = train(context.state["net"], context.state["trainloader"], \
                       context.state["local_epochs"], context.state["device"])
    logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}] Training completed.")


    # record network usage after training
    now = datetime.now()
    timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f')
    context.state["end_net"] = get_network_usage(context.state["interfaces"])
    net_usage_sent = context.state["end_net"]["bytes_sent"] - context.state["start_net"]["bytes_sent"]
    net_usage_recv = context.state["end_net"]["bytes_recv"] - context.state["start_net"]["bytes_recv"]
    context.state["start_net"] = context.state["end_net"]
    with open(context.state["fl_csv_fname"], 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f"{timestamp}", "fit_end", f"{net_usage_sent}", f"{net_usage_recv}"])

    # Add power info (if available)
    power_usage = None
    if "power_monitor" in context.state and context.state["power_monitor"] is not None:
        try:
            power_usage = context.state["power_monitor"].read_power()
            logger.info(f"[{timestamp}] Power usage during training: {power_usage} mW")
        except Exception as e:
            logger.warning(f"Failed to read power usage: {e}")

    # construct and return reply Message
    model_record = ArrayRecord(context.state["net"].state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(context.state["trainloader"].dataset),
        "net_sent": net_usage_sent,
        "net_recv": net_usage_recv,
    }
    if power_usage is not None:
        metrics["power_mw"] = power_usage
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluate the model on local data."""
    logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}] Client sampled for evaluate()")

    now = datetime.now()
    timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f')
    context.state["end_net"] = get_network_usage(context.state["interfaces"])
    net_usage_sent = context.state["end_net"]["bytes_sent"] - context.state["start_net"]["bytes_sent"]
    net_usage_recv = context.state["end_net"]["bytes_recv"] - context.state["start_net"]["bytes_recv"]
    context.state["start_net"] = context.state["end_net"]
    with open(context.state["fl_csv_fname"], 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f"{timestamp}", "eval_init", f"{net_usage_sent}", f"{net_usage_recv}"])

    
    logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}] Starting evaluation...")
    loss, accuracy = test(context.state["net"], context.state["valloader"], context.state["device"])
    logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}] Evaluation completed with accuracy: {accuracy}")


    now = datetime.now()
    timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f')
    context.state["end_net"] = get_network_usage(context.state["interfaces"])
    net_usage_sent = context.state["end_net"]["bytes_sent"] - context.state["start_net"]["bytes_sent"]
    net_usage_recv = context.state["end_net"]["bytes_recv"] - context.state["start_net"]["bytes_recv"]
    context.state["start_net"] = context.state["end_net"]
    with open(context.state["fl_csv_fname"], 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f"{timestamp}", "eval_end", f"{net_usage_sent}", f"{net_usage_recv}"])
    
    
    # Construct reply Message
    metrics = MetricRecord(
        {
            "eval_loss": loss,
            "eval_accuracy": accuracy,
            "num-examples": len(context.state["test_loader"].dataset),
        }
    )
    # Construct RecordDict and add MetricRecord
    content = RecordDict({"metrics": metrics})
    return Message(content=content, reply_to=msg)