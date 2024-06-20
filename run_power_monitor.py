from power import Monitor
import Monsoon.sampleEngine as sampleEngine
from log import WrlsEnv
from datetime import datetime
import subprocess, os, logging, time, socket, pickle
import paramiko, yaml
import re
import argparse
from typing import List, Tuple
import asyncio

import argparse 
from typing import List, Tuple
import flwr as fl
from flwr.common import Metrics

import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


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


def get_client_SSH(client_ip, wait_time):
    # Wait for boot up
    print(f"Wait {wait_time} seconds for the edge device to boot.")

    # Set up SSH service
    client_SSH = paramiko.SSHClient()
    client_SSH.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # Add the host key automatically AutoAddPolicy()
    mykey = paramiko.RSAKey.from_private_key_file(private_key_path)

    # ssh -i {rsa} {USER}@{IP_ADDRESS}
    try_count = 0
    time.sleep(150)
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

async def start_powermon(rpi3B):
    rpi3B.startSampling(numSamples = node_A_numSamples) # it will take measurements every 200us
    

# default parameters
root_path = os.path.abspath(os.getcwd())+'/'
node_A_name = 'rpi3B+'
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

# Set up Power Monitor
node_A_vout = 5.0
node_A_triggerBool = False
node_A_numSamples = sampleEngine.triggers.SAMPLECOUNT_INFINITE
node_A_thld_high = 100
node_A_thld_low = 10
node_A_CSVbool = True
node_A_CSVname = "default"
rpi3B = Monitor.PowerMon(   node = node_A_name,
                            vout = node_A_vout,
                            mode = node_A_mode,
                            ConsoleIO = False)
rpi3B.setTrigger(   bool = node_A_triggerBool,
                    thld_high = node_A_thld_high,
                    thld_low = node_A_thld_low )
rpi3B.setCSVOutput( bool = node_A_CSVbool,
                    filename = node_A_CSVname)

# Read the YAML config file.
with open(root_path+'config.yaml', 'r') as file:
    config = yaml.safe_load(file)  # Read YAML from the file and convert the structure of the python's dictionary.

# Get IP addresses.
server_ip = config['server']['host']  # Extract 'host' key's value from the 'server' key
client_ip1 = config['RPi3B+']['host']
client_ip2 = config['RPi3B+_b']['host']
client_ip3 = config['RPi4B']['host']
client_ip4 = config['RPi5']['host']
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
client_shells = []
for c_ip in [client_ip1]:#, client_ip2, client_ip3, client_ip4]:
    client_shells.append(get_client_SSH(client_ip = c_ip, wait_time = _UPTIME_RPI3B))


# Prepare a bucket to store the results.
measurements_dict = []

WiFi_rates = [1] #[1, 2, 5.5, 11, 6, 9, 12, 18, 24, 36, 48, 54]

while True:
    time_records = []

    # Log the start time.
    time_records.append(time.time())

    input_string = input("Press Enter to start power monitoring\n")

    # Start power monitoring.
    logger.info('Start the power monitor.')
    rpi3B.startSampling(numSamples = node_A_numSamples) # it will take measurements every 200us
    #task1 = asyncio.create_task(start_powermon(rpi3B))

    input_string = input("Press Enter to stop power monitoring\{Q: quit\}: ")

    # End power monitoring.
    logger.info('Stop the power monitor.')
    rpi3B.stopSampling()
    samples = rpi3B.getSamples()

    # Save the data.
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"power_measure_{current_time}.pickle"
    with open(filename, 'wb') as handle:
        pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"The measurement data is saved as {filename}.")

    if 'Q' in input_string:
        logger.info("Power Monitoring is closed.")
        break

    # Log the end time.
    #time_records.append(time.time())
    #logger.info([f'Wi-Fi end(rate: {rate})',time.time()])

    #measurements_dict.append({'rate': rate, 'time': time_records, 'power': samples})

# Close the SSH connection.
for client_SSH in client_shells:
    client_SSH.close()






