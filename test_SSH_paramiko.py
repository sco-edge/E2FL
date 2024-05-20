from power import Monitor
import Monsoon.sampleEngine as sampleEngine
from log import WrlsEnv
import subprocess
import re
import os
import logging
import time
from datetime import datetime
import socket
import paramiko
import yaml
import pickle

_UPTIME_RPI3B = 500

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

# default parameters
root_path = os.path.abspath(os.getcwd())+'/'
node_A_name = 'rpi3B+'
node_A_mode = "PyMonsoon"
client_ssh_id = 'pi'
ssh_port = 22
iperf3_server_port = 5201

# Set up logger
logger = logging.getLogger("test")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)

# Read the YAML config file.
with open(root_path+'config.yaml', 'r') as file:
    config = yaml.safe_load(file)  # Read YAML from the file and convert the structure of the python's dictionary.

# Get IP addresses.
server_ip = config['server']['host']  # Extract 'host' key's value from the 'server' key
client_ip = config['RPi4B']['host']
private_key_path = root_path + config['RPi4B']['ssh_key']
client_interf = config['RPi4B']['interface']

if server_ip or client_ip:
    print(f"The IP address of the server is: {server_ip}")
    print(f"The IP address of the client is: {client_ip}")
    print(f"The ID of the client is: {client_ssh_id}")
else:
    print("IP address could not be determined")
    exit(1)

# Set up SSH service
client_SSH = paramiko.SSHClient()
client_SSH.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # Add the host key automatically AutoAddPolicy()
#client_SSH.set_missing_host_key_policy(paramiko.RejectPolicy())  # Add the host key automatically AutoAddPolicy()
mykey = paramiko.RSAKey.from_private_key_file(private_key_path)

try_count = 0
try:
    client_SSH.connect(hostname = client_ip, port = ssh_port, username = client_ssh_id, passphrase="", key_filename=private_key_path, look_for_keys=False)
    print("SUCCESS")
except Exception as e:
    logger.error("SSH is failed: ", e)
    logger.error(private_key_path)
    exit(1)

