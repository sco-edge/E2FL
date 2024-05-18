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

# Set up Power Monitor
node_A_vout = 5.0
node_A_triggerBool = True
node_A_numSamples = sampleEngine.triggers.SAMPLECOUNT_INFINITE
node_A_thld_high = 100
node_A_thld_low = 10
node_A_CSVbool = False #True
node_A_CSVname = "default"
rpi3B = Monitor.PowerMon(   node = node_A_name,
                            vout = node_A_vout,
                            mode = node_A_mode)
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
client_ip = config['RPi4B']['host']
private_key_path = root_path + config['RPi4B']['ssh_key']
client_interf = config['RPi4B']['interface']

if server_ip or client_ip:
    print(f"The IP address of the server is: {server_ip}")
    print(f"The IP address of the client is: {client_ip}")
else:
    print("IP address could not be determined")
    exit(1)

# Set up SSH service
client_SSH = paramiko.SSHClient()
client_SSH.set_missing_host_key_policy(paramiko.RejectPolicy())  # Add the host key automatically AutoAddPolicy()
mykey = paramiko.RSAKey.from_private_key_file(private_key_path)

try_count = 0
start_time = time.time()
while 1:
    print(f'{try_count} try ...')
    try:
        client_SSH.connect(hostname = client_ip, port = ssh_port, username = client_ssh_id, pkey=mykey)
        break
    except:
        try_count = try_count + 1
        pass
    time.sleep(10)
    if time.time() - start_time > _UPTIME_RPI3B:
        try:
            client_SSH.connect(client_ip, ssh_port, client_ssh_id, pkey=mykey)
        except Exception as e:
            logger.error("SSH is failed: ", e)
            logger.error(private_key_path)
            exit(1)
