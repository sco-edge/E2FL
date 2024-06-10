from power import Monitor
import Monsoon.sampleEngine as sampleEngine
from log import WrlsEnv
from datetime import datetime
import subprocess, os, logging, time, socket, pickle
import paramiko, yaml
import re


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

def kill_running_iperf3_server():
    result = subprocess.run(['ps', 'aux'], stdout=subprocess.PIPE, text=True)
    process_list = result.stdout.splitlines()

    # Look for iperf3 server process
    for process in process_list:
        if 'iperf3 -s' in process:
            # Extract process ID
            parts = process.split()
            pid = parts[1]
            
            # Kill the iperf3 server process
            try:
                subprocess.run(['kill', '-9', pid], check=True)
                logger.info(f"iperf3 server process (PID: {pid}) has been terminated.")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to terminate iperf3 server process (PID: {pid}): {e}")
                exit(1)

def start_iperf3_server(server_ip, port=5201):
    """Start a iperf3 server at a server A asynchronously."""
    kill_running_iperf3_server()
    logger.info(f"Kill existing iperf3 server process and wait.")
    time.sleep(10)
    # Start iperf3 server. (Waitting at 5201 port)
    return subprocess.Popen(['iperf3', '-s', '--bind', str(server_ip), '-p', str(port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def run_iperf3_client(client_SSH, server_ip, duration = 10, server_port = 5201):
    """Run a iperf3 client at a edge device B to send data to server A."""
    try:
        # Run iperf3 client command.
        command = 'iperf3 -c '+str(server_ip)+' -p '+str(server_port)+' -t '+str(duration)
        output = client_SSH.send(command) # exec_command
        time.sleep(2)
    
        # Receive the output
        output = client_SSH.recv(65535).decode() # 65535 is the maximum bytes that can read by recv() method.
        
        if 'error' in output.lower() or 'command not found' in output.lower():
            logger.error("Error detected in the command output.")
            return False
        else:
            logger.info("iperf3 executed successfully.")
            logger.info("===============(Terminal Output START)===============")
            logger.info(output)
            logger.info("===============(Terminal Output  END )===============")

        '''
        # Print the results.
        for line in stdout.read().splitlines():
            logger.info('client_SSH:')
            logger.info(line.decode('utf-8'))
        for line in stderr.read().splitlines():
            logger.error(line.decode('utf-8'))
            return False
        '''
    except:
        logger.error("run_iperf3_client failed.")
        return False
    return True

# default parameters
root_path = os.path.abspath(os.getcwd())+'/'
node_A_name = 'rpi3B+'
node_A_mode = "PyMonsoon"
client_ssh_id = 'pi'
ssh_port = 22
iperf3_server_port = 5203
iperf3_duration = 10

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
client_ip = config['RPi3B+']['host']
private_key_path = root_path + config['RPi3B+']['ssh_key']
client_interf = config['RPi3B+']['interface']

if server_ip or client_ip:
    print(f"The IP address of the server is: {server_ip}")
    print(f"The IP address of the client is: {client_ip}")
    print(f"The ID of the client is: {client_ssh_id}")
else:
    print("IP address could not be determined")
    exit(1)

# Wait for boot up
print(f"Wait {_UPTIME_RPI3B} seconds for the edge device to boot.")

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

# Start the iperf3 server.
try:
    server_process = start_iperf3_server(server_ip = server_ip, port = iperf3_server_port)
    logger.info("Start iperf3 server.")
    # Wait for server to start iperf3 properly.
    time.sleep(5)
    #server_process.stdout.readline()
    #logger.info(server_process.stdout.readline())
except Exception as e:
    logger.error('iperf3 is failed: ', e)
    exit(1)

# Prepare a bucket to store the results.
measurements_dict = []

# Set up the edge device's the WiFi interface.
# Identify the capabilities of the Wi-Fi interface of the currently running system.
WiFi_rates = [1] #[1, 2, 5.5, 11, 6, 9, 12, 18, 24, 36, 48, 54]

for rate in WiFi_rates:
    time_records = []

    # Set the edge device's rate (protocol version) in the Wi-Fi interface from the low data rate.
    #result = change_WiFi_interface(interf = client_interf, channel = 11, rate = str(rate)+'M', txpower = 15)
    '''
    result = change_WiFi_interface_client(client_ssh = client_shell, interf = client_interf, channel = 11, rate = str(rate)+'M', txpower = 15)
    if result == False:
        logger.error("iwconfig IS FAILED>")
        exit(1)
    '''
    # Log the start time.
    time_records.append(time.time())
    logger.info([f'Wi-Fi start(rate: {rate})',time.time()])

    # Start power monitoring.
    rpi3B.startSampling(numSamples = node_A_numSamples) # it will take measurements every 200us
    logger.info('Start power monitor.')

    if client_SSH.get_transport().is_active():
        logger.info("client_SSH is alive.")

    # Use iperf3 to measure the Wi-Fi interface's power consumption.
    result = run_iperf3_client(client_shell, server_ip, duration = iperf3_duration, server_port = iperf3_server_port)
    if result == False:
        logger.error("IPERF3 CLIENT IS FAILED.")
        exit(1)

    # End power monitoring.
    rpi3B.stopSampling()
    samples = rpi3B.getSamples()

    # Log the end time.
    time_records.append(time.time())
    logger.info([f'Wi-Fi end(rate: {rate})',time.time()])

    measurements_dict.append({'rate': rate, 'time': time_records, 'power': samples})

# Close the SSH connection.
client_SSH.close()

# Terminate the iperf3 server process.
server_process.terminate()
logger.info("Close the iperf3 serveer.")

# Save the data.
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
filename = f"data_{current_time}.pickle"
with open(filename, 'wb') as handle:
    pickle.dump(measurements_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
logger.info(f"The measurement data is saved as {filename}.")

dataset = ['CIFAR']

client_cmd = "python3.10 ./FLOWER_embedded_devices/client_pytorch.py --cid=$client_id --server_address=$server_address --mnist"
server_cmd = "python3.10 ./FLOWER_embedded_devices/server.py --rounds $round --min_num_clients $num_clients --sample_fraction $sample_frac"
