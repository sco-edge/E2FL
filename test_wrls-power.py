from power import Monitor
import Monsoon.sampleEngine as sampleEngine
from log import WrlsEnv
import subprocess
import re
import iperf3
import logging
import time
from datetime import datetime

logging.basicCOnfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.debug, logging.info, logging.warning, logging.error, logging.critical

'''
Wi-Fi interface table
'Wi-Fi AP': IPTIME AX2002-Mesh
- b, g, n, ax 20MHz, Korea
- TX power = 100
- Beacon 100ms

'AX201'
'bcm434355c0' on the RPI3+/RPI4 https://github.com/seemoo-lab/nexmon
'



'''

def change_WiFi_interface(interf = 'wlan0', channel = 11, rate = '11M', txpower = 15):
    # change Wi-Fi interface 
    result = subprocess.run([f"iwconfig {interf} channel {channel} rate {rate} txpower {txpower}"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

time_records = []

# Set up WiFi interface
# 1. Identify the capabilities of the Wi-Fi interface of the currently running system.


# 2. Set the rate (protocl version) of the Wi-Fi interface from the low data rate.



# Log the start time.
time_records.append(['Wi-Fi start(rate: , )',time.time()])

# Start power monitoring.
node_A_name = 'rpi3B+'
node_A_vout = 5.0
node_A_mode = "PyMonsoon"
node_A_triggerBool = True
node_A_numSamples = 5000
node_A_thld_high = 100
node_A_thld_low = 10
node_A_CSVbool = False#True
node_A_CSVname = "default"

rpi3B = Monitor.PowerMon(   node = node_A_name,
                            vout = node_A_vout,
                            mode = node_A_mode )
'''
rpi3B.setTrigger(   bool = node_A_triggerBool,
                    numSamples = node_A_numSamples,
                    thld_high = node_A_thld_high,
                    thld_low = node_A_thld_low )
'''
rpi3B.setCSVOutput( bool = node_A_CSVbool,
                    filename = node_A_CSVname)


rpi3B.startSampling()
samples = rpi3B.getSamples()


# Use iperf3 to measure the Wi-Fi interface's power consumption.
'''
iperf3

-s : server mode
-c : client mode
-p {#}: port
-u : use UDP rather than TCP
-b : indicate bandwidth when using UDP
-t {#}: time interval
-w {#}: TCP window size (socket buffer size)
-M {#}: set TCP maximum setment size (MTU - 40 bytes)
-N : set TCP no delay, disabling Nagle's Algorithm
-V : set the domain to IPv6
-d : measure bi-direcitonal
-n {#}: number of bytes to transmit
-F {name}: input the data to be transmitted from a file
-I : input the data to be transmitted from stdin
-P #: number of parallel client thrads to run
-T : time-to-live, for multicast (default 1)
'''
iperf_time_interv = 2

iperf_client = iperf3.Client()
iperf_client.server.hostname = '192.168.0.1'
iperf_client.port = 5201
iperf_client.duration = 60
iperf_client.bind_address = '192.168.0.1' # wi-fi interface's ip address

iperf_result = iperf_client.run()

if iperf_result.error:
    logging.error(iperf_result.error)
else:
    print("iperf3 is done.")

# End power monitoring.


# Log the end time.
time_records.append(['Wi-Fi end(rate: , )',time.time()])


# Calculate each rate's average power consumption.
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
filename = f"data_{current_time}.txt"
try:
    # 파일을 열려고 시도합니다.
    f = open(filename, 'w')
    f.write(time_records)
    f.close()
except OSError as e:
    # OSError 발생 시 오류 코드와 메시지를 출력합니다.
    print(f"Error opening {filename}: {os.strerror(e.errno)}")


# Plot the result.