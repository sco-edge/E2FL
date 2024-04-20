from power import Monitor
from log import WrlsEnv
import subprocess
import re

def change_WiFi_interface():
    # change Wi-Fi interface 
    result = subprocess.run(['iwconfig'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
