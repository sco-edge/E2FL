from power import Monitor
from log import WrlsEnv
import subprocess
import re

result = subprocess.run(['iwconfig'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
