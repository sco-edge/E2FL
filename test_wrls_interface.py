from log import WrlsEnv
import time

WiFi_interface_arr = []
sleep_duration = 0.050 # scale = seconds, 0.001 means 1 ms.
while_count = 0
while_thold = 50
 
while True:
    temp = WrlsEnv.WiFi.parse_iwconfig_output()
    WiFi_interface_arr.append(temp)
    
    print(".", end=" ")
    time.sleep(sleep_duration) 

    while_count = while_count + 1
    if while_count >= while_thold:
        break

interf_count = 0
RSSI_temp1, RSSI_temp2, RSSI_temp3 = 0, 0, 0
for interf in WiFi_interface_arr:
    RSSI_temp1 = interf['wlan0']['Signal level']
    if RSSI_temp1 > RSSI_temp2:
        RSSI_temp2 = RSSI_temp1
        interf_count += 1
    elif RSSI_temp1 < RSSI_temp2:
        RSSI_temp2 = RSSI_temp1
        interf_count += 1


'''
the shortest time.sleep in Python?
https://discourse.julialang.org/t/julia-seems-an-order-of-magnitude-slower-than-python-when-printing-to-the-terminal-because-of-issue-with-sleep/78151?page=2#:~:text=A%20couple%20forums%20say%20that,Linux%2C%20based%20on%20tick%20rates.

Windows: ~10-13ms
Linux: ~1ms

'''



