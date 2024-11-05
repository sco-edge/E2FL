from log import WrlsEnv
import time

WiFi_interface_arr = []
sleep_duration = 0.010 # scale = seconds, 0.001 means 1 ms.
while_count = 0
while_thold = 100
 
while True:
    temp = WrlsEnv.WiFi.parse_iwconfig_output()
    WiFi_interface_arr.append(temp)
    
    #print(".", end=" ")
    time.sleep(sleep_duration) 

    while_count = while_count + 1
    if while_count >= while_thold:
        break

interf_count = 0
RSSI_temp1, RSSI_temp2 = 0, 0
link_temp1, link_temp2 = 0, 0
for interf in WiFi_interface_arr:
    RSSI_temp1 = int(interf['wlan0']['Signal level'])
    link_temp1 = int(interf['wlan0']['Signal level'])
    if RSSI_temp1 != RSSI_temp2:
        RSSI_temp2 = RSSI_temp1
        interf_count += 1
    elif link_temp1 != link_temp2:
        link_temp2 = link_temp1
        interf_count += 1

print()
print("interf_count: ",interf_count)


'''
the shortest time.sleep in Python?
https://discourse.julialang.org/t/julia-seems-an-order-of-magnitude-slower-than-python-when-printing-to-the-terminal-because-of-issue-with-sleep/78151?page=2#:~:text=A%20couple%20forums%20say%20that,Linux%2C%20based%20on%20tick%20rates.

Windows: ~10-13ms
Linux: ~1ms


$ time iwconfig
    rpi3B+        rpi4B       Desktop
real 0m0.018s   0m0.014s    0m0.004s
user 0m0.000s   0m0.000s    0m0.000s
sys  0m0.015s   0m0.011s    0m0.004s

https://stackoverflow.com/questions/556405/what-do-real-user-and-sys-mean-in-the-output-of-time1

Real is wall clock time
- time from start to finish of the call
User is the amount of CPU time spent in user-mode code within the process
-
Sys is the amount of CPU time spent in the kernel within the process

Ouputs of the test_wrls_interface.py
12 time changes in 1 second.
=> 100 ms ~ 83 ms
'''



