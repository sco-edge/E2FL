import matplotlib.pyplot as plt
import numpy as np
import csv
from datetime import datetime

fl_state_jetson = "fl_20250328_jetson.csv"
fl_state_rpi5 = "fl_20250328_RPi5.csv"
# timestamp, state, network sent, network recv

powr_jetson = [
                "power_jetson_2025-03-28_015632.827713.csv", 
               "power_jetson_2025-03-28_015701.067159.csv",
               "power_jetson_2025-03-28_015742.249796.csv",
               "power_jetson_2025-03-28_015813.987895.csv",
               "power_jetson_2025-03-28_015856.259900.csv",
               "power_jetson_2025-03-28_015928.537096.csv"
]
power_rpi5 = [
    "power_RPi5_2025-03-28_015929.085905.csv",
    "power_RPi5_2025-03-28_015640.230948.csv",
    "power_RPi5_2025-03-28_015704.585146.csv",
    "power_RPi5_2025-03-28_015751.286886.csv",
    "power_RPi5_2025-03-28_015816.661843.csv",
    "power_RPi5_2025-03-28_015904.114542.csv"
]
# timestamp, power (mW)




def load_power_data(file_list):
    timestamps = []
    power_values = []
    for fname in file_list:
        with open(fname, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                try:
                    ts = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S.%f")
                    power = float(row[1])
                    timestamps.append(ts)
                    if 'RPi5' in fname:
                        # https://github.com/jfikar/RPi5-power
                        power = power * 1.1451 + 0.5879
                    elif 'jetson' in fname:
                        # Convert Jetson power from mW to W
                        power = power / 1000
                    power_values.append(power)
                except:
                    continue  # Skip header or invalid rows
    return timestamps, power_values

def load_net_data(filename):
    timestamps = []
    net_sents = []
    net_recvs = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                ts = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S.%f")
                sent = float(row[2])
                recv = float(row[3])
                timestamps.append(ts)
                net_sents.append(sent)
                net_recvs.append(recv)
            except:
                continue  # Skip header or invalid rows
    return timestamps, net_sents, net_recvs

# 데이터 로드
ts_power_jetson, pw_jetson = load_power_data(powr_jetson)
ts_power_rpi5, pw_rpi5 = load_power_data(power_rpi5)
ts_net_jetson, net_sent_jetson, net_recv_jetson = load_net_data(fl_state_jetson)
ts_net_rpi5, net_sent_rpi5, net_recv_rpi5 = load_net_data(fl_state_rpi5)

# 시각화
import matplotlib.dates as mdates
# Plot for Jetson Power
fig1, ax1 = plt.subplots()
ax1.plot(ts_power_jetson, pw_jetson)
#ax1.set_title("Jetson Power Consumption Over Time")
ax1.set_xlabel("Time")
ax1.set_ylabel("Power (W)")
#ax1.legend()
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
#plt.show()
plt.savefig("jetson_power.png")
plt.close()

# Plot for RPi5 Power
fig2, ax2 = plt.subplots()
ax2.plot(ts_power_rpi5, pw_rpi5)
#ax2.set_title("RPi5 Power Consumption Over Time")
ax2.set_xlabel("Time")
ax2.set_ylabel("Power (W)")
#ax2.legend()
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
#plt.show()
plt.savefig("rpi5_power.png")
plt.close()

# Plot for Jetson Network Usage
fig3, ax3 = plt.subplots()
ax3.plot(ts_net_jetson, net_sent_jetson, label="Sent")
ax3.plot(ts_net_jetson, net_recv_jetson, label="Recv")
#ax3.set_title("Jetson Network Usage Over Time")
ax3.set_xlabel("Time")
ax3.set_ylabel("Bytes")
ax3.legend()
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
#plt.show()
plt.savefig("jetson_network.png")
plt.close()

# Plot for RPi5 Network Usage
fig4, ax4 = plt.subplots()
ax4.plot(ts_net_rpi5, net_sent_rpi5, label="Sent")
ax4.plot(ts_net_rpi5, net_recv_rpi5, label="Recv")
#ax4.set_title("RPi5 Network Usage Over Time")
ax4.set_xlabel("Time")
ax4.set_ylabel("Bytes")
ax4.legend()
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
#plt.show()
plt.savefig("rpi5_network.png")
plt.close()



