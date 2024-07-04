import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
from datetime import datetime, timedelta

# Load power consumption data
power_data_path = '../../../eval_E2FL/E2FL_20240628_175814.csv'
power_data = pd.read_csv(power_data_path)#, delimiter='\s+', names=["Time(ms)", "USB(mA)", "Aux(mA)", "USB Voltage(V)"])

# Load client data
client_data_path = './updated_client_data.pkl'
with open(client_data_path, 'rb') as f:
    client_data = pickle.load(f)

# Function to calculate total power (mW)
power_data['Total Power (mW)'] = power_data['USB(mA)'] * power_data['USB Voltage(V)']

# Function to identify start of computation
def find_start_of_computation(power_data, threshold=5):
    start_index = power_data[power_data['Total Power (mW)'] > threshold].index[0]
    return power_data.iloc[start_index]['Time(ms)']

# Function to calculate power consumption for each phase
def calculate_power_consumption(start_time, wait_times, power_data):
    power_consumption = []
    for wait_time in wait_times:
        phase_start_time = start_time + wait_time['start_time']
        phase_end_time = phase_start_time + wait_time['duration']
        phase_data = power_data[(power_data['Time(ms)'] >= phase_start_time) & (power_data['Time(ms)'] <= phase_end_time)]
        total_power = np.trapz(phase_data['Total Power (mW)'], phase_data['Time(ms)'])
        power_consumption.append(total_power)
    return power_consumption

# Identify the start of the computation
start_time = find_start_of_computation(power_data)

# Calculate power consumption for each phase
def convert_wait_times_to_intervals(client_data):
    intervals = []
    init_time = datetime.fromtimestamp(client_data['init'])
    for comm in client_data['Communication (after fit)'] + client_data['Communication (after evaluate)']:
        start_time = init_time + timedelta(seconds=comm['wait_time'])
        intervals.append({'start_time': (start_time - init_time).total_seconds(), 'duration': comm['wait_time']})
    return intervals

# Total duration for Shufflenet and Squeezenet in milliseconds
shufflenet_duration = (8 * 60 + 41) * 1000  # 8 minutes 41 seconds in milliseconds

# Split the power data into Shufflenet and Squeezenet
power_data_shufflenet = power_data[power_data['Time(ms)'] <= shufflenet_duration]
power_data_squeezenet = power_data[power_data['Time(ms)'] > shufflenet_duration].reset_index(drop=True)
power_data_squeezenet['Time(ms)'] -= power_data_squeezenet['Time(ms)'].iloc[0]

# Analyze power consumption for each client
for client_id, client_segments in client_data.items():
    for segment_index, segment_data in enumerate(client_segments):
        wait_times = convert_wait_times_to_intervals(segment_data)
        if segment_index == 0:  # Shufflenet
            power_consumption = calculate_power_consumption(start_time, wait_times, power_data_shufflenet)
            print(f'Client {client_id} (Shufflenet) - Power Consumption per Phase: {power_consumption}')
        else:  # Squeezenet
            power_consumption = calculate_power_consumption(start_time, wait_times, power_data_squeezenet)
            print(f'Client {client_id} (Squeezenet) - Power Consumption per Phase: {power_consumption}')

# Example plot to visualize power consumption
def plot_power_consumption(power_data, start_time, title, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(power_data['Time(ms)'], power_data['Total Power (mW)'], label='Total Power (mW)')
    plt.axvline(x=start_time, color='r', linestyle='--', label='Start of Computation')
    plt.xlabel('Time (ms)')
    plt.ylabel('Power (mW)')
    plt.title(title)
    plt.legend()
    #plt.show()
    plt.savefig('./'+filename)

plot_power_consumption(power_data_shufflenet, start_time, 'Power Consumption Over Time (Shufflenet)', 'plot_p_mnist(shufflenet).png')
plot_power_consumption(power_data_squeezenet, start_time, 'Power Consumption Over Time (Squeezenet)', 'plot_p_mnist(squeezenet).png')