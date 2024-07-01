import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load power consumption data
power_data_path = './E2FL_20240625_124846.csv'
power_data = pd.read_csv(power_data_path, delimiter='\s+', names=["Time(ms)", "USB(mA)", "Aux(mA)", "USB Voltage(V)"])

# Load client data
client_data_path = './updated_client_data.pkl'
with open(client_data_path, 'rb') as f:
    client_data = pickle.load(f)

# Function to identify start of computation
def find_start_of_computation(power_data, threshold=5):
    power_data['Total Power (mW)'] = power_data['USB(mA)'] * power_data['USB Voltage(V)']
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

# Analyze power consumption for each client
for client_id, client_segments in client_data.items():
    for segment_index, segment_data in enumerate(client_segments):
        wait_times = convert_wait_times_to_intervals(segment_data)
        power_consumption = calculate_power_consumption(start_time, wait_times, power_data)
        segment_name = 'Shufflenet' if segment_index == 0 else 'Squeezenet'
        print(f'Client {client_id} ({segment_name}) - Power Consumption per Phase: {power_consumption}')

# Example plot to visualize power consumption
def plot_power_consumption(power_data, start_time):
    plt.figure(figsize=(12, 6))
    plt.plot(power_data['Time(ms)'], power_data['Total Power (mW)'], label='Total Power (mW)')
    plt.axvline(x=start_time, color='r', linestyle='--', label='Start of Computation')
    plt.xlabel('Time (ms)')
    plt.ylabel('Power (mW)')
    plt.title('Power Consumption Over Time')
    plt.legend()
    plt.show()

plot_power_consumption(power_data, start_time)