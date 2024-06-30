import pickle
import re

# Load the updated pickle data file
file_path = './client_data_2024-06-28.pickle'
with open(file_path, 'rb') as f:
    data = pickle.load(f)
    
# Function to parse the nethogs log files
def parse_nethogs_log(file_path):
    network_data = []
    with open(file_path, 'r') as file:
        for line in file:
            if 'python3.10' in line:
                match = re.findall(r'\d+\.\d+', line)
                if len(match) >= 2:  # Ensure there are at least two matches
                    sent, recv = float(match[-2]), float(match[-1])
                    network_data.append({'sent': sent, 'recv': recv})
    return network_data

# Align network usage data with wait time
def align_network_usage_with_wait_time(network_data, wait_times):
    aligned_data = []
    for wait_time in wait_times:
        if network_data:
            aligned_data.append(network_data.pop(0))
        else:
            aligned_data.append({'sent': 0, 'recv': 0})
    return aligned_data

# Log file paths for each client
log_files = {
    0: ['./nethogs_log_0_20240628_1800.txt', './nethogs_log_0_20240628_1900.txt'],
    1: ['./nethogs_log_1_20240628_1800.txt', './nethogs_log_1_20240628_1900.txt'],
    2: ['./nethogs_log_2_20240628_1800.txt', './nethogs_log_2_20240628_1900.txt'],
    3: ['./nethogs_log_3_20240628_1800.txt', './nethogs_log_3_20240628_1900.txt'],
}

# Update data with network usage
for client_id, files in log_files.items():
    network_data = []
    for file in files:
        network_data.append(parse_nethogs_log(file))
    
    for net_i in range(len(network_data)):
        for phase in ['Communication (after fit)', 'Communication (after evaluate)']:
            wait_times = [comm['wait_time'] for comm in data[client_id][net_i][phase]]
            aligned_network_usage = align_network_usage_with_wait_time(network_data[net_i], wait_times)
            
            for idx, comm in enumerate(data[client_id][net_i][phase]):
                comm['network_usage']['nethogs'] = aligned_network_usage[idx]

# Save the updated data back to a pickle file
updated_file_path = './updated_client_data.pkl'
with open(updated_file_path, 'wb') as f:
    pickle.dump(data, f)

import shutil
shutil.move(updated_file_path, "./updated_client_data.pkl")
