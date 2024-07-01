import re
import pickle
from datetime import datetime, timedelta

# Function to parse the nethogs log files
def parse_nethogs_log(file_path, wait_times):
    network_data = []
    wait_time_index = 0
    current_wait_time = wait_times[wait_time_index] if wait_times else None

    refresh_time = timedelta(seconds=1)
    current_time = datetime.fromtimestamp(0)

    with open(file_path, 'r') as file:
        for line in file:
            if 'Refreshing:' in line:
                if current_wait_time and current_time >= current_wait_time['start_time']:
                    if wait_time_index < len(wait_times) - 1:
                        wait_time_index += 1
                        current_wait_time = wait_times[wait_time_index]
                    else:
                        current_wait_time = None

                current_time += refresh_time

            if current_wait_time and 'python3.10' in line:
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

# Convert wait times to start times
def convert_wait_times_to_intervals(client_data):
    intervals = []
    init_time = datetime.fromtimestamp(client_data['init'])
    for comm in client_data['Communication (after fit)'] + client_data['Communication (after evaluate)']:
        start_time = init_time + timedelta(seconds=comm['wait_time'])
        intervals.append({'start_time': start_time})
    return intervals

# Load the updated pickle data file
file_path = './client_data_2024-06-28.pickle'
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# Log file paths for each client
log_files = {
    0: {'1800': './nethogs_log_0_20240628_1800.txt', '1900': './nethogs_log_0_20240628_1900.txt'},
    1: {'1800': './nethogs_log_1_20240628_1800.txt', '1900': './nethogs_log_1_20240628_1900.txt'},
    2: {'1800': './nethogs_log_2_20240628_1800.txt', '1900': './nethogs_log_2_20240628_1900.txt'},
    3: {'1800': './nethogs_log_3_20240628_1800.txt', '1900': './nethogs_log_3_20240628_1900.txt'},
}

# Update data with network usage
for client_id, file_paths in log_files.items():
    for segment, file_path in file_paths.items():
        index = 0 if segment == '1800' else 1
        wait_times = convert_wait_times_to_intervals(data[client_id][index])
        network_data = parse_nethogs_log(file_path, wait_times)
        
        for phase in ['Communication (after fit)', 'Communication (after evaluate)']:
            aligned_network_usage = align_network_usage_with_wait_time(network_data, wait_times)
            
            for idx, comm in enumerate(data[client_id][index][phase]):
                comm['network_usage']['nethogs'] = aligned_network_usage[idx]

# Save the updated data back to a pickle file
updated_file_path = './updated_client_data.pkl'
with open(updated_file_path, 'wb') as f:
    pickle.dump(data, f)

import shutil
shutil.move(updated_file_path, "./updated_client_data.pkl")