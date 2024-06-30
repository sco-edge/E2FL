import argparse
import re
import os
from datetime import datetime
from collections import defaultdict
import pickle

def parse_log_file(file_path, current_client):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    client_data = {
            'init': None,
            'fit': [],
            'Communication (after fit)': [],
            'evaluate': [],
            'Communication (after evaluate)': []
        }


    first_wifi_end_recorded = False
    skip_first_evaluation_phase = True
    current_phase = {}
    current_round = -1


    for line in lines:
        if 'Wi-Fi end:' in line:
            wifi_end_time = float(re.findall(r'\d+\.\d+', line)[0])
            if not first_wifi_end_recorded:
                client_data['init'] = wifi_end_time
                first_wifi_end_recorded = True
                continue
            elif 'fit_wait_time' not in current_phase:
                fit_wait_time = wifi_end_time - current_phase['wifi_start']
                current_phase['fit_wait_time'] = fit_wait_time
            else:
                eval_wait_time = wifi_end_time - current_phase['wifi_start']
                current_phase['eval_wait_time'] = eval_wait_time
        elif 'Wi-Fi start:' in line:
            current_phase['wifi_start'] = float(re.findall(r'\d+\.\d+', line)[0])
        elif 'Computation phase (wlan0):' in line or 'Computation phase (wlp1s0):' in line:
            network_usage = [int(x) for x in re.findall(r'\d+', line)]
            print(f"Computation network usage: {network_usage}")  # Debugging: Print parsed network usage
            if len(network_usage) >= 2:
                current_phase['network_usage'] = {'sent': network_usage[-2], 'recv': network_usage[-1]}
            
            client_data['Communication (after fit)'].append({
                    'wait_time': current_phase['fit_wait_time'],
                    'network_usage': {'sent': current_phase['network_usage']['sent'], 'recv': current_phase['network_usage']['recv']}
                })
        elif 'Evaluation phase (wlan0):' in line or 'Evaluation phase (wlp1s0):' in line:
            if skip_first_evaluation_phase:
                skip_first_evaluation_phase = False
                continue
            network_usage = [int(x) for x in re.findall(r'\d+', line)]
            print(f"Evaluation network usage: {network_usage}")  # Debugging: Print parsed network usage
            if len(network_usage) >= 2:
                current_phase['network_usage'] = {'sent': network_usage[-2], 'recv': network_usage[-1]}
            
            client_data['Communication (after evaluate)'].append({
                    'wait_time': current_phase['eval_wait_time'],
                    'network_usage': {'sent': current_phase['network_usage']['sent'], 'recv': current_phase['network_usage']['recv']}
                })
            current_phase = {}
            current_round += 1
        elif 'Computation pahse completed in' in line:  # Corrected typo
            computation_time = float(re.findall(r'\d+\.\d+', line)[0])
            client_data['fit'].append(computation_time)
        elif 'Evaluation pahse completed in' in line:  # Corrected typo
            evaluation_time = float(re.findall(r'\d+\.\d+', line)[0])
            client_data['evaluate'].append(evaluation_time)

    return client_data

def analyze_client_data(client_ids, file_prefix, file_date1, file_date2):
    all_clients_data = {}
    for client_id in client_ids:
        file_path1 = f"{file_prefix}{client_id}_mnist_{file_date1[client_id]}.txt"
        file_path2  = f"{file_prefix}{client_id}_mnist_{file_date2[client_id]}.txt"
        all_clients_data[client_id] = []
        if os.path.exists(file_path1):
            client_data = parse_log_file(file_path1, client_id)
            all_clients_data[client_id].append(client_data)
        else:
            print(f"Log file for Client {client_id} not found.")
        if os.path.exists(file_path2):
            client_data = parse_log_file(file_path2, client_id)
            all_clients_data[client_id].append(client_data)
        else:
            print(f"Log file for Client {client_id} not found.")

    return all_clients_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Federated Learning log files.")
    parser.add_argument('--clients', type=int, nargs='+', default=[0, 1, 2, 3], help='List of client IDs to analyze.')
    args = parser.parse_args()

    file_date1 = [
        '2024-06-28_18-07-40',
        '2024-06-28_18-07-32',
        '2024-06-28_18-07-16',
        '2024-06-28_18-07-11'
    ]
    file_date2 = [
        '2024-06-28_18-57-19',
        '2024-06-28_18-57-32',
        '2024-06-28_18-57-03',
        '2024-06-28_18-57-19'
    ]

    file_prefix = "./fl_info_"

    all_clients_data = analyze_client_data(args.clients, file_prefix, file_date1, file_date2)


    with open('./client_data_2024-06-28.pickle', 'wb') as f:
        pickle.dump(all_clients_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Data has been saved to client_data_2024-06-28.pickle")

    # Output the parsed data
    for client_id, datas in all_clients_data.items():
        print(f"Client {client_id} Data:")
        print("\tModel")
        for data in datas:
            for key, value in data.items():
                print(f"{key}: {value}")
            print("\n")
    

# python analyze_fl_logs.py 2024-06-28 --clients 0 1 2 3
