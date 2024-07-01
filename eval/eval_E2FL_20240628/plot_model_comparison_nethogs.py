import pickle
import matplotlib.pyplot as plt

# Load the updated pickle data file
file_path = '/mnt/data/updated_client_data.pkl'
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# Function to extract and compare data
def extract_and_compare_data(data, segment_index=0):
    # Initialize dictionaries to hold the comparison data
    communication_after_fit_times = {}
    communication_after_evaluate_times = {}
    fit_network_usage_nethogs = {}
    evaluate_network_usage_nethogs = {}
    fit_times = {}
    evaluate_times = {}

    # Extract data for shufflenet (1800 segment)
    for client_id, client_data in data.items():
        client_segment = client_data[segment_index]
        
        # Extract communication phase times
        communication_after_fit_times[client_id] = [comm['wait_time'] for comm in client_segment['Communication (after fit)']]
        communication_after_evaluate_times[client_id] = [comm['wait_time'] for comm in client_segment['Communication (after evaluate)']]
        
        # Extract network usage data (nethogs)
        fit_network_usage_nethogs[client_id] = [comm['network_usage']['nethogs'] for comm in client_segment['Communication (after fit)']]
        evaluate_network_usage_nethogs[client_id] = [comm['network_usage']['nethogs'] for comm in client_segment['Communication (after evaluate)']]
        
        # Extract computation times
        fit_times[client_id] = client_segment['fit']
        evaluate_times[client_id] = client_segment['evaluate']

    return (communication_after_fit_times, communication_after_evaluate_times, fit_network_usage_nethogs, evaluate_network_usage_nethogs, fit_times, evaluate_times)

# Plot the comparisons
def plot_comparison(data, title, ylabel):
    plt.figure(figsize=(10, 6))
    for client_id, values in data.items():
        plt.plot(values, label=f'Client {client_id}')
    plt.title(title)
    plt.xlabel('Round')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot network usage comparisons (nethogs)
def plot_network_usage(data, title):
    plt.figure(figsize=(10, 6))
    for client_id, usages in data.items():
        sent = [usage['sent'] for usage in usages]
        recv = [usage['recv'] for usage in usages]
        plt.plot(sent, label=f'Client {client_id} - Sent')
        plt.plot(recv, label=f'Client {client_id} - Received')
    plt.title(title)
    plt.xlabel('Round')
    plt.ylabel('Network Usage (nethogs)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Extract the data for shufflenet (1800 segment)
(communication_after_fit_times, communication_after_evaluate_times, fit_network_usage_nethogs, evaluate_network_usage_nethogs, fit_times, evaluate_times) = extract_and_compare_data(data, segment_index=0)

# Plot Communication phase time after fit
plot_comparison(communication_after_fit_times, 'Communication Phase Time After Fit (Shufflenet)', 'Time (s)')

# Plot Communication phase time after evaluate
plot_comparison(communication_after_evaluate_times, 'Communication Phase Time After Evaluate (Shufflenet)', 'Time (s)')

# Plot network usage after fit (nethogs)
plot_network_usage(fit_network_usage_nethogs, 'Network Usage After Fit (Shufflenet) - Nethogs')

# Plot network usage after evaluate (nethogs)
plot_network_usage(evaluate_network_usage_nethogs, 'Network Usage After Evaluate (Shufflenet) - Nethogs')

# Plot time spent in fit phase
plot_comparison(fit_times, 'Time Spent in Fit Phase (Shufflenet)', 'Time (s)')

# Plot time spent in evaluate phase
plot_comparison(evaluate_times, 'Time Spent in Evaluate Phase (Shufflenet)', 'Time (s)')