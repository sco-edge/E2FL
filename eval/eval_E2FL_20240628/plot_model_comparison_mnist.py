import pickle
import matplotlib.pyplot as plt

# Load the pickle file
file_path = './client_data_2024-06-28.pickle'
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# Extract shufflenet data
shufflenet_data = []# {client_id: client_info for client_id, client_info in data.items() if 'shufflenet' in client_info}
for client_id in range(len(data)):
    shufflenet_data.append(data[client_id][0])

# 1. Compare Communication phase time after fit
fit_communication_times = {client_id: [comm['wait_time'] for comm in client_info['Communication (after fit)']] 
                           for client_id, client_info in shufflenet_data.items()}

# 2. Compare Communication phase time after evaluate
evaluate_communication_times = {client_id: [comm['wait_time'] for comm in client_info['Communication (after evaluate)']] 
                                for client_id, client_info in shufflenet_data.items()}

# 3. Compare network usage after fit
fit_network_usage = {client_id: [comm['network_usage'] for comm in client_info['Communication (after fit)']] 
                     for client_id, client_info in shufflenet_data.items()}

# 4. Compare network usage after evaluate
evaluate_network_usage = {client_id: [comm['network_usage'] for comm in client_info['Communication (after evaluate)']] 
                          for client_id, client_info in shufflenet_data.items()}

# 5. Compare time spent in fit phase
fit_times = {client_id: client_info['fit'] for client_id, client_info in shufflenet_data.items()}

# 6. Compare time spent in evaluate phase
evaluate_times = {client_id: client_info['evaluate'] for client_id, client_info in shufflenet_data.items()}

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

# Plot Communication phase time after fit
plot_comparison(fit_communication_times, 'Communication Phase Time After Fit', 'Time (s)')

# Plot Communication phase time after evaluate
plot_comparison(evaluate_communication_times, 'Communication Phase Time After Evaluate', 'Time (s)')

# Plot network usage after fit
def plot_network_usage(data, title):
    plt.figure(figsize=(10, 6))
    for client_id, usages in data.items():
        sent = [usage['sent'] for usage in usages]
        recv = [usage['recv'] for usage in usages]
        plt.plot(sent, label=f'Client {client_id} - Sent')
        plt.plot(recv, label=f'Client {client_id} - Received')
    plt.title(title)
    plt.xlabel('Round')
    plt.ylabel('Network Usage (bytes)')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_network_usage(fit_network_usage, 'Network Usage After Fit')

# Plot network usage after evaluate
plot_network_usage(evaluate_network_usage, 'Network Usage After Evaluate')

# Plot time spent in fit phase
plot_comparison(fit_times, 'Time Spent in Fit Phase', 'Time (s)')

# Plot time spent in evaluate phase
plot_comparison(evaluate_times, 'Time Spent in Evaluate Phase', 'Time (s)')
