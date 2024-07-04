import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load the updated pickle data file
file_path = './updated_client_data.pkl'
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# Function to extract and compare data
def extract_and_compare_data(data, segment_index=0):
    # Initialize dictionaries to hold the comparison data
    communication_after_fit_times, communication_after_evaluate_times, fit_network_usage_nethogs, evaluate_network_usage_nethogs, fit_times, evaluate_time, = {}, {}, {}, {}, {}, {}

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
        evaluate_time[client_id] = client_segment['evaluate']

    return (communication_after_fit_times, communication_after_evaluate_times, fit_network_usage_nethogs, evaluate_network_usage_nethogs, fit_times, evaluate_time)

# Function to plot 3D graphs
def plot_3d_comparison(data, title, ylabel, filename):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    client_ids = list(data.keys())
    for client_id in client_ids:
        rounds = np.arange(len(data[client_id]))
        values = data[client_id]
        ax.plot(client_ids, rounds, values, label=f'Client {client_id}')
    
    ax.set_xlabel('Client ID')
    ax.set_ylabel('Round')
    ax.set_zlabel(ylabel)
    ax.set_title(title)
    plt.legend()
    #plt.show()
    plt.savefig('./'+filename)

# Function to calculate and plot average values
def plot_average_values(data1, data2, title, ylabel, filename):
    shufflenet_avg = []
    squeezenet_avg = []
    rounds = np.arange(max(len(data1[0]), len(data2[0])))
    
    for round_num in rounds:
        shufflenet_values = [data1[client_id][round_num] for client_id in data1 if round_num < len(data1[client_id])]
        squeezenet_values = [data2[client_id][round_num] for client_id in data2 if round_num < len(data2[client_id])]
        
        shufflenet_avg.append(np.mean(shufflenet_values))
        squeezenet_avg.append(np.mean(squeezenet_values))
    
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, shufflenet_avg, label='Shufflenet')
    plt.plot(rounds, squeezenet_avg, label='Squeezenet')
    plt.title(title)
    plt.xlabel('Round')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig('./'+filename)

# Extract the data for shufflenet (1800 segment) and squeezenet (1900 segment)
(communication_after_fit_times_shufflenet, communication_after_evaluate_times_shufflenet, fit_network_usage_nethogs_shufflenet, evaluate_network_usage_nethogs_shufflenet, fit_times_shufflenet, evaluate_times_shufflenet) = extract_and_compare_data(data, segment_index=0)
(communication_after_fit_times_squeezenet, communication_after_evaluate_times_squeezenet, fit_network_usage_nethogs_squeezenet, evaluate_network_usage_nethogs_squeezenet, fit_times_squeezenet, evaluate_times_squeezenet) = extract_and_compare_data(data, segment_index=1)
'''
# Plot Communication phase time after fit
plot_3d_comparison(communication_after_fit_times_shufflenet, 'Communication Phase Time After Fit (Shufflenet)', 'Time (s)', 'plot_m_c_3d-comm_after_fit(shufflenet).png')

# Plot Communication phase time after evaluate
plot_3d_comparison(communication_after_evaluate_times_shufflenet, 'Communication Phase Time After Evaluate (Shufflenet)', 'Time (s)', 'plot_m_c_3d-comm_after_eval(shufflenet).png')

# Plot network usage after fit (nethogs)
plot_3d_comparison(fit_network_usage_nethogs_shufflenet, 'Network Usage After Fit (Shufflenet) - Nethogs', 'Network Usage (nethogs)', 'plot_m_c_3d-fit_nethogs(shufflenet).png')

# Plot network usage after evaluate (nethogs)
plot_3d_comparison(evaluate_network_usage_nethogs_shufflenet, 'Network Usage After Evaluate (Shufflenet) - Nethogs', 'Network Usage (nethogs)', 'plot_m_c_3d-eval_nethogs(shufflenet).png')

# Plot time spent in fit phase
plot_3d_comparison(fit_times_shufflenet, 'Time Spent in Fit Phase (Shufflenet)', 'Time (s)', 'plot_m_c_3d-fit(shufflenet).png')

# Plot time spent in evaluate phase
plot_3d_comparison(evaluate_times_shufflenet, 'Time Spent in Evaluate Phase (Shufflenet)', 'Time (s)', 'plot_m_c_3d-eval(shufflenet).png')
'''
# Plot average values for comparison between Shufflenet and Squeezenet
plot_average_values(communication_after_fit_times_shufflenet, communication_after_fit_times_squeezenet, 'Average Communication Phase Time After Fit', 'Time (s)', 'plot_m_c_3d-comm_after_fit_compare.png')
plot_average_values(communication_after_evaluate_times_shufflenet, communication_after_evaluate_times_squeezenet, 'Average Communication Phase Time After Evaluate', 'Time (s)', 'plot_m_c_3d-comm_after_eval_compare.png')
plot_average_values(fit_network_usage_nethogs_shufflenet, fit_network_usage_nethogs_squeezenet, 'Average Network Usage After Fit - Nethogs', 'Network Usage (nethogs)', 'plot_m_c_3d-fit_nethogs_compare.png')
plot_average_values(evaluate_network_usage_nethogs_shufflenet, evaluate_network_usage_nethogs_squeezenet, 'Average Network Usage After Evaluate - Nethogs', 'Network Usage (nethogs)', 'plot_m_c_3d-eval_nethogs_compare.png')
plot_average_values(fit_times_shufflenet, fit_times_squeezenet, 'Average Time Spent in Fit Phase', 'Time (s)', 'plot_m_c_3d-fit_compare.png')
plot_average_values(evaluate_times_shufflenet, evaluate_times_squeezenet, 'Average Time Spent in Evaluate Phase', 'Time (s)', 'plot_m_c_3d-eval_compare.png')