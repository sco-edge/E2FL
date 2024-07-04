import pickle
import matplotlib.pyplot as plt
import numpy as np

hatch_labels = {	'Before': '.', \
                    'After': '\\', \
                    'A': '\\', \
                    'B': '////', \
                    'C': '-', \
                    'D': 'x', \
                    'Before_label': '..', \
                    'After_label': '\\\\', \
                    'A_label': '\\\\', \
                    'B_label': '////', \
                    'C_label': '--', \
                    'D_label': 'x' \
                }
color_labels = ['#CC3311', '#0077BB', '#EE3377', \
                '#009988', '#33BBEE', '#AA4499', \
                '#CC3311', '#0077BB'] #'#ff9f11', '#7adf1e'
def cm2inch(value):
    	return value/2.54

params = {'figure.figsize': (cm2inch(24), cm2inch(12)),
    'font.family': 'Times New Roman', #monospace
    'font.weight': 'bold',
    'font.size': 18,
    'lines.linewidth': 3,
    'lines.markersize': 8,
    'lines.markeredgewidth': 2,
    'markers.fillstyle': 'none',
    'axes.labelweight': 'bold',
    'axes.labelsize': 'large',
    'axes.xmargin': 0.05,
    'axes.grid': False,
    'grid.linewidth': 0.5,
    'grid.linestyle': '--',
    'legend.loc': 'upper right',
    'legend.fontsize': 16,
    'figure.autolayout': True,
    'savefig.transparent': True,
    }

plt.rcParams.update(params)
plt.tight_layout()

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
    num_clients = max(len(data1), len(data2))
    #num_rounds = max(len(data1[0]), len(data2[0]))
    clients = np.arange(1, num_clients + 1)
    #rounds = np.arange(1, num_rounds + 1)
    
    # client_id
    for client_id in range(num_clients):
        shufflenet_avg.append(np.mean(data1[client_id]))
        squeezenet_avg.append(np.mean(data2[client_id]))
    
    width = 0.35  # the width of the bars
    plt.figure(figsize=(10, 6))
    
    bar1 = plt.bar(clients - width/2, shufflenet_avg, width, color = color_labels[0], label='Shufflenet')
    bar2 = plt.bar(clients + width/2, squeezenet_avg, width, color = color_labels[1], label='Squeezenet')
    
    plt.title(title)
    plt.xlabel('Client')
    plt.ylabel(ylabel)
    plt.xticks(clients, ['RPi3B+ (1)', 'RPi3B+ (2)', 'RPi4B', 'RPi5'])
    plt.legend(loc='best', fontsize=12)
    #plt.grid(True)
    
    # Determine the max value to set the y-axis limit
    max_value = max(max(shufflenet_avg), max(squeezenet_avg))
    plt.ylim(0, max_value * 1.2)  # Increase y-axis limit by 5%

    # Adding the values on top of the bars
    for bar in bar1:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/8.0, yval, f'{yval:.2f}', va='bottom', fontsize=12)  # va: vertical alignment bar.get_width()/4.0
    
    for bar in bar2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/8.0, yval, f'{yval:.2f}', va='bottom', fontsize=12)  # va: vertical alignment bar.get_width()/4.0
    
    #plt.show()
    plt.savefig('./'+filename)


def plot_nethogs(data1, data2, title, ylabel, filename):
    shufflenet_sent_avg = []
    shufflenet_recv_avg = []
    squeezenet_sent_avg = []
    squeezenet_recv_avg = []
    #rounds = np.arange(max(len(data1[0]), len(data2[0])))
    clients = np.arange(max(len(data1), len(data2)))
    
    for client_id in clients:
        shufflenet_sent_values = [data1[client_id][round_num]['sent'] for round_num in range(len(data1[client_id])) if client_id < len(data1)]
        shufflenet_recv_values = [data1[client_id][round_num]['recv'] for round_num in range(len(data1[client_id])) if client_id < len(data1)]
        squeezenet_sent_values = [data2[client_id][round_num]['sent'] for round_num in range(len(data2[client_id])) if client_id < len(data2)]
        squeezenet_recv_values = [data2[client_id][round_num]['recv'] for round_num in range(len(data2[client_id])) if client_id < len(data2)]
        
        shufflenet_sent_avg.append(np.mean(shufflenet_sent_values))
        shufflenet_recv_avg.append(np.mean(shufflenet_recv_values))
        squeezenet_sent_avg.append(np.mean(squeezenet_sent_values))
        squeezenet_recv_avg.append(np.mean(squeezenet_recv_values))
    
    width = 0.2  # the width of the bars
    plt.figure(figsize=(10, 6))
    
    bar1 = plt.bar(clients - width, shufflenet_sent_avg, width, color = color_labels[0], label='Sent (Shufflenet)', hatch = hatch_labels['A'])
    bar2 = plt.bar(clients, shufflenet_recv_avg, width, color = color_labels[0], label='Recv (Shufflenet)', hatch = hatch_labels['B'])
    bar3 = plt.bar(clients + width, squeezenet_sent_avg, width, color = color_labels[1], label='Sent (Squeezenet)', hatch = hatch_labels['A'])
    bar4 = plt.bar(clients + 2 * width, squeezenet_recv_avg, width, color = color_labels[1], label='Recv (Squeezenet)', hatch = hatch_labels['B'])
    
    
    plt.title(title)
    plt.xticks(clients, ['RPi3B+ (1)', 'RPi3B+ (2)', 'RPi4B', 'RPi5'])
    plt.xlabel('Client')
    plt.ylabel(ylabel)
    plt.legend(loc='best', fontsize=12)
    #plt.grid(True)

    # Determine the max value to set the y-axis limit
    max_value = max(max(shufflenet_sent_avg), max(shufflenet_recv_avg), max(squeezenet_sent_avg), max(squeezenet_recv_avg))
    plt.ylim(0, max_value * 1.2)  # Increase y-axis limit by 5%

    # Adding the values on top of the bars
    for bar in bar1 + bar2 + bar3 + bar4:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/8.0, yval, f'{yval:.2f}', va='bottom', ha='center', fontsize=12)

    #plt.show()
    plt.savefig('./'+filename)

# Sample usage
# plot_nethogs(data1, data2, "Network Usage", "Bytes", "network_usage.png")


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
plot_average_values(fit_times_shufflenet, fit_times_squeezenet, 'Average Time Spent in Fit', 'Time (s)', 'plot_m_c_3d-fit_compare.png')
plot_average_values(evaluate_times_shufflenet, evaluate_times_squeezenet, 'Average Time Spent in Evaluate', 'Time (s)', 'plot_m_c_3d-eval_compare.png')
plot_nethogs(fit_network_usage_nethogs_shufflenet, fit_network_usage_nethogs_squeezenet, 'Average Network Usage After Fit - Nethogs', 'Network Usage (byte)', 'plot_m_c_3d-fit_nethogs_compare.png')
plot_nethogs(evaluate_network_usage_nethogs_shufflenet, evaluate_network_usage_nethogs_squeezenet, 'Average Network Usage After Evaluate - Nethogs', 'Network Usage (byte)', 'plot_m_c_3d-eval_nethogs_compare.png')