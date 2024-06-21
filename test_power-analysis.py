import subprocess, os, logging, time, socket, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pickle


parser = argparse.ArgumentParser(description="Flower Embedded devices")
parser.add_argument(
    "--file_path1",
    type=str,
    required=True,
    help="Path to the first dataset file",
)
parser.add_argument(
    "--file_path2",
    type=str,
    required=True,
    help="Path to the second dataset file"
)
    

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


def load_and_process_data(file_path):
    # Load the data
    data = pd.read_csv(file_path)
    
    # Calculate power consumption (Power = USB Current * USB Voltage)
    data['Power(mW)'] = data['USB(mA)'] * data['USB Voltage(V)']
    
    return data

def calculate_energy_per_time(file_path):
    # Load the data
    data = pd.read_csv(file_path)
    
    # Calculate power consumption (Power = USB Current * USB Voltage)
    data['Power(mW)'] = data['USB(mA)'] * data['USB Voltage(V)']
    
    # Calculate the energy for each time interval (Energy = Power * Time interval)
    data['Time_diff(s)'] = np.nan_to_num(np.diff(data['Time(ms)'])) / 1000  # Convert ms to seconds
    data['Energy(mJ)'] = data['Power(mW)'] * data['Time_diff(s)']  # Energy in millijoules
    
    return data

def calculate_mean_std(data, label):
    avg_power = (data['Power(mW)'].to_numpy()).mean()
    std_power = (data['Power(mW)'].to_numpy()).std()
    total_energy = (np.dot(data['Power(mW)'].to_numpy(), np.nan_to_num((np.diff(data['Time(ms)'].to_numpy())) / 1000))).sum()  # Total energy in mJ
    
    print(f'{label} - Average Power Consumption: {avg_power:.2f} mW, Std Dev: {std_power:.2f} mW, Total Energy: {total_energy:.2f} mJ')
    
    return avg_power, std_power, total_energy 

def calculate_mean_sum(data):
    avg_power = (data['Power(mW)'].to_numpy()).mean()
    total_energy = (data['Energy(mJ)'].to_numpy()).sum()
    
    return avg_power, total_energy

def calculate_cumulative_energy(data):
    # Calculate the energy for each time interval (Energy = Power * Time interval)
    data['Time_diff(s)'] = np.nan_to_num(np.diff(data['Time(ms)'].to_numpy())) / 1000  # Convert ms to seconds
    data['Energy(mJ)'] = np.dot(data['Power(mW)'].to_numpy(), data['Time_diff(s)'])  # Energy in millijoules
    
    # Calculate cumulative energy
    data['Cumulative_Energy(mJ)'] = (data['Energy(mJ)'].to_numpy()).cumsum()
    
    return data

def plot_power_consumption(data1, data2, label1, label2):
    plt.figure(figsize=(12, 6))
    
    # Plot the power consumption for the first dataset
    plt.plot(data1['Time(ms)'].to_numpy(), data1['Power(mW)'].to_numpy(), label=label1)
    
    # Plot the power consumption for the second dataset
    plt.plot(data2['Time(ms)'].to_numpy(), data2['Power(mW)'].to_numpy(), label=label2)
    
    # Labeling the plot
    plt.xlabel('Time (ms)')
    plt.ylabel('Power (mW)')
    plt.title('Power Consumption Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('./fig/plot_power_consumption.png')

def plot_comparison_bar_graph(stats1, stats2, labels):
    # Unpack statistics
    avg_power1, std_power1, total_energy1 = stats1
    avg_power2, std_power2, total_energy2 = stats2
    
    # Bar plot for average power and standard deviation
    plt.figure(figsize=(12, 6))
    x_labels = [labels[0], labels[1]]
    avg_powers = [avg_power1, avg_power2]
    std_devs = [std_power1, std_power2]
    
    x = range(len(x_labels))
    
    plt.subplot(1, 2, 1)
    plt.bar(x, avg_powers, yerr=std_devs, capsize=5, color=['blue', 'green'])
    plt.xticks(x, x_labels)
    plt.xlabel('Datasets')
    plt.ylabel('Power (mW)')
    plt.title('Average Power Consumption and Std Dev')

    # Bar plot for total energy consumption
    plt.subplot(1, 2, 2)
    total_energies = [total_energy1, total_energy2]
    plt.bar(x, total_energies, color=['blue', 'green'])
    plt.xticks(x, x_labels)
    plt.xlabel('Datasets')
    plt.ylabel('Total Energy (mJ)')
    plt.title('Total Energy Consumption')
    
    plt.tight_layout()
    plt.savefig('./fig/plot_comparison_bar_graph.png')

def plot_cumulative_energy(data1, data2, label1, label2):
    plt.figure(figsize=(12, 6))
    
    # Plot the cumulative energy for the first dataset
    plt.plot(data1['Time(ms)'].to_numpy(), data1['Cumulative_Energy(mJ)'], label=label1)
    
    # Plot the cumulative energy for the second dataset
    plt.plot(data2['Time(ms)'].to_numpy(), data2['Cumulative_Energy(mJ)'], label=label2)
    
    # Labeling the plot
    plt.xlabel('Time (ms)')
    plt.ylabel('Cumulative Energy (mJ)')
    plt.title('Cumulative Energy Consumption Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('./fig/plot_cumulative_energy.png')

def plot_boxplot(data1, data2, label1, label2):
    # Combine data for box plot
    data1['Dataset'] = label1
    data2['Dataset'] = label2
    combined_data = pd.concat([data1, data2])
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Dataset', y='Energy(mJ)', data=combined_data)
    plt.title('Energy Consumption Distribution')
    plt.ylabel('Energy (mJ)')
    plt.savefig('./fig/plot_boxplot.png')

def plot_density(data1, data2, label1, label2):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data1['Energy(mJ)'].to_numpy(), label=label1, fill=True, alpha=0.5)
    sns.kdeplot(data2['Energy(mJ)'].to_numpy(), label=label2, fill=True, alpha=0.5)
    plt.title('Energy Consumption Density')
    plt.xlabel('Energy (mJ)')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('./fig/plot_density.png')

def plot_3d_scatter(data1, data2, label1, label2):
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(121, projection='3d')
    
    # 3D scatter plot for the first dataset
    ax.scatter(data1['Time(ms)'].to_numpy(), data1['Power(mW)'].to_numpy(), data1['Cumulative_Energy(mJ)'].to_numpy(), label=label1)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Power (mW)')
    ax.set_zlabel('Cumulative Energy (mJ)')
    ax.set_title(label1)
    
    ax2 = fig.add_subplot(122, projection='3d')
    
    # 3D scatter plot for the second dataset
    ax2.scatter(data2['Time(ms)'].to_numpy(), data2['Power(mW)'].to_numpy(), data2['Cumulative_Energy(mJ)'].to_numpy(), label=label2, color='orange')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Power (mW)')
    ax2.set_zlabel('Cumulative Energy (mJ)')
    ax2.set_title(label2)
    
    plt.savefig('./fig/plot_3d_scatter.png')

def plot_3d_line(data1, data2, label1, label2):
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(121, projection='3d')
    
    # 3D line plot for the first dataset
    ax.plot(data1['Time(ms)'].to_numpy(), data1['Power(mW)'].to_numpy(), data1['Cumulative_Energy(mmkdirJ)'].to_numpy(), label=label1)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Power (mW)')
    ax.set_zlabel('Cumulative Energy (mJ)')
    ax.set_title(label1)
    
    ax2 = fig.add_subplot(122, projection='3d')
    
    # 3D line plot for the second dataset
    ax2.plot(data2['Time(ms)'].to_numpy(), data2['Power(mW)'].to_numpy(), data2['Cumulative_Energy(mJ)'].to_numpy(), label=label2, color='orange')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Power (mW)')
    ax2.set_zlabel('Cumulative Energy (mJ)')
    ax2.set_title(label2)
    
    plt.savefig('./fig/plot_3d_line.png')


def plot_comparison_bar_chart(stats1, stats2, labels):
    avg_power1, total_energy1 = stats1
    avg_power2, total_energy2 = stats2
    
    labels = [f'{labels[0]}', f'{labels[1]}']
    avg_powers = [avg_power1, avg_power2]
    total_energies = [total_energy1, total_energy2]
    
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Bar plot for average power consumption
    ax[0].bar(x - width/2, avg_powers, width, label='Avg Power (mW)', color='b')
    ax[0].set_ylabel('Avg Power (mW)')
    ax[0].set_title('Average Power Consumption')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels)

    # Bar plot for total energy consumption
    ax[1].bar(x + width/2, total_energies, width, label='Total Energy (mJ)', color='r')
    ax[1].set_ylabel('Total Energy (mJ)')
    ax[1].set_title('Total Energy Consumption')
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels)

    plt.tight_layout()
    plt.savefig('./fig/plot_comparison_bar_chart.png')

if __name__ == "__main__":
    
    args = parser.parse_args()
    plt.rcParams.update(params)
    plt.tight_layout()

    # Load and process the datasets
    data1 = load_and_process_data(args.file_path1)
    data2 = load_and_process_data(args.file_path2)

    # Calculate statistics for both datasets
    stats1 = calculate_mean_std(data1, 'Dataset 1 (CIFAR10)')
    stats2 = calculate_mean_std(data2, 'Dataset 2 (Another Dataset)')

    # Plot and compare the power consumption
    plot_power_consumption(data1, data2, 'Dataset 1 (CIFAR10)', 'Dataset 2 (Another Dataset)')

    # Plot comparison bar graph for average power, std deviation, and total energy
    plot_comparison_bar_graph(stats1, stats2, ['Dataset 1 (CIFAR10)', 'Dataset 2 (Another Dataset)'])



    # Calculate cumulative energy consumption for both datasets
    data1 = calculate_cumulative_energy(data1)
    data2 = calculate_cumulative_energy(data2)

    # Plot 3D scatter plot
    #plot_3d_scatter(data1, data2, 'Dataset 1 (CIFAR10)', 'Dataset 2 (Another Dataset)')

    # Plot 3D line plot
    #plot_3d_line(data1, data2, 'Dataset 1 (CIFAR10)', 'Dataset 2 (Another Dataset)')

    # Plot and compare the cumulative energy consumption
    plot_cumulative_energy(data1, data2, 'Dataset 1 (CIFAR10)', 'Dataset 2 (Another Dataset)')




    # Load the new set of datasets for the latest comparison
    latest_complete_file_paths = [
        "./data_0_2024-06-20_16-41-13.pickle",
        "./data_1_2024-06-20_16-41-13.pickle",
        "./data_2_2024-06-20_16-41-13.pickle",
        "./data_3_2024-06-20_16-41-13.pickle"
    ]

    latest_complete_data_frames = []

    for file_path in latest_complete_file_paths:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            latest_complete_data_frames.append(data)

    # Extracting the network I/O statistics
    latest_complete_network_stats = []

    for data in latest_complete_data_frames:
        latest_complete_network_stats.append(data)

    # Creating a DataFrame for the latest complete datasets comparison
    latest_complete_comparison_df = pd.DataFrame(latest_complete_network_stats, index=[f"Client {i+1}" for i in range(len(latest_complete_network_stats))])

    # Plotting the network I/O data for the latest complete datasets
    latest_complete_comparison_df[['bytes_sent', 'bytes_recv']].plot(kind='bar', figsize=(14, 8))
    plt.xlabel('Clients')
    plt.ylabel('Bytes')
    plt.title('Network I/O Usage Comparison Among Different FL Datasets (Latest Complete Set)')
    plt.legend(['Bytes Sent', 'Bytes Received'])
    plt.grid(True)
    plt.savefig('./fig/networkIO.png')

    # Load and process the datasets
    data1 = calculate_energy_per_time(args.file_path1)
    data2 = calculate_energy_per_time(args.file_path2)

    # Calculate statistics for both datasets
    stats1 = calculate_mean_sum(data1)
    stats2 = calculate_mean_sum(data2)

    # Plot comparison bar chart
    plot_comparison_bar_chart(stats1, stats2, ['Dataset 1 (CIFAR10)', 'Dataset 2 (Another Dataset)'])


    # Load the dataset paths for comparison
    dataset_paths_1 = [
        "./data_0_2024-06-20_15-10-29.pickle",
        "./data_1_2024-06-20_15-10-29.pickle",
        "./data_2_2024-06-20_15-10-29.pickle",
        "./data_3_2024-06-20_15-10-29.pickle"
    ]

    dataset_paths_2 = [
        "./data_0_2024-06-20_16-41-13.pickle",
        "./data_1_2024-06-20_16-41-13.pickle",
        "./data_2_2024-06-20_16-41-13.pickle",
        "./data_3_2024-06-20_16-41-13.pickle"
    ]

    data_frames_1 = []
    data_frames_2 = []

    for file_path in dataset_paths_1:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            data_frames_1.append(data)

    for file_path in dataset_paths_2:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            data_frames_2.append(data)

    # Extracting the network I/O statistics for both datasets
    network_stats_1 = []
    network_stats_2 = []

    for data in data_frames_1:
        network_stats_1.append(data)

    for data in data_frames_2:
        network_stats_2.append(data)

    # Creating DataFrames for both datasets
    comparison_df_1 = pd.DataFrame(network_stats_1, index=[f"Client {i+1}" for i in range(len(network_stats_1))])
    comparison_df_2 = pd.DataFrame(network_stats_2, index=[f"Client {i+1}" for i in range(len(network_stats_2))])

    # Plotting the network I/O data for both datasets
    fig, axes = plt.subplots(2, 1, figsize=(14, 16))

    comparison_df_1[['bytes_sent', 'bytes_recv']].plot(kind='bar', ax=axes[0])
    axes[0].set_xlabel('Clients')
    axes[0].set_ylabel('Bytes')
    axes[0].set_title('Network I/O Usage for Dataset 1')
    axes[0].legend(['Bytes Sent', 'Bytes Received'])
    axes[0].grid(True)

    comparison_df_2[['bytes_sent', 'bytes_recv']].plot(kind='bar', ax=axes[1])
    axes[1].set_xlabel('Clients')
    axes[1].set_ylabel('Bytes')
    axes[1].set_title('Network I/O Usage for Dataset 2')
    axes[1].legend(['Bytes Sent', 'Bytes Received'])
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('./fig/plot_networkIO_difference.png')
