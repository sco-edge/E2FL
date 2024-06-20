import subprocess, os, logging, time, socket, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

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

def plot_power_consumption(data1, data2, label1, label2):
    plt.figure(figsize=(12, 6))
    
    # Plot the power consumption for the first dataset
    plt.plot(data1['Time(ms)'], data1['Power(mW)'], label=label1)
    
    # Plot the power consumption for the second dataset
    plt.plot(data2['Time(ms)'], data2['Power(mW)'], label=label2)
    
    # Labeling the plot
    plt.xlabel('Time (ms)')
    plt.ylabel('Power (mW)')
    plt.title('Power Consumption Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_statistics(data, label):
    avg_power = data['Power(mW)'].mean()
    std_power = data['Power(mW)'].std()
    total_energy = (data['Power(mW)'] * (data['Time(ms)'].diff().fillna(0) / 1000)).sum()  # Total energy in mJ
    
    print(f'{label} - Average Power Consumption: {avg_power:.2f} mW, Std Dev: {std_power:.2f} mW, Total Energy: {total_energy:.2f} mJ')
    
    return avg_power, std_power, total_energy

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
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare energy consumption between two datasets.")
    parser.add_argument('file_path1', type=str, help="Path to the first dataset file")
    parser.add_argument('file_path2', type=str, help="Path to the second dataset file")
    
    args = parser.parse_args()

    # Load and process the datasets
    data1 = load_and_process_data(args.file_path1)
    data2 = load_and_process_data(args.file_path2)

    # Calculate statistics for both datasets
    stats1 = calculate_statistics(data1, 'Dataset 1 (CIFAR10)')
    stats2 = calculate_statistics(data2, 'Dataset 2 (Another Dataset)')

    # Plot and compare the power consumption
    plot_power_consumption(data1, data2, 'Dataset 1 (CIFAR10)', 'Dataset 2 (Another Dataset)')

    # Plot comparison bar graph for average power, std deviation, and total energy
    plot_comparison_bar_graph(stats1, stats2, ['Dataset 1 (CIFAR10)', 'Dataset 2 (Another Dataset)'])



