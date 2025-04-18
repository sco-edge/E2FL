import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
from datetime import datetime, timedelta

#############################################################################################

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

#############################################################################################

def calculate_time_duration(data, filename):
    output = []

    # Extract timestamp from the filename
    filename_split = filename.split('_')
    filename_timestamp_str = filename_split[-2] + '_'+ filename_split[-1].replace('.txt', '')
    filename_timestamp = datetime.strptime(filename_timestamp_str, '%Y-%m-%d_%H-%M-%S')

    # Convert the given timestamp value to a datetime object
    given_timestamp = datetime.fromtimestamp(data['init'])

    # Calculate the time difference
    time_difference = given_timestamp - filename_timestamp
    output.append(time_difference.seconds)

    num_rounds = len(data['Communication (after fit)'])
    for round_i in range(num_rounds): 
        output.append(data['fit'][round_i])
        output.append(data['Communication (after fit)'][round_i]['wait_time'])
        output.append(data['evaluate'][round_i])
        output.append(data['Communication (after evaluate)'][round_i]['wait_time'])

    return output

# output; init, fit, comm_fit, eval, comm_eval
def convert_timescale(power_data, times):
    '''
    power monitor operates in milliseconds,
    but FL logging operates in seconds.
    Find array index in power_data according to time_arr
    '''
    # times_index
    output_index = []

    current_time = times[0] # init
    for data_i in range(len(power_data)):
        time_power = power_data[data_i][0] # Time(ms)
        if current_time <= time_power:
            output_index.append(data_i-1)
            if len(output_index) == len(times):
                 break
            current_time += times[len(output_index)]
    return output_index

def plot_power_consumption(data_time, data_power, time_i, filename='./plot_p_m3.png'):
    plt.figure(figsize=(12, 8))
    
    plt.plot(data_power[:time_i[-1]])
    
    for index in time_i:
        plt.axvline(x=index, color='r', linestyle='--')
    
    # Labeling the plot
    plt.xlabel('Time (ms)')
    plt.ylabel('Power (mW)')
    plt.title('Power Consumption Over Time')
    plt.legend()
    #plt.grid(True)
    plt.savefig(filename)

def plot_power_avg(data_power1, data_power2, phase_i1, phase_i2, filename):
    avg_power1 = []
    avg_power1.append(np.mean(data_power1[:phase_i1[0]]))
    pre_ind = phase_i1[0]
    for ind in phase_i1[1:]:
        avg_power1.append(np.mean(data_power1[pre_ind:ind]))
        pre_ind = ind

    avg_power2 = []
    avg_power2.append(np.mean(data_power2[:phase_i2[0]]))
    pre_ind = phase_i2[0]
    for ind in phase_i2[1:]:
        avg_power2.append(np.mean(data_power2[pre_ind:ind]))
        pre_ind = ind

    # init, fit, comm_fit, eval, comm_eval (1, 2, 3, 4), (5, 6, 7, 8)
    num_rounds = len(phase_i1) // 4
    avg_power_shuf = []
    avg_power_sqez = []
    avg_power_shuf.append(avg_power1[0])
    avg_power_sqez.append(avg_power2[0])

    for phase_ind in (1, 2, 3, 4):
        temp1 = 0
        temp2 = 0
        for round in range(num_rounds):
            temp1 += avg_power1[4*round + phase_ind]
            temp2 += avg_power2[4*round + phase_ind]
        avg_power_shuf.append(np.mean(temp1))
        avg_power_sqez.append(np.mean(temp2))
    
    plt.figure(figsize=(12, 8))
    width = 0.35  # the width of the bars
    phases = np.arange(1, len(avg_power_shuf) + 1)

    bar1 = plt.bar(phases - width/2, avg_power_shuf, width, color = color_labels[0], label='Shufflenet')
    bar2 = plt.bar(phases + width/2, avg_power_sqez, width, color = color_labels[1], label='Squeezenet')

    max_value = max(max(avg_power_shuf), max(avg_power_sqez))
    plt.ylim(0, max_value * 1.2)
    
    # Labeling the plot
    plt.xlabel('Federated Leearning Phase')
    plt.xticks(phases, ['init', 'fit', 'comm (fit)', 'eval', 'comm (eval)'])
    plt.ylabel('Average Power (mW)')
    #plt.title('Power Consumption Over Time')
    plt.legend(loc='best', fontsize=14)
    #plt.grid(True)
    plt.savefig(filename)

def plot_power_sum(data_power1, data_power2, phase_i1, phase_i2, filename):
    avg_power1 = []
    avg_power1.append(np.sum(data_power1[:phase_i1[0]]))
    pre_ind = phase_i1[0]
    for ind in phase_i1[1:]:
        avg_power1.append(np.sum(data_power1[pre_ind:ind]))
        pre_ind = ind

    avg_power2 = []
    avg_power2.append(np.sum(data_power2[:phase_i2[0]]))
    pre_ind = phase_i2[0]
    for ind in phase_i2[1:]:
        avg_power2.append(np.sum(data_power2[pre_ind:ind]))
        pre_ind = ind

    # init, fit, comm_fit, eval, comm_eval (1, 2, 3, 4), (5, 6, 7, 8)
    num_rounds = len(phase_i1) // 4
    avg_power_shuf = []
    avg_power_sqez = []
    avg_power_shuf.append(avg_power1[0])
    avg_power_sqez.append(avg_power2[0])

    for phase_ind in (1, 2, 3, 4):
        temp1 = 0
        temp2 = 0
        for round in range(num_rounds):
            temp1 += avg_power1[4*round + phase_ind]
            temp2 += avg_power2[4*round + phase_ind]
        avg_power_shuf.append(np.mean(temp1))
        avg_power_sqez.append(np.mean(temp2))
    
    plt.figure(figsize=(12, 8))
    width = 0.35  # the width of the bars
    phases = np.arange(1, len(avg_power_shuf) + 1)

    bar1 = plt.bar(phases - width/2, avg_power_shuf, width, color = color_labels[0], label='Shufflenet')
    bar2 = plt.bar(phases + width/2, avg_power_sqez, width, color = color_labels[1], label='Squeezenet')

    max_value = max(max(avg_power_shuf), max(avg_power_sqez))
    plt.ylim(0, max_value * 1.4)
    
    # Labeling the plot
    plt.xlabel('Federated Leearning Phase')
    plt.xticks(phases, ['init', 'fit', 'comm (fit)', 'eval', 'comm (eval)'])
    plt.ylabel('Total Power (mW)')
    #plt.title('Power Consumption Over Time')
    plt.legend(loc='best', fontsize=12)
    #plt.grid(True)
    plt.savefig(filename)

#############################################################################################

# Load power consumption data
power_data_path_shuf = '../../../eval_E2FL/E2FL_20240628_175814.csv'
power_data_shuf = pd.read_csv(power_data_path_shuf)#, delimiter='\s+', names=["Time(ms)", "USB(mA)", "Aux(mA)", "USB Voltage(V)"])

power_data_path_seqz = './power_measure_2024-06-28_18-59-27.pickle'
with open(power_data_path_seqz, 'rb') as f:
    power_data_seqz = pickle.load(f)

# Function to calculate total power (mW)
power_data_shuf['Total Power (mW)'] = power_data_shuf['USB(mA)'] * power_data_shuf['USB Voltage(V)']
power_data_shuf = power_data_shuf.to_numpy()
# 0~13114289; each index has 6 [Time(ms), USB(mA), Aux(mA), USB Voltage(V), Unamed, Total Power (mW)]

power_data_seqz.append([a * b for a ,b in zip(power_data_seqz[2], power_data_seqz[5])])
# 0~909523; each index has 6 [Time(ms), Aux(mA), USB(mA), Unamed, USB Voltage(V), Total Power (mW)]


# Load client data
client_data_path = './updated_client_data.pkl'
shufflenet_info = './fl_info_0_mnist_2024-06-28_18-07-40.txt'
squeezenet_info = './fl_info_0_mnist_2024-06-28_18-57-19.txt'
with open(client_data_path, 'rb') as f:
    client_data = pickle.load(f)
client_data = client_data[0]
client_data_shuf = client_data[0]
client_data_seqz = client_data[1]

num_rounds = max(len(client_data_seqz['fit']), len(client_data_shuf['fit']))
shuf_time = calculate_time_duration(client_data_shuf, shufflenet_info) # seconds
seqz_time = calculate_time_duration(client_data_seqz, squeezenet_info) # seconds

shuf_time_i = convert_timescale(power_data_shuf, shuf_time)
seqz_time_i = convert_timescale(power_data_shuf, seqz_time)

plot_power_consumption(power_data_shuf[:,0], power_data_shuf[:,5]*-1, shuf_time_i, filename='./plot_p_m3_shufflenet.png')
plot_power_consumption(power_data_seqz[0], np.array(power_data_seqz[6])*-1, seqz_time_i, filename='./plot_p_m3_squeezenet.png')
plot_power_avg(power_data_shuf[:,5]*-1, np.array(power_data_seqz[6])*-1, shuf_time_i, seqz_time_i, filename='./plot_p_m3_avg_bar.png')
plot_power_sum(power_data_shuf[:,5]*-1, np.array(power_data_seqz[6])*-1, shuf_time_i, seqz_time_i, filename='./plot_p_m3_sum_bar.png')
# init


# fit


# Communication (after fit)


# evaluate

# Communication (after evaluate)