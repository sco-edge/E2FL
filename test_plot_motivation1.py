import matplotlib.pyplot as plt
import numpy as np

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

'''
[tqdm function from tqdm.py]
100%|████████████████████████████████████████████████████████████████| 57/57 [02:43<00:00,  2.87s/it]
{elapsed}<{remaining}, seconds per iteration
-> {ealpsed} = mm:ss

# attempts: (rounds, epochs)
1st attempt: (3, 3)
'''
FL_latency = { 
    'fit':{
        'rpi3B+': [
            [[2*60+40, 2*60+44, 3*60+24], [3*60+17, 3*60+30, 3*60+38], [3*60+39, 3*60+45, 3*60+48]]
        ],
        'rpi4B': [
            [[1*60+26, 1*60+26, 1*60+29], [ 1*60+26, 1*60+29, 1*60+35], [1*60+27, 1*60+29, 1*60+34]]
        ]
    },
    'eval':{
        'rpi3B+': [
            [5, 5, 5]
        ],
        'rpi4B': [
            [2, 2, 2]
        ]
    },
    'server': {
        'time': [1866.4741201259985], # flower/src/py/flwr/server/server.py -> timeit module
        'loss': [[4.344347953796387, 3.749884247779846, 3.5971572399139404]],
        'accuracy': [[0.185, 0.255, 0.29]]
    }
    # https://flower.ai/docs/framework/tutorial-series-build-a-strategy-from-scratch-pytorch.html
}
'''
    rpi3B+        rpi4B       Desktop
real 0m0.018s   0m0.014s    0m0.004s
user 0m0.000s   0m0.000s    0m0.000s
sys  0m0.015s   0m0.011s    0m0.004s
'''
wrls_latency = {
    'rpi3B+':[0.018, 0.000, 0.015],
    'rpi4B':[0.014, 0.000, 0.011],
    'server':[0.004, 0.000, 0.004]
}

rpi3_fit = np.sum(FL_latency['fit']['rpi3B+'][0])
rpi3_fit_mean = np.mean(FL_latency['fit']['rpi3B+'][0])
rpi3_fit_var = np.var(FL_latency['fit']['rpi3B+'][0])
rpi3_fit_std = np.std(FL_latency['fit']['rpi3B+'][0])

rpi4_fit = np.sum(FL_latency['fit']['rpi4B'][0])
rpi4_fit_mean = np.mean(FL_latency['fit']['rpi4B'][0])
rpi4_fit_var = np.var(FL_latency['fit']['rpi4B'][0])
rpi4_fit_std = np.std(FL_latency['fit']['rpi4B'][0])

total = FL_latency['server']['time'][0]

# Fixing random state for reproducibility
#np.random.seed(19680801)

plt.rcParams.update(params)
fig, ax = plt.subplots()

# Example data
people = ('Pi 3B+', 'Pi 4B', 'Total')
y_pos = np.arange(len(people))
performance = [rpi3_fit, rpi4_fit, total]
error = [rpi3_fit_std, rpi4_fit_std, 0]

ax.barh(y_pos, performance, xerr=error, align='center', color='gray')
ax.set_yticks(y_pos, labels=people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Time in seconds')
# ax.set_title('How fast do you want to go today?')

plt.tight_layout()
plt.show()
fig.savefig('./1.png')
plt.close()

fig, ax = plt.subplots()
people = ('Comp.', 'Comm.', 'Wi-Fi Interface')
# rounds; https://flower.ai/docs/framework/example-pytorch-from-centralized-to-federated.html
x = np.arange(len(people))
values = [np.log(rpi4_fit_mean), np.log((total - rpi3_fit) / 3), 0.083] #83 ms

ax.bar(x, values, color='gray')
ax.set_xticks(x, people)
ax.set_ylabel('Time in seconds (in log)')

plt.tight_layout()
plt.show()
fig.savefig('./2.png')
plt.close()