import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./PowerBox_6721_-8585021977281168046.csv', header=None)

timescale = 0.2 # ms = 0.001s (1/1000)
# 0.0002 s 
#print(df[2])
#print(df[2][2])

time_series = df[0].tolist()
avg_main_power = df[2].tolist()[1:]
avg_main_power = [float(i) for i in avg_main_power]
avg_main_power2 = []

for index in range(len(avg_main_power)):
    if index > len(avg_main_power):
        break
    temp = avg_main_power[index:index+100]
    avg_main_power2.append(np.average(temp))
    index = index + 99

plt.plot(avg_main_power2[100000:]) #20 seconds
plt.show()