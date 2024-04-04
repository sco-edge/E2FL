from power import Monitor
import Monsoon.sampleEngine as sampleEngine

node_A_name = 'rpi3B+'
node_A_vout = 5.0
node_A_mode = "PyMonsoon"
node_A_triggerBool = True
node_A_numSamples = 5000
node_A_thld_high = 100
node_A_thld_low = 10
node_A_CSVbool = False#True
node_A_CSVname = "default"

rpi3B = Monitor.PowerMon(   node = node_A_name,
                            vout = node_A_vout,
                            mode = node_A_mode )
'''
rpi3B.setTrigger(   bool = node_A_triggerBool,
                    numSamples = node_A_numSamples,
                    thld_high = node_A_thld_high,
                    thld_low = node_A_thld_low )
'''
rpi3B.setCSVOutput( bool = node_A_CSVbool,
                    filename = node_A_CSVname)


rpi3B.startSampling()
samples = rpi3B.getSamples()

for i in range(len(samples[sampleEngine.channels.timeStamp])):
    timeStamp = samples[sampleEngine.channels.timeStamp][i]
    Current = samples[sampleEngine.channels.MainCurrent][i]
    print("Main current at time " + repr(timeStamp) + " is: " + repr(Current) + "mA")