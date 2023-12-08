import monsoon.LVPM as LVPM
import monsoon.sampleEngine as sampleEngine
import monsoon.Operations as op

'''
pip install monsoon
'''
class MyMon():
    def __init__(self, mode, node, vout, channel):
        self.mode = mode
        self.node = node
        self.vout = vout
        self.channel = channel
        if mode == "PyMonsoon":
            Mon = LVPM.Monsoon()
            Mon.setup_usb()

            Mon.setVout(5.5)
            engine = sampleEngine.SampleEngine(Mon)
            engine.enableCSVOutput("E2FL_20231208.csv")
            engine.ConsoleOutput(True)
            numSamples = 5000 # sample for one second
            engine.startSampling(numSamples)

            #Disable Main Channels
            engine.disableChannel(sampleEngine.channels.MainCurrent)
            engine.disableChannel(sampleEngine.channels.MainVoltage)

            #Enable USB channels
            engine.enableChannel(sampleEngine.channels.USBCurrent)
            engine.enableChannel(sampleEngine.channels.USBVoltage)

            #Enable AUX channels
            engine.enableChannel(sampleEngine.channels.AuxCurrent)

            #Set USB Pasthrough mode to 'on', since it defaults to 'auto' 
            #and will turn off when sampling mode begins
            Mon.setUSBPassthroughMode(op.USB_Passthrough.On)

            engine.enableCSVOutput("USB Example.csv")
            engine.startSampling(numSamples)

            #Don't stop based on sample count, continue until 
            #the trigger conditions have been satisfied.
            numSamples=sampleEngine.triggers.SAMPLECOUNT_INFINITE

            #Start when we exceed 100 mA
            engine.setStartTrigger(sampleEngine.triggers.GREATER_THAN,100)

            #Stop when we drop below 10 mA
            engine.setStopTrigger(sampleEngine.triggers.LESS_THAN,10)

            #Start and stop judged by the main channel
            engine.setTriggerChannel(sampleEngine.channels.MainCurrent)

            #Start sampling
            engine.startSampling(numSamples)

            engine.disableCSVOutput()
            engine.startSampling(5000)
            samples = engine.getSamples()

            #Samples are stored in order, indexed sampleEngine.channels values
            for i in range(len(samples[sampleEngine.channels.timeStamp])):
                timeStamp = samples[sampleEngine.channels.timeStamp][i]
                Current = samples[sampleEngine.channels.timeStamp][i]
                print("Main current at time " + repr(timeStamp) + " is: " + repr(Current) + "mA")
            
            

