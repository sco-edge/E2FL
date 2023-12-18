import monsoon.LVPM as LVPM
import monsoon.sampleEngine as sampleEngine
import monsoon.Operations as op
import datetime

'''
pip install monsoon
'''
class PowerMon():
    
    def __init__(self, node, vout, channel, mode = "PyMonsoon"):
        self.mode = mode
        self.node = node
        self.vout = vout
        self.channel = channel
        if mode == "PyMonsoon":
            Mon = LVPM.Monsoon()
            Mon.setup_usb()

            Mon.setVout(vout) # vout = 5.5V
            self.engine = sampleEngine.SampleEngine(Mon)
            self.engine.ConsoleOutput(True)
            #self.engine.startSampling(numSamples)

            # The main power regulator can source 3.0 A of continuous current and 4.5 A of peak current
            # Power up with no current limit for 20 ms, run continuously with the current limit set to 4.6 A
            # If you require a higher measurement voltage, the AUX channel can support up to 5.5V.
            # If you require larger sustaned currents, the AUX channel can support up to 4.5 Amps continuous current
            # If it is necessary to vary voltage continuously, without ending the sampling run.
            if vout >= 4.6 and vout <= 5.5:
                #Disable Main channels
                self.engine.disableChannel(sampleEngine.channels.MainCurrent)
                self.engine.disableChannel(sampleEngine.channels.MainVoltage)

                #Enable USB channels
                self.engine.enableChannel(sampleEngine.channels.USBCurrent)
                self.engine.enableChannel(sampleEngine.channels.USBVoltage)

                #Enable AUX channels
                self.engine.enableChannel(sampleEngine.channels.AuxCurrent)

                #Set USB Pasthrough mode to 'on', since it defaults to 'auto' 
                #and will turn off when sampling mode begins
                Mon.setUSBPassthroughMode(op.USB_Passthrough.On) # == 1
            elif vout < 4.6:
                #Disable USB channels
                self.engine.disableChannel(sampleEngine.channels.USBCurrent)
                self.engine.disableChannel(sampleEngine.channels.USBVoltage)

                #Disable AUX channels
                self.engine.disableChannel(sampleEngine.channels.AuxCurrent)

                #Enable Main channels
                self.engine.enableChannel(sampleEngine.channels.MainCurrent)
                self.engine.enableChannel(sampleEngine.channels.Mainoltage)
                
                #Set USB Pasthrough mode to 'auto' as default
                Mon.setUSBPassthroughMode(2) # op.USB_Passthrough.Auto? == 2
            else:
                raise Exception("The required voltage is not supported on Monsoon Power Monitor.")
            
            '''
            # self.engine.disableCSVOutput()
            samples = self.engine.getSamples()

            #Samples are stored in order, indexed sampleEngine.channels values
            for i in range(len(samples[sampleEngine.channels.timeStamp])):
                timeStamp = samples[sampleEngine.channels.timeStamp][i]
                Current = samples[sampleEngine.channels.timeStamp][i]
                print("Main current at time " + repr(timeStamp) + " is: " + repr(Current) + "mA")
            '''

    def setTrigger(self, bool, numSamples = 5000):
        if bool: # trigger mode
            #Don't stop based on sample count, continue until 
            #the trigger conditions have been satisfied.
            numSamples = sampleEngine.triggers.SAMPLECOUNT_INFINITE

            #Start when we exceed 100 mA
            self.engine.setStartTrigger(sampleEngine.triggers.GREATER_THAN,100)

            #Stop when we drop below 10 mA
            self.engine.setStopTrigger(sampleEngine.triggers.LESS_THAN,10)
                    
            #Start and stop judged by the channel
            if self.vout >= 4.6 and self.vout <= 5.5: # AUX channel
                self.engine.setTriggerChannel(sampleEngine.channels.AUXCurrent)
            else:  # main channel
                self.engine.setTriggerChannel(sampleEngine.channels.MainCurrent)
            self.engine.startSampling(numSamples)
        else: # turn off the trigger mode and start the sampling mode
            self.engine.startSampling(numSamples)

    def setCSVOutput(self, bool):
        if bool:
            init_run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.engine.enableCSVOutput("E2FL_"+init_run_timestamp+".csv")
        else:
            self.engine.disableCSVOutput()

    def getSamples(self):
        return self.engine.getSamples()

    def startSampling(self, numSamples = 5000):
        # numSamples = sample for one second
        self.engine.startSampling(numSamples)

    def stopSampling(self):
        self.engine.stopSampling()

    #def calibration(self)
    


            

