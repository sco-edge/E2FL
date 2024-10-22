import Monsoon.LVPM as LVPM
import Monsoon.sampleEngine as sampleEngine
import Monsoon.Operations as op
import datetime
import subprocess
import re
import time
import logging
import threading
import csv
from _power_monitor_interface import PowerMonitor

'''
pip install monsoon
It is broken for LVPM. Use HVPM or uses manual mode.
'''
class MonsoonMonitor(PowerMonitor):
    def __init__(self, node, vout, ConsoleIO = False):
        super().__init__('Monsoon')
        self.node = node
        self.vout = vout

        Mon = LVPM.Monsoon()
        Mon.setup_usb()
        if type(Mon.DEVICE) == type(None):
            logging.info("Mon.Device is NoneType")
            exit()

        if self.vout >= 4.56:
            Mon.setVout(4.55) # vout = 5.5V
        else:
            Mon.setVout(vout) # vout = 5.5V

        self.engine = sampleEngine.SampleEngine(Mon)
        self.engine.ConsoleOutput(ConsoleIO)
        #self.engine.startSampling(numSamples)

        # The main power regulator can source 3.0 A of continuous current and 4.5 A of peak current
        # Power up with no current limit for 20 ms, run continuously with the current limit set to 4.6 A
        # If you require a higher measurement voltage, the AUX channel can support up to 5.5V.
        # If you require larger sustaned currents, the AUX channel can support up to 4.5 Amps continuous current
        # If it is necessary to vary voltage continuously, without ending the sampling run.
        if vout >= 4.56 and vout <= 5.5:
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
        elif vout < 4.56:
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
            logging.error("The required voltage is not supported on Monsoon Power Monitor.")
            return None
    def _setTrigger(self, bool, numSamples = 5000, thld_high = 100, thld_low = 10):
        '''
            Set the threshold for trigger that starts sampleEngine's recording measurements.
            sampleEngine begins recording measurements when the 'start' trigger condition is met,
                      and stops sampling completely when the 'stop' trigger condition is met.
            
                * numSamples: sample for 1 second
        '''
        if bool: # trigger mode
            #Don't stop based on sample count, continue until 
            #the trigger conditions have been satisfied.
            numSamples = sampleEngine.triggers.SAMPLECOUNT_INFINITE

            #Start when we exceed threshold_high (100 mA)
            self.engine.setStartTrigger(sampleEngine.triggers.GREATER_THAN, thld_high)

            #Stop when we drop below threshold_low (10 mA)
            self.engine.setStopTrigger(sampleEngine.triggers.LESS_THAN, thld_low)
                    
            #Start and stop judged by the channel
            if self.vout >= 4.6 and self.vout <= 5.5: # AUX channel
                self.engine.setTriggerChannel(sampleEngine.channels.AuxCurrent)
            else:  # main channel
                self.engine.setTriggerChannel(sampleEngine.channels.MainCurrent)
            #self.engine.startSampling(numSamples)
            return True
        else: # turn off the trigger mode and start the sampling mode
            return False
            #self.engine.startSampling(numSamples)

    def _setCSVOutput(self, bool, filename="default"):
        '''
            Opens a file and causes the sampleEngine to periodically output samples when taking measurements.
            
            The output CSV file will consist of one row of headers, followed by measurements.
            If every output channel is enabled, it will have the format:
                    Time,    Main Current,   USB Current,   Aux Current,    Main Voltage,   USB Voltage,
                timestamp 1, main current 1,    usb 1,          aux 1,    main voltage 1,       usb 1
                timestamp 2, main current 2,    usb 2,          aux 2,    main voltage 2,       usb 2
        '''
        if bool:
            if filename == "default":
                init_run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = "E2FL_"+init_run_timestamp+".csv"
            self.engine.enableCSVOutput("E2FL_"+init_run_timestamp+".csv")
        else:
            self.engine.disableCSVOutput()

    
    def _getSamples(self):
        '''
            Returns samples in a Python list.
            
            Format is [timestamp, main, usb, aux, mainVolts, usbVolts]
            Channels that were excluded with the disableChannel() function will have an empty list in their array index.
        '''
        return self.engine.getSamples()

    def _startSampling(self, numSamples = 5000):
        '''
        '''
        # numSamples = sample for one second
        self.engine.startSampling(numSamples)

    def _stopSampling(self, closeCSV=False):
        '''
        closeCSV: closes the CSV file along with existile mode.
        '''
        self.engine.periodicStopSampling(closeCSV)

    def start(self, freq):
        if self.monitoring:
            logging.info("Energy monitoring is already running.")
            return
        
        self.freq = freq
        self.monitoring = True
        self.power_data = []
        self.start_time = datetime.now() #time.strftime("%Y/%m/%d %H:%M:%S")
        #self.thread = threading.Thread(target=self._monitor)
        #self.thread.start()
        self._startSampling(self.freq)
        logging.debug(f"{self.device_name}: Monitoring started with frequency {self.freq}s at {self.start_time}.")

    def stop(self):
        if not self.monitoring:
            logging.info("Energy monitoring is not running.")
            return None, None

        self.monitoring = False
        #self.thread.join() # Wait for thread to finish
        self._stopSampling(closeCSV = True)
        self.end_time = datetime.now() # time.strftime("%Y/%m/%d %H:%M:%S")
        elapsed_time = self.end_time - self.start_time
        data_size = len(self.power_data)
        logging.debug(f"{self.device_name}: Monitoring stopped. Time: {elapsed_time}s, Data size: {data_size}.")
        return elapsed_time, data_size


    def save(self, filepath):
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([f"start_time", f"{self.start_time}"])
            writer.writerow(["timestamp", "power_mW"])
            with self.lock:
                for timestamp, power in self.power_data:
                    writer.writerow([f"{timestamp:.2f}", power])
        logging.info(f"{self.device_name}: Data saved to {filepath}.")

    def close(self):
        elapsed_time, data_size = None, None
        if self.monitoring:
            elapsed_time, data_size = self.stop()
        if elapsed_time == None:
            return
        logging.info(f"{self.device_name}: Resources (data_size: {data_size}, elapsed_time: {elapsed_time}) cleaned up.")