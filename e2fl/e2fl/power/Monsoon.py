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
    def __init__(self, vout, ConsoleIO = False):
        super().__init__('Monsoon')
        self.vout = vout

        self.Mon = LVPM.Monsoon()
        self.Mon.setup_usb()
        if type(self.Mon.DEVICE) == type(None):
            logging.info("Mon.Device is NoneType")
            exit()

        if self.vout >= 4.56:
            self.Mon.setVout(4.55) # vout = 5.5V
        else:
            self.Mon.setVout(vout) # vout = 5.5V

        self.engine = sampleEngine.SampleEngine(self.Mon)
        self.engine.ConsoleOutput(ConsoleIO)
        #self.engine.startSampling(numSamples)

        # The main power regulator can source 3.0 A of continuous current and 4.5 A of peak current
        # Power up with no current limit for 20 ms, run continuously with the current limit set to 4.6 A
        # If you require a higher measurement voltage, the AUX channel can support up to 5.5V.
        # If you require larger sustaned currents, the AUX channel can support up to 4.5 Amps continuous current
        # If it is necessary to vary voltage continuously, without ending the sampling run.
        if self.vout >= 4.56 and self.vout <= 5.5:
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
            self.Mon.setUSBPassthroughMode(op.USB_Passthrough.On) # == 1
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
            self.Mon.setUSBPassthroughMode(2) # op.USB_Passthrough.Auto? == 2
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
            self.numSamples = sampleEngine.triggers.SAMPLECOUNT_INFINITE

            #Start when we exceed threshold_high (100 mA)
            self.engine.setStartTrigger(sampleEngine.triggers.GREATER_THAN, thld_high)

            #Stop when we drop below threshold_low (10 mA)
            self.engine.setStopTrigger(sampleEngine.triggers.LESS_THAN, thld_low)
                    
            #Start and stop judged by the channel
            if self.vout >= 4.6 and self.vout <= 5.5: # AUX channel
                self.engine.setTriggerChannel(sampleEngine.channels.AuxCurrent)
            else:  # main channel
                self.engine.setTriggerChannel(sampleEngine.channels.MainCurrent)
            #self.engine.startSampling(self.numSamples)
            return True
        else: # turn off the trigger mode and start the sampling mode
            return False
            #self.engine.startSampling(self.numSamples)

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

    def start(self):
        """
        Start energy monitoring in a separate thread.
        """
        if self.monitoring:
            logging.warning("Sampling is already running.")
            return

        # Configure the sampling parameters
        self.monitoring = True
        self.sampling_thread = threading.Thread(target=self._run_sampling, daemon=True)
        self.sampling_thread.start()
        logging.info("Monsoon sampling started.")

    def _run_sampling(self):
        """
        Run sampling loop in a background thread.
        """
        try:
            with self.lock:
                # Start sampling using __startSampling with provided parameters
                self.engine.startSampling(
                    samples=sampleEngine.triggers.SAMPLECOUNT_INFINITE,
                    legacy_timestamp=True
                )
                
                # Loop to continuously collect samples while sampling is active
                while self.monitoring:
                    # The actual sample processing happens in the Monsoon library's __sampleLoop
                    sample_count = self.monsoon.__sampleLoop(0, [], self.granularity, self.legacy_timestamp)
                    
                    # Delay to allow sampling at the specified interval
                    time.sleep(1 / self.granularity)

        except Exception as e:
            logging.error(f"Error in sampling: {e}")

        finally:
            # Clean up and stop sampling
            self.monsoon.stopSampling()
            logging.info("Monsoon sampling stopped.")

    def stop(self):
        """
        Stop energy monitoring gracefully.
        """
        if not self.monitoring:
            logging.info("Energy monitoring is not running.")
            return None, None

        with self.lock:
            if not self.monitoring:
                logging.warning("Sampling is not running.")
                return

            # Signal the sampling loop to stop and wait for the thread to complete
            self.monitoring = False
            self.sampling_thread.join()
            logging.info("Monsoon sampling has been stopped.")
            self.end_time = datetime.now() # time.strftime("%Y/%m/%d %H:%M:%S")
            elapsed_time = self.end_time - self.start_time
            self.power_data = self._getSamples()
            data_size = len(self.power_data)
            logging.debug(f"{self.device_name}: Monitoring stopped. Time: {elapsed_time}s, Data size: {data_size}.")

        return elapsed_time, data_size
        
    def save(self, filepath):
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([f"start_time", f"{self.start_time}"])
            writer.writerow(["timestamp", "power_mW"])
            with self.lock:
                for entry in self.power_data:
                    timestamp, main, usb, aux, mainVolts, usbVolts = entry
                
                # Calculate main and USB power in mW
                main_power = main * mainVolts
                usb_power = usb * usbVolts
                
                # Write the row with calculated powers
                #writer.writerow([
                #    experiment_details, f"{timestamp:.2f}", f"{main_power:.2f}", f"{usb_power:.2f}", 
                #    f"{aux:.2f}", f"{mainVolts:.2f}", f"{usbVolts:.2f}"
                #])
                with self.lock:
                    for ind, power in enumerate(usb_power):
                        writer.writerow([f"{(timestamp[ind]):.2f}", power])
        logging.info(f"{self.device_name}: Data saved to {filepath}.")

    def close(self):
        elapsed_time, data_size = None, None
        if self.monitoring:
            elapsed_time, data_size = self.stop()
        if elapsed_time == None:
            return
        logging.info(f"{self.device_name}: Resources (data_size: {data_size}, elapsed_time: {elapsed_time}) cleaned up.")
        
    