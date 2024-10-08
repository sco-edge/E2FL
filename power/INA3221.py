# https://github.com/tgraf/bmon; sysfs
# https://github.com/zzzDavid/xavier-power-profiling

'''
https://forums.developer.nvidia.com/t/jetson-orin-nx-power-management-parameters/280460
https://docs.nvidia.com/jetson/archives/r35.4.1/DeveloperGuide/text/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html?#jetson-orin-nx-series-and-jetson-orin-nano-series
INA3221 (on jetson orin nx)

To read INA3221 at 0x40, the channel-1 rail name, enter the command:
$ cat /sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon/in1_label
=> VDD_IN

To read channel-1 voltage and current, enter the commands:
$ cat /sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon/in1_input
=> 5072
$ cat /sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon/curr1_input
=> 1312 or 1320

To read the channel-1 instantaneous current limit, enter the command:
$ cat /sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon/curr1_crit
=> 5920

To set the channel-1 instantaneous current limit, enter the command:
$ echo  > /sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon/curr1_crit


To read the channel-1 average current limit, enter the command:
$ cat /sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon/curr1_max
=> 4936

To set the channel-1 average current limit, enter the command:
$ echo  > /sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon/curr1_max


Where is the current limit to be set for the rail, in milliamperes.

--
There are 3 types of OC events in the Orin series, 
which are Under Voltage, Average Overcurrent, and Instantaneous Overcurrent events respectively.

To check which OC event is enabled, the following sysfs nodes can be used:

$ grep "" /sys/class/hwmon/hwmon/oc*_throt_en
The following sysfs nodes can be used to learn the number of OC events occurred:

$ grep "" /sys/class/hwmon/hwmon/oc*_event_cnt

in1_label: VDD IN; The total power of module
in2_label: VDD CPU GPU CV; CV includes DLA (Deep Learning Accelerator), PVA (Programmable Vision Accelerator), and etc. (CV hardware)
in3_label: VDD SOC
in4_label: Sum of shunt voltages

CPU, GPU, CV? https://developer.nvidia.com/blog/maximizing-deep-learning-performance-on-nvidia-jetson-orin-with-dla/

'''
import threading
import time
import os
import pickle

import csv
import time
from datetime import datetime, timedelta
from _power_monitor_interface import PowerMonitor

class INA3221(PowerMonitor):
    def __init__(self, sysfs_path='/sys/bus/i2c/drivers/ina3221/1-0040/iio_device/in_power0_input'):
        """
        Initialize the EnergyMonitor class
        :param sysfs_path: Path to the sysfs file for energy data (default path for Jetson devices)
        """
        super().__init__('INA3221')
        self.sysfs_path = sysfs_path
        self.monitoring = False
        self.thread = None
        self.start_time = None
        self.end_time = None
        self.power_data = []
        self.freq = 1 # Frequency in seconds (default)
    
    def _read_sysfs(self):
        """
        Read the power consumption value from sysfs
        """
        try:
            with open(self.sysfs_path, 'r') as f:
                power = f.read().strip() # v
            return float(power)  # Assuming power value is in float format (e.g., mW)
        except Exception as e:
            print(f"Error reading sysfs: {e}")
            return None
    
        try:
            with open(self.sysfs_path, 'r') as f:
                power = f.read().strip()
            current_time = datetime.now() - self.start_time  # 측정된 시간 계산 (상대적 시간)
            self.power_data.append((current_time, float(power)))  # (timestamp, power) 형태로 저장
            return float(power)
        except Exception as e:
            print(f"{self.device_name} Error reading power: {e}")
            return None

    def _monitor(self):
        """
        Monitor the energy usage in a separate thread
        """
        while self.monitoring:
            power = self._read_sysfs()
            current_time = datetime.now() - self.start_time
            if power is not None:
                self.energy_data.append(power)
            time.sleep(self.freq)

    def start(self, freq):
        """
        Start energy monitoring at the specified frequency
        :param freq: Frequency in seconds to sample energy data
        """
        if self.monitoring:
            print("Energy monitoring is already running.")
            return

        self.freq = freq
        self.monitoring = True
        self.power_data = []  # Reset energy data on start
        self.start_time = datetime.now() #time.strftime("%Y/%m/%d %H:%M:%S")
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()
        print(f"{self.device_name}: Monitoring started with frequency {self.freq} at {self.start_time} (UTC).")

    def stop(self):
        """
        Stop energy monitoring and return the elapsed time and the amount of data collected
        :return: Elapsed time (seconds), data size (number of power readings)
        """
        if not self.monitoring:
            print("Energy monitoring is not running.")
            return None, None
        
        self.monitoring = False
        self.thread.join() # Wait for thread to finish
        self.end_time = datetime.now() # time.strftime("%Y/%m/%d %H:%M:%S")
        elapsed_time = self.end_time - self.start_time
        data_size = len(self.power_data)
        print(f"{self.device_name}: Monitoring stopped. Time: {elapsed_time}s, Data size: {data_size}.")
        return elapsed_time, data_size

    def save(self, filepath):
        # CSV 모듈을 사용하여 데이터를 저장
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # CSV 헤더: 글로벌 시작 시간 기록
            writer.writerow([f"start_time", f"{self.start_time}"])
            writer.writerow(["timestamp", "power_mW"])
            # 측정된 데이터 저장
            for timestamp, power in self.power_data:
                writer.writerow([timestamp, power])
        print(f"{self.device_name}: Data saved to {filepath}.")

    def close(self):
        if self.is_monitoring:
            self.stop()
        print(f"{self.device_name}: Resources cleaned up.")



class EnergyMonitor:
    def __init__(self, sysfs_path='/sys/bus/i2c/drivers/ina3221/1-0040/iio_device/in_power0_input'):
        """
        Initialize the EnergyMonitor class
        :param sysfs_path: Path to the sysfs file for energy data (default path for Jetson devices)
        """
        self.sysfs_path = sysfs_path
        self.freq = 1  # Frequency in seconds (default)
        self.monitoring = False
        self.thread = None
        self.energy_data = []  # List to store energy data (power readings)
        self.start_time = None
        self.end_time = None

    def _read_sysfs(self):
        """
        Read the power consumption value from sysfs
        """
        try:
            with open(self.sysfs_path, 'r') as f:
                power = f.read().strip()
            return float(power)  # Assuming power value is in float format (e.g., mW)
        except Exception as e:
            print(f"Error reading sysfs: {e}")
            return None

    def _monitor(self):
        """
        Monitor the energy usage in a separate thread
        """
        while self.monitoring:
            power = self._read_sysfs()
            if power is not None:
                self.energy_data.append(power)
            time.sleep(self.freq)

    def start(self, freq=1):
        """
        Start energy monitoring at the specified frequency
        :param freq: Frequency in seconds to sample energy data
        """
        if self.monitoring:
            print("Energy monitoring is already running.")
            return

        self.freq = freq
        self.monitoring = True
        self.energy_data = []  # Reset energy data on start
        self.start_time = time.time()  # Capture start time
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()
        print(f"Energy monitoring started with a frequency of {self.freq} second(s).")

    def stop(self):
        """
        Stop energy monitoring and return the elapsed time and the amount of data collected
        :return: Elapsed time (seconds), data size (number of power readings)
        """
        if not self.monitoring:
            print("Energy monitoring is not running.")
            return None, None

        self.monitoring = False
        self.thread.join()  # Wait for thread to finish
        self.end_time = time.time()  # Capture end time
        elapsed_time = self.end_time - self.start_time
        data_size = len(self.energy_data)
        print(f"Energy monitoring stopped. Time elapsed: {elapsed_time:.2f}s, Data size: {data_size}")
        return elapsed_time, data_size

    def save(self, fpath):
        """
        Save the energy data to a file using pickle
        :param fpath: Path to save the data (file extension can be .pkl or any other format)
        """
        if not self.energy_data:
            print("No energy data to save.")
            return

        with open(fpath, 'wb') as f:
            pickle.dump(self.energy_data, f)
        print(f"Energy data saved to {fpath}.")

    def close(self):
        """
        Gracefully terminate the thread and cleanup resources
        """
        if self.monitoring:
            self.stop()

        print("EnergyMonitor closed.")

# Example usage in any Python code
if __name__ == "__main__":
    monitor = EnergyMonitor()

    # Start monitoring with a frequency of 2 seconds
    monitor.start(freq=2)

    # Simulate a task
    time.sleep(10)

    # Stop monitoring
    elapsed, data_size = monitor.stop()

    # Save the collected energy data to a file
    monitor.save('energy_data.pkl')

    # Close the monitor and clean up
    monitor.close()
