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
import csv
from datetime import datetime
import logging
from power._power_monitor_interface import PowerMonitor

class INA3221(PowerMonitor):
    def __init__(self, voltage_path='/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon/in1_input',
                 current_path='/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon/curr1_input'):
        """
        Initialize the INA3221 power monitor.
        :param voltage_path: Path to read voltage in mV.
        :param current_path: Path to read current in mA.
        """
        super().__init__('INA3221')
        self.voltage_path = voltage_path
        self.current_path = current_path
        self.monitoring = False
        self.power_data = []
        self.lock = threading.Lock()
        self.thread = None

        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def _read_sysfs(self, path):
        """
        Helper function to read a value from a sysfs path.
        :param path: sysfs file path to read from.
        :return: Value as float, or None if an error occurs.
        """
        with self.lock:
            try:
                with open(path, 'r') as f:
                    value = f.read().strip()
                return float(value)
            except Exception as e:
                logging.error(f"{self.device_name}: Error reading {path}: {e}")
                return None

    def read_power(self):
        """
        Reads power consumption in mW by reading voltage (mV) and current (mA) from sysfs.
        :return: Power consumption in mW (float), or None if reading fails.
        """
        voltage = self._read_sysfs(self.voltage_path)  # in mV
        current = self._read_sysfs(self.current_path)  # in mA

        if voltage is None or current is None:
            return None  # Return None if any reading fails

        # Convert mV * mA to mW
        power_mw = (voltage / 1000.0) * (current / 1000.0) * 1000  # Convert to mW
        logging.debug(f"{self.device_name}: Voltage={voltage}mV, Current={current}mA, Power={power_mw:.2f}mW")
        return power_mw

    def _monitor(self):
        """
        Background thread function that records timestamped power data.
        """
        logging.info(f"{self.device_name}: Power monitoring started.")
        while self.monitoring:
            timestamp = (datetime.now() - self.start_time).total_seconds()
            power = self.read_power()
            if power is not None:
                with self.lock:
                    self.power_data.append((timestamp, power))
            time.sleep(self.freq)

    def start(self, freq):
        """
        Start power monitoring in a separate background thread.
        :param freq: Sampling frequency in seconds.
        """
        with self.lock:
            if self.monitoring:
                logging.warning(f"{self.device_name}: Monitoring is already running.")
                return

            self.freq = freq
            self.monitoring = True
            self.power_data = []
            self.start_time = datetime.now()

        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        logging.info(f"{self.device_name}: Monitoring started with frequency {self.freq}s at {self.start_time}.")

    def stop(self):
        """
        Stop power monitoring and return elapsed time and data size.
        :return: Elapsed time (seconds), data size (number of power readings).
        """
        with self.lock:
            if not self.monitoring:
                logging.warning(f"{self.device_name}: Monitoring is not running.")
                return None, None

            self.monitoring = False

        self.thread.join()
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        data_size = len(self.power_data)
        logging.info(f"{self.device_name}: Monitoring stopped. Duration: {elapsed_time:.2f}s, Data size: {data_size} samples.")
        return elapsed_time, data_size

    def save(self, filepath):
        """
        Save collected power data to a CSV file.
        :param filepath: File path for saving data.
        """
        with self.lock:
            if not self.power_data:
                logging.warning(f"{self.device_name}: No power data to save.")
                return

            try:
                with open(filepath, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["Timestamp (s)", "Power (mW)"])
                    for timestamp, power in self.power_data:
                        writer.writerow([f"{timestamp:.2f}", f"{power:.2f}"])
                logging.info(f"{self.device_name}: Data saved to {filepath}.")
            except Exception as e:
                logging.error(f"{self.device_name}: Failed to save power data: {e}")

    def close(self):
        """
        Stop monitoring and clean up resources.
        """
        elapsed_time, data_size = None, None
        if self.monitoring:
            elapsed_time, data_size = self.stop()

        if elapsed_time is None:
            logging.info(f"{self.device_name}: No active monitoring to clean up.")
            return

        logging.info(f"{self.device_name}: Resources cleaned up (Elapsed Time: {elapsed_time:.2f}s, Data Size: {data_size} samples).")
