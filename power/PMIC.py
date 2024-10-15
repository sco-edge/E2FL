# https://github.com/librerpi/rpi-tools/blob/master/pi5_voltage.py

import subprocess
import time
import threading

import csv
import datetime
import logging
from _power_monitor_interface import PowerMonitor

class PMICMonitor(PowerMonitor):
    def __init__(self, sysfs_path='/sys/class/power_supply/pmic_device/power_now', logging_config=None):
        super().__init__('PMIC')
        self.sysfs_path = sysfs_path
        self.lock = threading.Lock()
        self.global_start_time = None  # 글로벌 시작 시간을 기록할 변수

        # Set up logging based on the passed configuration
        if logging_config:
            logging.basicConfig(**logging_config)

    def start(self, freq):
        self.sampling_interval = freq
        self.is_monitoring = True
        self.power_data = []
        self.start_time = time.time()
        self.global_start_time = datetime.datetime.utcnow()
        print(f"{self.device_name}: Monitoring started with frequency {self.sampling_interval}s at {self.global_start_time} (UTC).")

    def stop(self):
        self.is_monitoring = False
        elapsed_time = len(self.power_data) * self.sampling_interval
        data_size = len(self.power_data)
        print(f"{self.device_name}: Monitoring stopped. Time: {elapsed_time}s, Data size: {data_size}.")
        return elapsed_time, data_size

    def read_power(self):
        """
        Executes the 'vgencmd pmmic_read_adc' command to retrieve the power consumption.
        :return: Power consumption in mW (float)
        """
        try:
            #result = subprocess.run(['vgencmd', 'pmmic_read_adc'], stdout=subprocess.PIPE, text=True, check=True)
            result = subprocess.run(['vgencmd', 'pmmic_read_adc'], stdout=subprocess.PIPE, text=True, check=True, timeout=5)
            power_str = result.stdout.strip()
            power = float(power_str)
            current_time = time.time() - self.start_time
            with self.lock:
                self.power_data.append((current_time, power))
            logging.debug(f"{self.device_name}: Current power: {power} mW at {current_time:.2f}s")
            return power
        except subprocess.CalledProcessError as e:
            logging.error(f"{self.device_name} Command failed: {e}")
            return None
        except ValueError:
            logging.error(f"{self.device_name} Invalid power value received.")
            return None
        except subprocess.TimeoutExpired:
            print(f"{self.device_name} Command timed out.")
            return None
        except Exception as e:
            logging.error(f"{self.device_name} Unexpected error: {e}")
            return None

    def save(self, filepath):
        # Save the power data to a CSV file using the csv module
        try:
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Write the global start time in the header
                writer.writerow([f"global_start_time", f"{self.global_start_time} (UTC)"])
                writer.writerow(["timestamp", "power_mW"])
                # Write each (timestamp, power) pair into the file
                with self.lock:
                        for timestamp, power in self.power_data:
                            writer.writerow([f"{timestamp:.2f}", power])
            logging.info(f"{self.device_name}: Data saved to {filepath}.")
        except Exception as e:
            logging.error(f"Failed to save data to {filepath}: {e}")

    def close(self):
        if self.is_monitoring:
            self.stop()
        logging.info(f"{self.device_name}: Resources cleaned up.")