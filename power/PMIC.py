# https://github.com/librerpi/rpi-tools/blob/master/pi5_voltage.py

import subprocess
import time
import threading
import re

import csv
import datetime
import logging
from power._power_monitor_interface import PowerMonitor
from datetime import datetime  # 추가: datetime.now()를 사용하기 위해 datetime 클래스를 가져옵니다.

class PMICMonitor(PowerMonitor):
    def __init__(self):
        super().__init__('PMIC')
        self.monitoring = False
        self.power_data = []
        self.lock = threading.Lock()
        self.thread = None
        self.start_time = 0

        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def read_power_avg(self):
        """
        Reads the average power consumption in mW using the 'vcgencmd pmic_read_adc' command.
        :return: Average power consumption in mW (float), or None if reading fails.
        """
        if not self.power_data:
            return 0
        else:
            data = self.power_data
            return float(sum(p for _, p in data) / len(data))
        return 0

    def read_power(self):
        """
        Implements the abstract method from PowerMonitor.
        Reads the power consumption using the '_read_power' method.
        :return: Power consumption in mW (float), or None if reading fails.
        """
        return self._read_power()

    def _read_power(self, timeout=5):
        """
        Executes the 'vcgencmd pmic_read_adc' command to retrieve the power consumption.
        :return: Power consumption in mW (float)

        $ sudo mknod /dev/vcio c 100 0
        $ sudo chmod 666 /dev/vcio

        """
        try:
            result = subprocess.run(['vcgencmd', 'pmic_read_adc'], stdout=subprocess.PIPE, text=True, check=True, timeout=timeout)
            lines = result.stdout.splitlines()
            
            volt_dict = {}
            curr_dict = {}

            for line in lines:
                match = re.search(r'([A-Z0-9_]+)_[VA] (current|volt)\((\d+)\)=([\d.]+)', line)
                if match:
                    name = match.group(1)
                    typ = match.group(2)  # current or volt
                    value = float(match.group(4))
                    if typ == "current":
                        curr_dict[name] = value
                    elif typ == "volt":
                        volt_dict[name] = value

            # 공통 key 기준으로 전력 계산
            power = 0
            for key in curr_dict:
                if key in volt_dict:
                    power += curr_dict[key] * volt_dict[key]

            logging.info(f"Power read: {power} mW")
            return float(power)
        except subprocess.CalledProcessError as e:
            logging.error(f"Command failed: {e}")
            return None
        except ValueError:
            logging.error("Invalid power value received.")
            return None
        except subprocess.TimeoutExpired:
            logging.error("Command timed out.")
            return None
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return None

    def _monitor(self):
        """
        Monitor the energy usage in a separate thread
        """
        logging.info(f"RPi5: Power monitoring started.")
        while self.monitoring:
            timestamp = (datetime.now() - self.start_time).total_seconds()
            power = self.read_power()
            if power is not None:
                with self.lock:
                    self.power_data.append((timestamp, power))
            time.sleep(self.freq)
        #logging.info(f"RPi5: Power monitoring stopped.")

    def get_clock(self, name):
        res = subprocess.run(["vcgencmd","measure_clock",name], capture_output=True)
        res = re.search('=([0-9]+)', res.stdout.decode("utf-8"))
        return res.group(1)

    def start(self, freq):
        """
        Start energy monitoring at the specified frequency
        :param freq: Frequency in seconds to sample energy data
        """
        with self.lock:
            if self.monitoring:
                logging.warning(f"RPi5: Monitoring is already running.")
                return

            self.freq = freq
            self.monitoring = True
            self.power_data = []
            self.start_time = datetime.now()

        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        logging.info(f"RPi5: Monitoring started with frequency {self.freq}s at {self.start_time}.")

    def stop(self):
        """
        Stop power monitoring and return elapsed time and data size.
        """
        with self.lock:
            if not self.monitoring:
                logging.warning(f"RPi5: Monitoring is not running.")
                return None, None

            self.monitoring = False

        if self.thread and self.thread.is_alive():
            self.thread.join()  # Ensure the thread is properly joined
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        data_size = len(self.power_data)
        logging.info(f"RPi5: Monitoring stopped. Duration: {elapsed_time:.2f}s, Data size: {data_size}.")
        return elapsed_time, data_size

    def save(self, filepath):
        # Save the power data to a CSV file using the csv module
        try:
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Write the global start time in the header
                writer.writerow([f"start_time", f"{self.start_time}"])
                writer.writerow(["timestamp", "power_mW"])
                # Write each (timestamp, power) pair into the file
                with self.lock:
                    for timestamp, power in self.power_data:
                        writer.writerow([f"{timestamp:.2f}", power])
            logging.info(f"RPi5: Data saved to {filepath}.")
        except Exception as e:
            logging.error(f"Failed to save data to {filepath}: {e}")

    def close(self):
        elapsed_time, data_size = None, None
        if self.monitoring:
            elapsed_time, data_size = self.stop()
        if elapsed_time == None:
            return
        logging.info(f"RPi5: Resources (data_size: {data_size}, elapsed_time: {elapsed_time}) cleaned up.")