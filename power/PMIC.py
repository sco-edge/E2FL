# https://github.com/librerpi/rpi-tools/blob/master/pi5_voltage.py

import subprocess
import time
import threading

import csv
import datetime
import logging
from power._power_monitor_interface import PowerMonitor
from datetime import datetime  # 추가: datetime.now()를 사용하기 위해 datetime 클래스를 가져옵니다.

class PMICMonitor(PowerMonitor):
    def __init__(self):
        super().__init__('PMIC')    

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
            power = result.stdout.strip()
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
        logging.info(f"{self.device_name}: Power monitoring started.")
        while self.monitoring:
            power = self._read_power()
            current_time = datetime.now() - self.start_time  # 수정: datetime.now() 사용
            if power is not None:
                with self.lock:
                    self.power_data.append((current_time, float(power)))  # (timestamp, power) 형태로 저장
            time.sleep(self.freq)
        logging.info(f"{self.device_name}: Power monitoring stopped.")

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
        self.power_data = []
        self.start_time = datetime.now()  # 수정: datetime.now() 사용
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()
        logging.debug(f"{self.device_name}: Monitoring started with frequency {self.freq}s at {self.start_time} (UTC).")

    def stop(self):
        """
        Stop power monitoring and return elapsed time and data size.
        """
        with self.lock:
            if not self.monitoring:
                logging.warning(f"{self.device_name}: Monitoring is not running.")
                return None, None

            self.monitoring = False

        if self.thread and self.thread.is_alive():
            self.thread.join()  # Ensure the thread is properly joined
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        data_size = len(self.power_data)
        logging.info(f"{self.device_name}: Monitoring stopped. Duration: {elapsed_time:.2f}s, Data size: {data_size}.")
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
            logging.info(f"{self.device_name}: Data saved to {filepath}.")
        except Exception as e:
            logging.error(f"Failed to save data to {filepath}: {e}")

    def close(self):
        elapsed_time, data_size = None, None
        if self.monitoring:
            elapsed_time, data_size = self.stop()
        if elapsed_time == None:
            return
        logging.info(f"{self.device_name}: Resources (data_size: {data_size}, elapsed_time: {elapsed_time}) cleaned up.")