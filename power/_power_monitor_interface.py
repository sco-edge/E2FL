from abc import ABC, abstractmethod
import logging
import threading

class PowerMonitor(ABC):
    def __init__(self):
        self.power_data = []  # List to store power data [(timestamp, power)]
        self.thread = None
        self.monitoring = False  # Flag to track if monitoring is ongoing
        self.lock = threading.Lock()
        self.thread = None

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    @abstractmethod
    def start(self, freq):
        """
        Start power monitoring at the given frequency.
        :param freq: Sampling interval in seconds
        """
        pass

    @abstractmethod
    def stop(self):
        """
        Stop power monitoring and return the elapsed time and size of the data collected.
        :return: (elapsed_time, data_size)
        """
        pass

    @abstractmethod
    def read_power(self):
        """
        Read the current power consumption.
        :return: float (power, in mW)
        """
        pass

    @abstractmethod
    def save(self, filepath):
        """
        Save the collected power data to the specified file.
        :param filepath: Path to the file where data will be saved
        """
        pass

    @abstractmethod
    def close(self):
        """
        Stop monitoring and clean up resources.
        """
        pass

    def handle_error(self, error_message):
        """
        Handle errors during power monitoring.
        :param error_message: Error message to log.
        """
        logging.error(f"RPi5: {error_message}")
        self.stop()
