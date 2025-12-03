from abc import ABC, abstractmethod
import logging
import threading

class PowerMonitor(ABC):
    def __init__(self, device_name, logging_config=None):
        self.device_name = device_name  # Name of the device
        self.freq = 1  # Sampling interval in seconds
        self.power_data = []  # List to store power data [(timestamp, power)]
        self.thread = None
        self.monitoring = False  # Flag to track if monitoring is ongoing
        self.start_time = None  # Time when monitoring started (for relative time calculations)
        self.end_time = None  # Time when monitoring endded (for relative time calculations)
        self.lock = threading.Lock()

        if logging_config:
            logging.basicConfig(**logging_config)
        else:
            # Default logging configuration (prints to console, level set to INFO)
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
    def read_power_avg(self):
        """
        Reads the average power consumption in mW using the 'vcgencmd pmic_read_adc' command.
        :return: Average power consumption in mW (float), or None if reading fails.
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
        logging.error(f"{self.device_name}: {error_message}")
        self.stop()
