"""Common interface for LoRACast power monitors."""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import threading


class PowerMonitor(ABC):
    """Abstract continuous-sampling power monitor.

    Concrete monitors in the registration build return instantaneous power in
    watts.  ``read_power_avg`` returns the arithmetic mean of sampled watts.
    """

    def __init__(self, device_name: str, logging_config: dict | None = None):
        self.device_name = device_name
        self.freq = 1.0
        self.power_data: list[tuple[float, float]] = []
        self.thread: threading.Thread | None = None
        self.monitoring = False
        self.start_time = None
        self.end_time = None
        self.lock = threading.Lock()

        if logging_config:
            logging.basicConfig(**logging_config)

    @abstractmethod
    def start(self, freq: float):
        """Start continuous sampling."""

    @abstractmethod
    def stop(self):
        """Stop continuous sampling."""

    @abstractmethod
    def read_power(self):
        """Return instantaneous power in watts, or ``None`` when unavailable."""

    @abstractmethod
    def read_power_avg(self):
        """Return mean sampled power in watts."""

    @abstractmethod
    def save(self, filepath):
        """Save samples to a file."""

    @abstractmethod
    def close(self):
        """Release monitor resources."""

    def handle_error(self, error_message: str):
        logging.error("%s: %s", self.device_name, error_message)
