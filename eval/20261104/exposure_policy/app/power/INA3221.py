"""Jetson INA3221 power monitor for LoRACast.

This implementation reads Linux hwmon sysfs values directly.  It is written as
an integration layer around the kernel-provided INA3221 interface and does not
embed third-party monitor source code.
"""

from __future__ import annotations

import csv
from datetime import datetime
import logging
from pathlib import Path
import threading
import time

from power._power_monitor_interface import PowerMonitor


class INA3221(PowerMonitor):
    def __init__(
        self,
        voltage_path: str | None = None,
        current_path: str | None = None,
    ):
        super().__init__("INA3221")
        discovered_voltage, discovered_current = self._discover_input_rail()

        self.voltage_path = Path(
            voltage_path or discovered_voltage
        )
        self.current_path = Path(
            current_path or discovered_current
        )

    @staticmethod
    def _discover_input_rail() -> tuple[str, str]:
        base = Path("/sys/bus/i2c/drivers/ina3221")
        voltage_candidates = sorted(base.glob("*/hwmon/hwmon*/in1_input"))
        current_candidates = sorted(base.glob("*/hwmon/hwmon*/curr1_input"))

        if not voltage_candidates or not current_candidates:
            # Retain the historical Jetson path as a useful fallback while
            # allowing callers to override either path explicitly.
            root = "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1"
            return f"{root}/in1_input", f"{root}/curr1_input"

        voltage = voltage_candidates[0]
        sibling_current = voltage.with_name("curr1_input")
        if sibling_current.exists():
            return str(voltage), str(sibling_current)

        return str(voltage), str(current_candidates[0])

    @staticmethod
    def _read_number(path: Path) -> float | None:
        try:
            return float(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError) as exc:
            logging.error("INA3221: failed to read %s: %s", path, exc)
            return None

    def read_power(self) -> float | None:
        """Return module input power in watts."""
        millivolts = self._read_number(self.voltage_path)
        milliamps = self._read_number(self.current_path)
        if millivolts is None or milliamps is None:
            return None

        volts = millivolts / 1000.0
        amps = milliamps / 1000.0
        return float(volts * amps)

    def read_power_avg(self) -> float:
        with self.lock:
            samples = list(self.power_data)
        if not samples:
            return 0.0
        return float(sum(power for _, power in samples) / len(samples))

    def _monitor(self):
        while self.monitoring:
            started = self.start_time
            if started is None:
                break

            elapsed = (datetime.now() - started).total_seconds()
            power = self.read_power()
            if power is not None:
                with self.lock:
                    self.power_data.append((elapsed, power))

            time.sleep(max(float(self.freq), 0.001))

    def start(self, freq: float):
        with self.lock:
            if self.monitoring:
                return

            self.freq = float(freq)
            self.power_data = []
            self.start_time = datetime.now()
            self.monitoring = True

        self.thread = threading.Thread(
            target=self._monitor,
            name="loracast-ina3221",
            daemon=True,
        )
        self.thread.start()

    def stop(self):
        with self.lock:
            if not self.monitoring:
                return None, None
            self.monitoring = False
            started = self.start_time

        thread = self.thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(2.0, 2.0 * float(self.freq)))

        self.end_time = datetime.now()
        elapsed = (
            (self.end_time - started).total_seconds()
            if started is not None
            else 0.0
        )
        with self.lock:
            size = len(self.power_data)
        return elapsed, size

    def save(self, filepath):
        with self.lock:
            rows = list(self.power_data)

        with open(filepath, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp_s", "power_w"])
            writer.writerows(rows)

    def close(self):
        if self.monitoring:
            self.stop()
