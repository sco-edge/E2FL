from PMIC import PMICMonitor # power.
from INA3221 import INA3221 # power.

def get_power_monitor(power_mode, device_name=None):
    """
    Return the appropriate power monitor instance based on the power mode and device name.
    :param power_mode: The power monitoring mode (e.g., "PMIC", "INA3221").
    :param device_name: The device name to determine the appropriate monitor.
    :return: An instance of the appropriate power monitor, or None if not applicable.
    """
    if power_mode == "PMIC":
        return PMICMonitor()
    elif power_mode == "INA3221":
        return INA3221()
    elif device_name:
        # Add logic for additional devices if needed
        if "jetson" in device_name.lower():
            return INA3221()
        elif "raspberry" in device_name.lower():
            return PMICMonitor()
    return None
