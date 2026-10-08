from power.PMIC import PMICMonitor # power.
from power.INA3221 import INA3221 # power.

def get_power_monitor(power_mode, device_name=None):
    """
    Return the appropriate power monitor instance based on the power mode and device name.
    :param power_mode: The power monitoring mode (e.g., "PMIC", "INA3221").
    :param device_name: The normalized device name from get_device_name().
    :return: An instance of the appropriate power monitor, or None if not applicable.
    """
    mode = str(power_mode or "").strip().upper()
    name = str(device_name or "").strip().lower()

    # Prefer the normalized device name when available.  This avoids relying on
    # hostnames, which may not contain strings like "jetson" or "raspberry".
    if name in ("jetson_orin_nx", "jetson_orin_nano_super") or "jetson" in name:
        return INA3221()
    if name in ("rpi5", "raspberry_pi_5") or "raspberry" in name or "rpi" in name:
        return PMICMonitor()

    # Fall back to the explicitly requested power mode.
    if mode == "PMIC":
        return PMICMonitor()
    if mode == "INA3221":
        return INA3221()

    return None
