# https://github.com/librerpi/rpi-tools/blob/master/pi5_voltage.py

import subprocess
import time
import threading

def read_power():
    try:
        result = subprocess.run(['vgencmd', 'pmic_read_adc'], capture_output=True, text=True)
        power_value = float(result.stdout.strip())
        return power_value
    except Exception as e:
        print(f"Error reading power consumption: {e}")
        return None

def measure_power_consumption(function_to_measure, *args, **kwargs):
    def measure_power_during_function():
        nonlocal start_power, end_power
        start_power = read_power()
        if start_power is None:
            return
        function_to_measure(*args, **kwargs)
        end_power = read_power()
    
    start_power = None
    end_power = None
    
    thread = threading.Thread(target=measure_power_during_function)
    start_time = time.time()
    thread.start()
    thread.join()
    end_time = time.time()
    
    if start_power is None or end_power is None:
        print("Failed to measure power consumption")
        return

    time_taken = end_time - start_time
    power_consumed = end_power - start_power
    
    print(f"Function executed in: {time_taken} seconds")
    print(f"Power consumed: {power_consumed} units")
    
    return

# Example function to measure
def example_function(duration):
    time.sleep(duration)

# Measure the power consumption of the example_function
measure_power_consumption(example_function, 5)
