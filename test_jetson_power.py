import time
from power.INA3221 import INA3221

# Initialize power monitor
monitor = INA3221()  # Sample every 0.5 seconds

# Start monitoring
monitoring_time = 1 * 60 * 60 * 24
monitor.start(freq=0.5)
print(f"Power monitoring started. Collecting data for {monitoring_time} seconds...")

# Let it run for 10 seconds
time.sleep(monitoring_time) 

# Stop monitoring
monitor.stop()
print("Power monitoring stopped.")

# Save data
filename = "jetson_power_log_24h_20250131.csv"
monitor.save(filename)
print(f"Power data saved to {filename}")

# Clean up
monitor.close()