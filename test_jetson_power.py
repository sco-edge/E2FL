import time
from power.INA3221 import INA3221

# Initialize power monitor
monitor = INA3221()  # Sample every 0.5 seconds

# Start monitoring
monitor.start(freq=0.5)
print("Power monitoring started. Collecting data for 10 seconds...")

# Let it run for 10 seconds
time.sleep(10)

# Stop monitoring
monitor.stop()
print("Power monitoring stopped.")

# Save data
monitor.save("power_log.csv")
print("Power data saved to power_log.csv")

# Clean up
monitor.close()