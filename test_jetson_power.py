import time
from power import INA3221

# Initialize power monitor
monitor = INA3221(freq=0.5)  # Sample every 0.5 seconds

# Start monitoring
monitor.start()
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