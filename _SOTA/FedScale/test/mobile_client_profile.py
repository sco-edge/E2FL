class MobileClientProfile:
    def __init__(self, client_id, cpu_speed, network_latency, battery_capacity):
        self.client_id = client_id
        self.cpu_speed = cpu_speed  # GHz
        self.network_latency = network_latency  # milliseconds
        self.battery_capacity = battery_capacity  # milliamp hours (mAh)
        self.current_battery = battery_capacity  # Current battery level

    def simulate_computation(self, computation_time):
        # Simulate computation by reducing battery based on CPU speed and time
        battery_usage = computation_time * (self.cpu_speed / 2.0)  # Arbitrary scaling
        self.current_battery -= battery_usage
        if self.current_battery < 0:
            self.current_battery = 0
        return battery_usage

    def simulate_network_delay(self):
        # Simulate network latency
        import time
        time.sleep(self.network_latency / 1000.0)  # Convert ms to seconds

    def is_battery_sufficient(self):
        # Check if battery is sufficient for more computation
        return self.current_battery > 0

