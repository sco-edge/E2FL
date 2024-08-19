import flwr as fl
from mobile_client_profile import MobileClientProfile

class MobileClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test, profile):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = y_test
        self.y_test = y_test
        self.profile = profile

    def get_parameters(self):
        return self.model.get_weights()

    def set_parameters(self, parameters):
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        if not self.profile.is_battery_sufficient():
            print(f"Client {self.profile.client_id}: Battery too low to continue training.")
            return self.get_parameters(), 0, {}

        self.set_parameters(parameters)
        
        # Simulate computation and network delay
        computation_time = len(self.x_train) / 1000  # Arbitrary scaling for computation time
        self.profile.simulate_computation(computation_time)
        self.profile.simulate_network_delay()

        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32)

        return self.get_parameters(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        if not self.profile.is_battery_sufficient():
            print(f"Client {self.profile.client_id}: Battery too low to evaluate.")
            return float('inf'), 0, {}

        self.set_parameters(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        return loss, len(self.x_test), {"accuracy": accuracy}