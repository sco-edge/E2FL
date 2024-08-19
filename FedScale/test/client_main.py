# client_main.py
import flwr as fl
from mobile_client import MobileClient
from mobile_client_profile import MobileClientProfile

def start_mobile_client(client_id, model, x_train, y_train, x_test, y_test):
    # Define the mobile client profile
    profile = MobileClientProfile(
        client_id=client_id,
        cpu_speed=2.0,  # GHz
        network_latency=100,  # milliseconds
        battery_capacity=3000  # mAh
    )
    client = MobileClient(model, x_train, y_train, x_test, y_test, profile)
    fl.client.start_numpy_client("localhost:8080", client=client)

if __name__ == "__main__":
    # Example model and data
    model = ...  # Define your Keras/PyTorch model
    x_train, y_train = ...  # Load your training dataset
    x_test, y_test = ...  # Load your test dataset

    # Start clients with different profiles
    start_mobile_client("Client1", model, x_train, y_train, x_test, y_test)
    start_mobile_client("Client2", model, x_train, y_train, x_test, y_test)
    start_mobile_client("Client3", model, x_train, y_train, x_test, y_test)