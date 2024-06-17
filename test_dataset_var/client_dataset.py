from power import Monitor
import Monsoon.sampleEngine as sampleEngine
from log import WrlsEnv
from datetime import datetime
import subprocess, os, logging, time, socket, pickle
import paramiko, yaml
import re
import psutil
import argparse
import warnings
from collections import OrderedDict

import flwr as fl
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.models import resnet18, mobilenet_v3_small
from torchvision.transforms import Compose, Normalize, ToTensor
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from flwr_datasets import FederatedDataset


parser = argparse.ArgumentParser(description="Flower Embedded devices")
parser.add_argument(
    "--server_address",
    type=str,
    default="0.0.0.0:8080",
    help=f"gRPC server address (default '0.0.0.0:8080')",
)
parser.add_argument(
    "--cid",
    type=int,
    required=True,
    help="Client id. Should be an integer between 0 and NUM_CLIENTS",
)
parser.add_argument(
    "--dataset",
    type=str,
    default='MNIST',
    help="\{MNIST, CIFAR10\}",
)
parser.add_argument(
    "--Model",
    type=str,
    default='ResNet20',
    help="\{ResNet20, LSTM, ResNet34, VGG16, VGG19\}",
)
'''
parser.add_argument(
    "--mnist",
    action="store_true",
    help="If you use Raspberry Pi Zero clients (which just have 512MB or RAM) use "
    "MNIST",
)
'''
warnings.filterwarnings("ignore", category=UserWarning)
NUM_CLIENTS = 50


# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.debug, logging.info, logging.warning, logging.error, logging.critical
class CustomFormatter(logging.Formatter):
    
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def get_ip_address():
    try:
        hostname = socket.gethostname()
        ip_addr = socket.gethostbyname(hostname)
        return ip_addr
    except socket.error as e:
        print(f'Unable to get IP address: {e}')
        return None


def log_network_usage(log_file):
    net_io = psutil.net_io_counters()
    with open(log_file, 'a') as f:
        f.write(f"{time.time()},{net_io.bytes_sent},{net_io.bytes_recv}\n")


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')."""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(net, trainloader, optimizer, epochs, device):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        for batch in tqdm(trainloader):
            batch = list(batch.values())
            images, labels = batch[0], batch[1]
            optimizer.zero_grad()
            criterion(net(images.to(device)), labels.to(device)).backward()
            optimizer.step()

def test(net, testloader, device):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in tqdm(testloader):
            batch = list(batch.values())
            images, labels = batch[0], batch[1]
            outputs = net(images.to(device))
            labels = labels.to(device)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy

def prepare_dataset(dataset):
    """Get MNIST/CIFAR-10 and return client partitions and global testset."""
    if dataset == 'MNIST':
        fds = FederatedDataset(dataset="mnist", partitioners={"train": NUM_CLIENTS})
        img_key = "image"
        norm = Normalize((0.1307,), (0.3081,))
    else:#dataset == 'CIFAR10':
        fds = FederatedDataset(dataset="cifar10", partitioners={"train": NUM_CLIENTS})
        img_key = "img"
        norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    pytorch_transforms = Compose([ToTensor(), norm])

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch[img_key] = [pytorch_transforms(img) for img in batch[img_key]]
        return batch

    trainsets = []
    validsets = []
    for partition_id in range(NUM_CLIENTS):
        partition = fds.load_partition(partition_id, "train")
        # Divide data on each node: 90% train, 10% test
        partition = partition.train_test_split(test_size=0.1)
        partition = partition.with_transform(apply_transforms)
        trainsets.append(partition["train"])
        validsets.append(partition["test"])
    testset = fds.load_split("test")
    testset = testset.with_transform(apply_transforms)
    return trainsets, validsets, testset


# Flower client, adapted from Pytorch quickstart/simulation example
class FlowerClient(fl.client.NumPyClient):
    """A FlowerClient that trains a MobileNetV3 model for CIFAR-10 or a much smaller CNN
    for MNIST."""

    def __init__(self, trainset, valset, dataset):
        self.trainset = trainset
        self.valset = valset
        # Instantiate model
        if dataset == 'MNIST':
            self.model = Net()
        else:
            self.model = mobilenet_v3_small(num_classes=10)
        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # send model to device

    def set_parameters(self, params):
        """Set model weights from a list of NumPy ndarrays."""
        params_dict = zip(self.model.state_dict().keys(), params)
        state_dict = OrderedDict(
            {
                k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0])
                for k, v in params_dict
            }
        )
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        print("Client sampled for fit()")
        self.set_parameters(parameters)
        # Read hyperparameters from config set by the server
        batch, epochs = config["batch_size"], config["epochs"]
        # Construct dataloader
        #   Shuffle=True 
        trainloader = DataLoader(self.trainset, batch_size=batch, shuffle=True)
        # Define optimizer
        #   momentum: after step calculation, we keep the inertia of SGD to deal with fast training speed and local minima problems.
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        # Train
        train(self.model, trainloader, optimizer, epochs=epochs, device=self.device)
        # Return local model and statistics
        return self.get_parameters({}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        print("Client sampled for evaluate()")
        self.set_parameters(parameters)
        # Construct dataloader
        valloader = DataLoader(self.valset, batch_size=64)
        # Evaluate
        loss, accuracy = test(self.model, valloader, device=self.device)
        # Return statistics
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}


def main():
    '''
    server_address=192.168.0.6
    client_id=1

    echo "Enter the client id: "
    read client_id
    echo "client_id: $client_id"

    echo "Enter the server address: "
    read server_address
    echo "server_address: $server_address"

    # Run the default example (CIFAR-10)
    # python3 client_pytorch.py --cid=$client_id --server_address=$server_address

    # Use MNIST (and a smaller model) if your devices require a more lightweight workload
    python3.10 ./FLOWER_embedded_devices/client_pytorch.py --cid=$client_id --server_address=$server_address --mnist
    # python3.10 ./FLOWER_embedded_devices/client_pytorch.py --cid=0 --server_address=192.168.0.6 --mnist

        
    dataset = ['CIFAR']

    client_cmd = "python3.10 ./FLOWER_embedded_devices/client_pytorch.py --cid=$client_id --server_address=$server_address --mnist"
    server_cmd = "python3.10 ./FLOWER_embedded_devices/server.py --rounds $round --min_num_clients $num_clients --sample_fraction $sample_frac"

    '''

    args = parser.parse_args()
    print(args)

    assert args.cid < NUM_CLIENTS

    # default parameters
    root_path = os.path.abspath(os.getcwd())+'/'
    arg_dataset = args.dataset
    trainsets, valsets, _ = prepare_dataset(arg_dataset)

    # Set up logger
    logger = logging.getLogger("test")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)

    # Prepare a bucket to store the results.
    measurements_dict = []
    time_records = []

    # Log the start time.
    time_records.append(time.time())
    logger.info([f'Wi-Fi start: {time.time()}'])

    # Start Flower client setting its associated data partition
    fl.client.start_client(
        server_address=args.server_address,
        client=FlowerClient(
            trainset=trainsets[args.cid], valset=valsets[args.cid], dataset=arg_dataset
        ).to_client(),
    )

    # Log the end time.
    time_records.append(time.time())
    logger.info([f'Wi-Fi end: {time.time()}'])

    measurements_dict.append({'time': time_records})


    # Save the data.
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"data_{current_time}.pickle"
    with open(filename, 'wb') as handle:
        pickle.dump(measurements_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"The measurement data is saved as {filename}.")

if __name__ == "__main__":
    main()



