from log import WrlsEnv
from datetime import datetime
import subprocess, os, logging, time, socket, pickle
# from log.NetLogger import *
import psutil
import argparse
import warnings
from collections import OrderedDict
from power import Monitor # PMIC
import threading

import flwr as fl
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.models as models #import resnet18, mobilenet_v3_small
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.datasets import CIFAR10
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
    "--interface",
    type=str,
    default="wlan0",
    help="Wi-Fi Interface",
)
parser.add_argument(
    "--dataset",
    type=str,
    default='cifar10',
    help="\{mnist, cifar10, fashion_mnist, sasha/dog-food, 'zh-plus/tiny-imagenet\}",
) # the currently tested dataset are ['mnist', 'cifar10', 'fashion_mnist', 'sasha/dog-food', 'zh-plus/tiny-imagenet']
parser.add_argument(
    "--model",
    type=str,
    default='Net',
    help="\{resnet18, resnext50, resnet50, vgg16, alexnet, convnext_tiny, squeezenet1, densenet161, inception_v3, googlenet, shufflenet_v2, mobilenet_v2, mnasnet1\}",
)
parser.add_argument(
    "--power",
    type=str,
    default='None',
    help="\{None, PMIC, INA3221\}",
)
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

# Set up logger
logger = logging.getLogger("test")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)
start_net, end_net, wlan_interf = 0, 0, 'wlan0'

'''


'''
warnings.filterwarnings("ignore", category=UserWarning)
NUM_CLIENTS = 50


def get_network_usage(interf):
    net_io = psutil.net_io_counters(pernic=True)
    #net_io = psutil.net_io_counters(pernic=True)
    return {"bytes_sent": net_io[interf].bytes_sent, "bytes_recv": net_io[interf].bytes_recv}

def get_ip_address():
    try:
        hostname = socket.gethostname()
        ip_addr = socket.gethostbyname(hostname)
        return ip_addr
    except socket.error as e:
        print(f'Unable to get IP address: {e}')
        return None



class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')."""

    def __init__(self):
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
    """
    Get MNIST/CIFAR-10 and return client partitions and global testset.
    https://flower.ai/docs/baselines/
    https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html
    https://flower.ai/docs/datasets/ref-api/flwr_datasets.FederatedDataset.html#flwr_datasets.FederatedDataset.load_partition
    """
    if dataset == 'mnist':
        fds = FederatedDataset(dataset="mnist", partitioners={"train": NUM_CLIENTS})
        img_key = "image"
        norm = Normalize((0.1307,), (0.3081,))
    elif dataset == 'fashion_mnist':
        '''
        interrupted by custom code [y/N] message
        '''
        fds = FederatedDataset(dataset="fashion_mnist", partitioners={"train": NUM_CLIENTS})
        img_key = "image"
        norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif dataset == 'sasha/dog-food':
        fds = FederatedDataset(dataset="sasha/dog-food", partitioners={"train": NUM_CLIENTS})
        img_key = "image"
        norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif dataset == 'zh-plus/tiny-imagenet':
        fds = FederatedDataset(dataset="zh-plus/tiny-imagenet", partitioners={"train": NUM_CLIENTS})
        img_key = "image"
        norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else: # dataset == 'cifar10'
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

    def __init__(self, trainset, valset, dataset, model, start_net, end_net):
        self.trainset = trainset
        self.valset = valset
        # Instantiate model
        #if dataset == 'mnist':
        #    self.model = Net()
        #else:
        #    self.model = models.mobilenet_v3_small(num_classes=10)
        
        # https://pytorch.org/vision/main/models.html
        # https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
        if model == 'resnet18' and dataset == 'mnist':
            resnet18 = models.resnet18(pretrained=False)
            # Modify the first convolutional layer to accept 1 channel input
            resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.model = resnet18
        elif model == 'resnet18':
            # Pretrained ResNet model
            self.model = models.resnet18(pretrained=False)
        elif model == 'resnext50':
            self.model = models.resnext50_32x4d(pretrained=False)
        elif model == 'resnet50':
            self.model = models.wide_resnet50_2(pretrained=False)
        elif model == 'vgg16' and dataset == 'mnist':
            # Pretrained VGG model
            vgg16 = models.vgg16(pretrained=False)
            # Modify the first convolutional layer to accept 1 channel input
            vgg16.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
            self.model = vgg16
        elif model == 'vgg16':
            self.model = models.vgg16(pretrained=False)
        elif model == 'alexnet' and dataset == 'mnist':
            # Pretrained AlexNet model
            alexnet = models.alexnet(pretrained=False)
            # Modify the first convolutional layer to accept 1 channel input
            alexnet.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2, bias=False)
            self.model = alexnet
        elif model == 'alexnet':
            self.model = models.alexnet(pretrained=False)
        elif model == 'convnext_tiny' and dataset == 'mnist':
            # Create a ConvNeXt Tiny model
            convnext_tiny = models.convnext_tiny(pretrained=False)
            # Modify the first convolutional layer to accept 1 channel input
            convnext_tiny.stem[0] = nn.Conv2d(1, 96, kernel_size=4, stride=4, padding=0)
            self.model = convnext_tiny
        elif model == 'convnext_tiny':
            self.model = models.convnext_tiny(pretrained=False)
        elif model == 'squeezenet1' and dataset == 'mnist':
            # Pretrained SqueezeNet1 model
            squeezenet = models.squeezenet1_0(pretrained=False)
            # Modify the first convolutional layer to accept 1 channel input
            squeezenet.features[0] = nn.Conv2d(1, 96, kernel_size=7, stride=2, padding=3, bias=False)
            squeezenet.classifier[1] = nn.Conv2d(512, 10, kernel_size=1)
            self.model = squeezenet
        elif model == 'squeezenet1':
            self.model = models.squeezenet1_0(pretrained=False)
        elif model == 'densenet121' and dataset == 'mnist':
            # Pretrained DenseNet model
            densenet = models.densenet121(pretrained=False)
            # Modify the first convolutional layer to accept 1 channel input
            densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.model = densenet
        elif model == 'densenet121':
            self.model = models.densenet121(pretrained=False)
        elif model == 'densenet161':
            self.model = models.densenet161(pretrained=False)
        elif model == 'inception_v3' and dataset == 'mnist':
            # Pretrained Inception V3 model
            inception_v3 = models.inception_v3(pretrained=False, aux_logits=False)
            # Modify the first convolutional layer to accept 1 channel input
            inception_v3.Conv2d_1a_3x3.conv = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=0, bias=False)
            self.model = inception_v3
        elif model == 'inception_v3':
            self.model = models.inception_v3(pretrained=False)
        elif model == 'googlenet' and dataset == 'mnist':
            # Pretrained GoogleNet model
            googlenet = models.googlenet(pretrained=False)
            # Modify the first convolutional layer to accept 1 channel input
            googlenet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.model = googlenet
        elif model == 'googlenet':
            self.model = models.googlenet(pretrained=False)
        elif model == 'shufflenet_v2' and dataset == 'mnist':
            # Pretrained ShuffleNet V2 model
            shufflenet_v2 = models.shufflenet_v2_x1_0(pretrained=False)
            # Modify the first convolutional layer to accept 1 channel input
            shufflenet_v2.conv1[0] = nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False)
            shufflenet_v2.fc = nn.Linear(1024, 10)
            self.model = shufflenet_v2
        elif model == 'shufflenet_v2':
            self.model = models.shufflenet_v2_x1_0(pretrained=False)
        elif model == 'mobilenet_v2' and dataset == 'mnist':
            # Pretrained MobileNet V2 model
            mobilenet_v2 = models.mobilenet_v2(pretrained=False)
            # Modify the first convolutional layer to accept 1 channel input
            mobilenet_v2.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
            mobilenet_v2.classifier[1] = nn.Linear(mobilenet_v2.last_channel, 10)
            self.model = mobilenet_v2
        elif model == 'mobilenet_v2':
            self.model = models.mobilenet_v2(pretrained=False)
        elif model == 'mobilenet_v3_small' and dataset == 'mnist':
            mobilenet_v3 = models.mobilenet_v3_small(pretrained=False)
            mobilenet_v3.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
            mobilenet_v3.classifier[3] = nn.Linear(mobilenet_v3.classifier[3].in_features, 10)
            self.model = mobilenet_v3
        elif model == 'mobilenet_v3_small':
            self.model = models.mobilenet_v3_small(pretrained=False)
        elif model == 'mnasnet1' and dataset == 'mnist':
            # Pretrained MNASNet model
            mnasnet = models.mnasnet1_0(pretrained=False)
            # Modify the first convolutional layer to accept 1 channel input
            mnasnet.layers[0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
            self.model = mnasnet
        elif model == 'mnasnet1':
            self.model = models.mnasnet1_0(pretrained=False)
        elif model == 'lenet':
            self.model = LeNet()
        else: #default:
            self.model = Net()
        
        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # send model to device

        self.start_net = start_net
        self.end_net = end_net

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
        logger.info(f"[{time.time()}] Client sampled for fit()")

        start_time = time.time()
        if start_net != 0:
            logger.info([f'[{time.time()}] Communication end: {start_time - end_time}'])
            #global wlan_interf, start_net, end_net
            self.end_net = get_network_usage(wlan_interf)
            net_usage_sent = self.end_net["bytes_sent"] - self.start_net["bytes_sent"]
            net_usage_recv = self.end_net["bytes_recv"] - self.start_net["bytes_recv"]
            logger.info([f'[{time.time()}] Computation phase (evaluation) ({wlan_interf}): [sent: {net_usage_sent}, recv: {net_usage_recv}]'])
        else:
            logger.info(f"[{time.time()}] Client initialization")

        self.set_parameters(parameters)
        # Read hyperparameters from config set by the server
        batch, epochs = config["batch_size"], config["epochs"]
        # Construct dataloader
        #   Shuffle=True 
        trainloader = DataLoader(self.trainset, batch_size=batch, shuffle=True)
        # Define optimizer
        #   momentum: after step calculation, we keep the inertia of SGD to deal with fast training speed and local minima problems.
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        
        logger.info(f"[{time.time()}] Computation phase (fit) started")
        start_time = time.time()
        
        # Train
        train(self.model, trainloader, optimizer, epochs=epochs, device=self.device)
        
        end_time = time.time()
        self.start_net = get_network_usage(wlan_interf)
        computation_time = end_time - start_time
        logger.info(f"[{time.time()}] Computation pahse (fit) completed in {computation_time} seconds.")
        logger.info([f'[{time.time()}] Communication start: {end_time}'])

        # Return local model and statistics
        return self.get_parameters({}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        logger.info(f"[{time.time()}] Client sampled for evaluate()")

        start_time = time.time()
        logger.info([f'[{time.time()}] Communication end: {start_time - end_time}'])

        #global wlan_interf, start_net, end_net
        self.end_net = get_network_usage(wlan_interf)
        net_usage_sent = self.end_net["bytes_sent"] - self.start_net["bytes_sent"]
        net_usage_recv = self.end_net["bytes_recv"] - self.start_net["bytes_recv"]
        logger.info([f'[{time.time()}] Computation phase (fit) ({wlan_interf}): [sent: {net_usage_sent}, recv: {net_usage_recv}]'])
        

        self.set_parameters(parameters)
        # Construct dataloader
        valloader = DataLoader(self.valset, batch_size=64)

        start_time = time.time()
        

        # Evaluate
        loss, accuracy = test(self.model, valloader, device=self.device)

        global start_net
        end_time = time.time()
        self.start_net = get_network_usage(wlan_interf)
        computation_time = end_time - start_time
        logger.info(f"[{time.time()}] Computation pahse (evaluation) completed in {computation_time} seconds.")

        logger.info([f'[{time.time()}] Communication start: {end_time}'])
        # Return statistics
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}

def measure_power_during_function(logger, duration):
    start_power = Monitor.PowreMon(node = 'rpi5', vout = 5.0, mode = 'PMIC')
    if start_power is None:
        logger.error("PMIC FAILED")
        exit(1)
    time.sleep(duration)
    end_power = Monitor.PowreMon(node = 'rpi5', vout = 5.0, mode = 'PMIC')

def main():
    
    args = parser.parse_args()
    logger.info(args)
    pid = psutil.Process().ppid()
    logger.info(f"[{time.time()}] PPID: {pid}")
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    logging.basicConfig(filename=f"fl_info_{args.cid}_{args.dataset}_{current_time}.txt")
    fl.common.logger.configure(identifier="myFlowerExperiment", filename=f"fl_log_{args.cid}_{args.dataset}_{current_time}.txt")
    logger.info([f'[{time.time()}] Client Start!'])

    assert args.cid < NUM_CLIENTS

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # default parameters
    root_path = os.path.abspath(os.getcwd())+'/'
    arg_dataset = args.dataset
    trainsets, valsets, _ = prepare_dataset(arg_dataset)
    global wlan_interf, start_net, end_net
    wlan_interf = args.interface
    start_net = get_network_usage(wlan_interf)
    

    # Prepare a bucket to store the results.
    usage_record = {}
    if 'PMIC' in args.power:
        thread = threading.Thread(target = measure_power_during_function)
        start_time = time.time()
        thread.start()
        thread.join()
        end_time = time.time()
        logger.info()
        
    # Start Flower client setting its associated data partition
    fl.client.start_client(
        server_address=args.server_address,
        client=FlowerClient(
            trainset=trainsets[args.cid], valset=valsets[args.cid], dataset=arg_dataset, model=args.model, start_net=start_net, end_net=end_net
        ).to_client(),
    )

    end_time = time.time()
    logger.info([f'[{time.time()}] Communication end: {end_time}'])

    # Log the network IO
    end_net = get_network_usage(wlan_interf)
    net_usage_sent = end_net["bytes_sent"] - start_net["bytes_sent"]
    net_usage_recv = end_net["bytes_recv"] - start_net["bytes_recv"]
    logger.info([f'[{time.time()}] Evaluation phase ({wlan_interf}): [sent: {net_usage_sent}, recv: {net_usage_recv}]'])

    '''
    usage_record["execution_time"] = end_time - start_time
    usage_record["bytes_sent"] = end_net["bytes_sent"] - start_net["bytes_sent"]
    usage_record["bytes_recv"] = end_net["bytes_recv"] - start_net["bytes_recv"]


    # Save the data.
    
    filename = f"data_{args.cid}_{current_time}.pickle"
    with open(filename, 'wb') as handle:
        pickle.dump(usage_record, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"The measurement data is saved as {filename}.")
    '''

if __name__ == "__main__":
    main()

import grpc
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoggingInterceptor(grpc.UnaryUnaryClientInterceptor):
    def intercept_unary_unary(self, continuation, client_call_details, request):
        method = client_call_details.method
        start_time = time.time()
        response = continuation(client_call_details, request)
        end_time = time.time()
        logger.info(f"Method: {method}, Start: {start_time}, End: {end_time}, Duration: {end_time - start_time}")
        return response
def create_channel():
    channel = grpc.insecure_channel('localhost:8080')
    intercept_channel = grpc.intercept_channel(channel, LoggingInterceptor())
    return intercept_channel

def start_flower_client():
    channel = create_channel()
    flower_client = FlowerClient(channel)
    flower_client.start()

if __name__ == "__main__":
    start_flower_client()
