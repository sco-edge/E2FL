from log import WrlsEnv
from datetime import datetime
import subprocess, os, logging, time, socket, pickle
# from log.NetLogger import *
import psutil
import argparse
import warnings
from collections import OrderedDict
from power import _power_monitor_interface # PMIC
from power.powermon import get_power_monitor
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


import grpc # intercept grpc

'''
https://github.com/adap/flower/blob/ad811b5a0afc8bd32fb27305a8d0063f41a09ce5/src/py/flwr/client/app.py#L74
- app.py
-> Line: 645 
    connection, error_type = grpc_connection, RpcError
    _init_connection -> return conneciton ,address, error_type
   
   Line: 275 -> "connection, address, connection_error_type = _init_connection( ~ )
   Line 326 ~ ...
    -> Line 336: receive, send, create_node, delete_node, get_run = conn




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

'''


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
    """A FlowerClient that trains a model and manages network usage."""

    def __init__(self, trainset, valset, dataset, model, interface):
        self.trainset = trainset
        self.valset = valset
        self.dataset = dataset
        self.interface = interface  # 네트워크 인터페이스

        # 네트워크 상태 초기화
        self.start_net = None
        self.end_net = None

        # 모델 초기화
        self.model = self.initialize_model(model, dataset)
        if self.model is None:
            raise ValueError(f"Failed to initialize model '{model}' for dataset '{dataset}'.")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # send model to device

    # https://pytorch.org/vision/main/models.html
    # https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
    def initialize_model(self, model, dataset):
        """Initialize the model based on the dataset and model type."""
        try:
            if model == 'resnet18' and dataset == 'mnist':
                resnet18 = models.resnet18(pretrained=False)
                # Modify the first convolutional layer to accept 1 channel input
                resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                return resnet18
            elif model == 'resnet18':
                return models.resnet18(pretrained=False)
            elif model == 'resnext50':
                return models.resnext50_32x4d(pretrained=False)
            elif model == 'resnet50':
                return models.wide_resnet50_2(pretrained=False)
            elif model == 'vgg16' and dataset == 'mnist':
                vgg16 = models.vgg16(pretrained=False)
                vgg16.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
                return vgg16
            elif model == 'vgg16':
                return models.vgg16(pretrained=False)
            elif model == 'alexnet' and dataset == 'mnist':
                # Pretrained AlexNet model
                alexnet = models.alexnet(pretrained=False)
                # Modify the first convolutional layer to accept 1 channel input
                alexnet.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2, bias=False)
                return alexnet
            elif model == 'alexnet':
                return models.alexnet(pretrained=False)
            elif model == 'convnext_tiny' and dataset == 'mnist':
                # Create a ConvNeXt Tiny model
                convnext_tiny = models.convnext_tiny(pretrained=False)
                # Modify the first convolutional layer to accept 1 channel input
                convnext_tiny.stem[0] = nn.Conv2d(1, 96, kernel_size=4, stride=4, padding=0)
                return convnext_tiny
            elif model == 'convnext_tiny':
                return models.convnext_tiny(pretrained=False)
            elif model == 'squeezenet1' and dataset == 'mnist':
                # Pretrained SqueezeNet1 model
                squeezenet = models.squeezenet1_0(pretrained=False)
                # Modify the first convolutional layer to accept 1 channel input
                squeezenet.features[0] = nn.Conv2d(1, 96, kernel_size=7, stride=2, padding=3, bias=False)
                squeezenet.classifier[1] = nn.Conv2d(512, 10, kernel_size=1)
                return squeezenet
            elif model == 'squeezenet1':
                return models.squeezenet1_0(pretrained=False)
            elif model == 'densenet121' and dataset == 'mnist':
                # Pretrained DenseNet model
                densenet = models.densenet121(pretrained=False)
                # Modify the first convolutional layer to accept 1 channel input
                densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                return densenet
            elif model == 'densenet121':
                return models.densenet121(pretrained=False)
            elif model == 'densenet161':
                return models.densenet161(pretrained=False)
            elif model == 'inception_v3' and dataset == 'mnist':
                # Pretrained Inception V3 model
                inception_v3 = models.inception_v3(pretrained=False, aux_logits=False)
                # Modify the first convolutional layer to accept 1 channel input
                inception_v3.Conv2d_1a_3x3.conv = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=0, bias=False)
                return inception_v3
            elif model == 'inception_v3':
                return models.inception_v3(pretrained=False)
            elif model == 'googlenet' and dataset == 'mnist':
                # Pretrained GoogleNet model
                googlenet = models.googlenet(pretrained=False)
                # Modify the first convolutional layer to accept 1 channel input
                googlenet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                return googlenet
            elif model == 'googlenet':
                return models.googlenet(pretrained=False)
            elif model == 'shufflenet_v2' and dataset == 'mnist':
                # Pretrained ShuffleNet V2 model
                shufflenet_v2 = models.shufflenet_v2_x1_0(pretrained=False)
                # Modify the first convolutional layer to accept 1 channel input
                shufflenet_v2.conv1[0] = nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False)
                shufflenet_v2.fc = nn.Linear(1024, 10)
                return shufflenet_v2
            elif model == 'shufflenet_v2':
                return models.shufflenet_v2_x1_0(pretrained=False)
            elif model == 'mobilenet_v2' and dataset == 'mnist':
                # Pretrained MobileNet V2 model
                mobilenet_v2 = models.mobilenet_v2(pretrained=False)
                # Modify the first convolutional layer to accept 1 channel input
                mobilenet_v2.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
                mobilenet_v2.classifier[1] = nn.Linear(mobilenet_v2.last_channel, 10)
                return mobilenet_v2
            elif model == 'mobilenet_v2':
                return models.mobilenet_v2(pretrained=False)
            elif model == 'mobilenet_v3_small' and dataset == 'mnist':
                mobilenet_v3 = models.mobilenet_v3_small(pretrained=False)
                mobilenet_v3.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
                mobilenet_v3.classifier[3] = nn.Linear(mobilenet_v3.classifier[3].in_features, 10)
                return mobilenet_v3
            elif model == 'mobilenet_v3_small':
                return models.mobilenet_v3_small(pretrained=False)
            elif model == 'mnasnet1' and dataset == 'mnist':
                # Pretrained MNASNet model
                mnasnet = models.mnasnet1_0(pretrained=False)
                # Modify the first convolutional layer to accept 1 channel input
                mnasnet.layers[0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
                return mnasnet
            elif model == 'mnasnet1':
                return models.mnasnet1_0(pretrained=False)
            elif model == 'lenet':
                return LeNet()
            else:
                logging.warning(f"Model '{model}' is not recognized. Using default model 'Net'.")
                return Net()
        except Exception as e:
            logging.error(f"Error initializing model '{model}': {e}")
            return None

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

    def get_parameters(self):
        """Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def get_network_usage(self):
        """Get the current network usage for the specified interface."""
        net_io = psutil.net_io_counters(pernic=True)
        if self.interface in net_io:
            return {"bytes_sent": net_io[self.interface].bytes_sent, "bytes_recv": net_io[self.interface].bytes_recv}
        else:
            raise ValueError(f"Interface {self.interface} not found.")

    def fit(self, parameters, config):
        """Train the model on the training set."""
        logger.info(f"[{time.time()}] Client sampled for fit()")

        # 네트워크 사용량 기록
        self.end_net = self.get_network_usage()
        net_usage_sent = self.end_net["bytes_sent"] - self.start_net["bytes_sent"]
        net_usage_recv = self.end_net["bytes_recv"] - self.start_net["bytes_recv"]
        logger.info(f"Network usage during fit: [sent: {net_usage_sent}, recv: {net_usage_recv}]")

        # 모델 파라미터 설정
        self.set_parameters(parameters)

        # 하이퍼파라미터 읽기
        batch_size, epochs = config["batch_size"], config["epochs"]

        # 데이터 로더 생성
        trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True)

        # 옵티마이저 정의
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

        # 모델 학습
        logger.info(f"[{time.time()}] Starting training...")
        train(self.model, trainloader, optimizer, epochs, self.device)
        logger.info(f"[{time.time()}] Training completed.")

        # 학습 후 네트워크 상태 업데이트
        self.start_net = self.get_network_usage()

        # 로컬 모델과 통계 반환
        return self.get_parameters({}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the validation set."""
        logger.info(f"[{time.time()}] Client sampled for evaluate()")

        # 모델 파라미터 설정
        self.set_parameters(parameters)

        # 데이터 로더 생성
        valloader = DataLoader(self.valset, batch_size=64)

        # 모델 평가
        logger.info(f"[{time.time()}] Starting evaluation...")
        loss, accuracy = test(self.model, valloader, self.device)
        logger.info(f"[{time.time()}] Evaluation completed with accuracy: {accuracy}")

        # 평가 결과 반환
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}
    
    def measure_power_during_function(self, duration):
        """Measure power consumption during a specific duration."""
        try:
            logger.info(f"Starting power measurement for {duration} seconds.")
            start_power = Monitor.PowreMon(node='rpi5', vout=5.0, mode='PMIC')
            if start_power is None:
                logger.error("PMIC initialization failed.")
                return None
            time.sleep(duration)
            end_power = Monitor.PowreMon(node='rpi5', vout=5.0, mode='PMIC')
            power_consumed = end_power - start_power
            logger.info(f"Power consumed: {power_consumed} mW.")
            return power_consumed
        except Exception as e:
            logger.error(f"Error during power measurement: {e}")
            return None

    def measure_power_during_function(self, duration, power_monitor):
        """Measure power consumption during a specific duration."""
        if power_monitor is None:
            logger.info("Power monitoring is disabled.")
            return None

        try:
            logger.info(f"Starting power measurement for {duration} seconds.")
            power_monitor.start(freq=1)  # Start monitoring with 1-second intervals
            time.sleep(duration)
            elapsed_time, data_size = power_monitor.stop()
            logger.info(f"Power monitoring completed. Duration: {elapsed_time}s, Data size: {data_size} samples.")
            return elapsed_time, data_size
        except Exception as e:
            logger.error(f"Error during power measurement: {e}")
            return None

def validate_network_interface(interface):
    """
    Validate if the given network interface exists on the system.
    :param interface: Network interface name to validate.
    :return: True if the interface exists, False otherwise.
    """
    if interface in psutil.net_if_addrs():
        return True
    else:
        logging.error(f"Invalid network interface: {interface}")
        return False

def validate_and_get_network_interface(interface):
    """
    Validate if the given network interface exists on the system.
    If not, return the first available interface.
    :param interface: Network interface name to validate.
    :return: Valid network interface name.
    """
    available_interfaces = psutil.net_if_addrs().keys()
    if interface in available_interfaces:
        return interface
    else:
        logging.warning(f"Invalid network interface '{interface}'. Using the first available interface.")
        return next(iter(available_interfaces), None)

if __name__ == "__main__":
    # Set up logger
    logger = logging.getLogger("test")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    start_net, end_net, wlan_interf = 0, 0, 'wlan0'
    
    warnings.filterwarnings("ignore", category=UserWarning)
    NUM_CLIENTS = 50

    
    print("Client Start!")
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
    wlan_interf = args.interface if validate_network_interface(args.interface) else "wlan0"
    logger.info(f"Using network interface: {wlan_interf}")
    start_net = get_network_usage(wlan_interf)
    

    # Prepare a bucket to store the results.
    usage_record = {}
    if 'PMIC' in args.power:
        power_monitor = FlowerClient(None, None, None, None, None)
        power_consumed = power_monitor.measure_power_during_function(duration=10)
        if power_consumed is not None:
            logger.info(f"Measured power consumption: {power_consumed} mW.")

    # Initialize power monitor based on the --power argument and device name
    power_monitor = None
    if args.power != "None":
        power_monitor = get_power_monitor(args.power, device_name=socket.gethostname())

    # Measure power consumption if a power monitor is initialized
    if power_monitor:
        logger.info("Starting power monitoring...")
        power_monitor.start(freq=1)  # Start monitoring with 1-second intervals
        time.sleep(10)  # Example duration for monitoring
        elapsed_time, data_size = power_monitor.stop()
        if elapsed_time is not None:
            logger.info(f"Measured power consumption: Duration={elapsed_time}s, Data size={data_size} samples.")
        else:
            logger.warning("Power monitoring failed or returned no data.")

    # Start Flower client setting its associated data partition
    print(f"Client {args.cid} connecting to {args.server_address}")
    fl.client.start_client(
        server_address=args.server_address,
        client=FlowerClient(
            trainset=trainsets[args.cid],
            valset=valsets[args.cid],
            dataset=arg_dataset,
            model=args.model,
            interface=wlan_interf
        ),
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
