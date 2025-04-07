"""E2FL: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

'''
def initialize_model(self, model, dataset):
    """Initialize the model based on the dataset and model type."""
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
'''


fds = None  # Cache FederatedDataset


def load_data(partition_id: int, num_partitions: int):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
