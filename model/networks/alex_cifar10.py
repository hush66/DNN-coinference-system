from networks.utils import Flatten, PrintLayer, Branch, calculate_entropy
from .utils import *
from ..branchynet.net import BranchyNet
import torch.nn as nn


# Model layer presets
def norm():
    """
    A function used to built-in block of layers to norm the input
    """
    return [nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LocalResponseNorm(3, alpha=5e-05, beta=0.75)]

def pool():
    """
    A function used to apply pool operation
    """
    return [nn.MaxPool2d(kernel_size=3, stride=2)]

# Built-in block of convolutional layer to extract features
conv = lambda n: [nn.Conv2d(n, 32, 3, padding=1, stride=1), nn.ReLU()]

# Built-in block of fully-connected layer to classify the samples
cap = lambda n: [nn.MaxPool2d(kernel_size=2, stride=2), Flatten(), nn.Linear(n, 10)]

# model structure define
def gen_2b(branch1, branch2, branch3, branch4):
    """
    Generate the architecture of AlexNet with four side branches or four early exits

    branch1, branch2, branch3, branch4: Branch(nn.Module)
        The ordered side branches inserted in main branch, which is AlexNet
    """

    network = [
        nn.Conv2d(3, 32, 5, padding=2, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2),
        nn.LocalResponseNorm(3, alpha=5e-05, beta=0.75),
        nn.Conv2d(32, 64, 5, padding=2, stride=1),
        Branch(branch1),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2),
        nn.LocalResponseNorm(3, alpha=5e-05, beta=0.75),
        nn.Conv2d(64, 96, 3, padding=1, stride=1),
        Branch(branch2),
        nn.ReLU(),
        nn.Conv2d(96, 96, 3, padding=1, stride=1),
        Branch(branch3),
        nn.ReLU(),
        nn.Conv2d(96, 64, 3, padding=1, stride=1),
        Branch(branch4),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2),
        Flatten(),
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Dropout(0.5, True),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.5, True),
        # Branch([nn.Linear(128, 10)])
        nn.Linear(128, 10)
    ]
    return network


def get_network(percentTrainKeeps=1, lr=0.1):
    """
    Built a BranchyNet, where the main branch is a AlexNet and four side branches are inserted
    in main branch.
    """

    branch1 = norm() + conv(64) + conv(32) + cap(512)
    branch2 = norm() + conv(96) + cap(128)
    branch3 = conv(96) + cap(512)
    branch4 = conv(64) + cap(512)

    network = gen_2b(branch1, branch2, branch3, branch4)
    net = BranchyNet(network, lr)

    return net
