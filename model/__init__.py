import sys, os
sys.path.append(os.path.dirname(__file__))

from branchynet.net import BranchyNet
from networks.alex_cifar10 import get_network
from datasets.pcifar10 import get_test_data