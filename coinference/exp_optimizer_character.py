from Optimizer import *
from co_inference_config import *
from coinference_utils import load_model
from model import get_network
import csv


def get_bandwidth(max_bandwidth):
    bandwidths = []
    for b in range(50, max_bandwidth, 50):
        bandwidths.append(b * 1024)
    return bandwidths

if __name__ == "__main__":
    branchyNet = get_network()
    load_model(branchyNet)
    branchyNet.testing()

    optimizer = Optimizer(branchyNet, Q, INPUT_DATA_INFO, QUANTIZATION_BITS)
    op_for_bandwidths = dict()
    bandwidths = get_bandwidth(2000)
    for b in bandwidths:
        op_for_bandwidths[b] = optimizer.overallOptimization(b)
    print(op_for_bandwidths)
