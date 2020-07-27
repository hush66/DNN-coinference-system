from Optimizer import *
from co_inference_config import *
import csv

BANDWIDTHS = [1000, 10000, 512]
optimizer = Optimizer(branchyNet, Q, INPUT_DATA_INFO, QUANTIZATION_BITS)
op_for_bandwidths = dict()
for B in BANDWIDTHS:
    op_for_bandwidths[B] = optimizer.overallOptimization(B)
with open('bandwidth_character.csv', 'wb') as f:
    w = csv.writer(f)
    w.writerows(op_for_bandwidths.items())



