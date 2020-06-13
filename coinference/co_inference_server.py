import thriftpy2 as thriftpy
import numpy as np
import torch
import os
from thriftpy2.rpc import make_server
from DNN_inference import infer
from utils import *
from co_inference_config import *

import sys
sys.path.append('../')
from model import get_network


branchyNet = get_network()
load_model(branchyNet)
branchyNet.testing()


def server_start():
    partition_thrift = thriftpy.load('partition.thrift', module_name='partition_thrift')
    server = make_server(partition_thrift.Partition, Dispacher(), '127.0.0.1', 6000)
    print('Thriftpy server is listening...')
    server.serve()


class Dispacher(object):
    def partition(self, ep, pp, qb):
        file_path = os.path.join(REMOTE_DIR, "intermediate_{b}_{p}_{c}.npy".format(b=str(ep), p=str(pp), c=str(qb)))
        readed = np.load(file_path)
        input = torch.from_numpy(readed)
        out = infer(branchyNet, 'Server', ep, pp, qb, input)
        prob = torch.exp(out).detach().numpy().tolist()[0]
        pred = str((prob.index(max(prob)), max(prob)))
        return pred


if __name__ == '__main__':
    server_start()
