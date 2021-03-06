from offline_profile import *
from DNN_inference import infer_main_branch, infer
from coinference_utils import load_model
import time
import sys
import torch
sys.path.append('../')
from model import get_network
from model import get_test_data


def get_fps():
    # 计算FPS，一秒可以运行多少帧，树莓派，服务器
    start_time = time.time()
    output = infer_main_branch(branchyNet, img)
    end_time = time.time()
    print("totaltime ", end_time - start_time)
    fps = 1 / (end_time - start_time)


def get_accuracy(branchyNet, ep, pp, qb, dataloader):
    for _ in range(1000):
        img, tag = dataloader.next()
        intermediate = infer(branchyNet, 'client', ep, pp, qb, img)
        result = infer(branchyNet, 'server', ep, pp, qb, intermediate)
        prob = torch.exp(result).detach().numpy().tolist()[0]
        pred = (prob.index(max(prob)), max(prob))


def get_latency(branchyNet, img):
    start_time = time.time()
    infer_main_branch(branchyNet, img)
    end_time = time.time()
    return end_time - start_time


if __name__ == "__main__":
    branchyNet = get_network()
    load_model(branchyNet)
    branchyNet.testing()

    dataloader = get_test_data()

    img, tag = dataloader.next()
    print(img.size())
    
    latency = 0
    for _ in range(10):
        img, tag = dataloader.next()
        latency += get_latency(branchyNet, img)
    print(latency / 10)
