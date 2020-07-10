import pypeln as pl
import time
import paramiko
import thriftpy2 as thriftpy

from thriftpy2.rpc import make_client
from Optimizer import *

from DNN_inference import infer
from co_inference_config import *
from coinference_utils import  *

import sys
sys.path.append('../')
from model import get_network
from model import get_test_data

def generation_stage(img_tag):
    time.sleep(1/Q)
    print('generate frame...')
    return img_tag


def device_exe_stage(img_tag):
    print('Device start process')
    b, p, c = optimizer.overallOptimization(B)
    img, tag = img_tag
    output = infer(branchyNet, 'Client', b, p, c, img)
    print('get answer')
    intermediate = output.detach().numpy()

    file_name = "intermediate_{b}_{p}_{c}.npy".format(b=str(b), p=str(p), c=str(c))
    print(file_name)
    np.save(LOCAL_DIR + file_name, intermediate)
    print('file saved')
    return file_name, b, p, c


def upload_stage(part_result):
    filename, b, p, c = part_result
    print('upload stage')
    print(filename)
    local_path = os.path.join(LOCAL_DIR, filename)
    remote_path = os.path.join(REMOTE_DIR, filename)

    print(remote_path)
    sftp.put(local_path, remote_path)
    print('file sended')
    return b, p, c


def server_exe_stage(partition_config):
    print('server execution')
    b, p, c = partition_config
    print(b, p, c)
    tag = rpc_client.partition(b, p, c)
    print('server finish')
    return tag


def client_start():
    partition_thrift = thriftpy.load('partition.thrift', module_name='partition_thrift')
    return make_client(partition_thrift.Partition, '127.0.0.1', 6000)


if __name__ == '__main__':
    # init resource objects
    branchyNet = get_network()
    load_model(branchyNet)
    branchyNet.testing()

    optimizer = Optimizer(branchyNet, Q, INPUT_DATA_INFO, QUANTIZATION_BITS)
    sf = paramiko.Transport((HOST, PORT))
    sf.connect(username=USER, password=PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(sf)
    rpc_client = client_start()

    def on_start():
        return dict(optimizer=optimizer, branchyNet=branchyNet, sftp=sftp, rpc_client = rpc_client)

    # init data to be processed
    dataloader = get_test_data()
    print('Start process data...')
    data = [dataloader.next() for _ in range(3)]
    print('Finsh process data...')

    start_time = time.time()
    stage = (
            data
            | pl.thread.map(generation_stage, workers=1, maxsize=1)
            | pl.thread.map(device_exe_stage, workers=1, maxsize=1, on_start=on_start)
            | pl.thread.map(upload_stage, workers=1, maxsize=1, on_start=on_start)
            | pl.thread.map(server_exe_stage, workers=1, maxsize=1, on_start=on_start)
            | list
    )
    end_time = time.time()
    print(end_time - start_time)

    sf.close()