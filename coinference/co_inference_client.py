import thriftpy2 as thriftpy
import numpy as np
import time

from thriftpy2.rpc import make_client

def client_start():
    partition_thrift = thriftpy.load('partition.thrift', module_name='partition_thrift')
    return make_client(partition_thrift.Partition, '127.0.0.1', 6000)

class Client:

    def __init__(self):
        self.client