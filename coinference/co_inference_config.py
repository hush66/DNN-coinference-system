import os

MODEL_LOCATION = os.path.join('..', 'model', 'trained_model', 'BranchyNet.pt')
INPUT_DATA_INFO = (32, 3)
QUANTIZATION_BITS = [32, 16, 8]
Q = 2
B = 100 * 1024

#HOST = '192.168.199.127'
HOST = '192.168.199.237'
PORT = 22
USER = 'pi'
PASSWORD = 'raspberry'
#LOCAL_DIR = './intermediate_data/'
LOCAL_DIR = 'C:/Users/Hu/Desktop/Workstation/DNN-coinference-system/coinference/'
#REMOTE_DIR = 'C:/Users/Hu/Desktop/Workstation/DNN-coinference-system/coinference/recevied/intermediate_data'
REMOTE_DIR = '/home/pi/WorkStation/'