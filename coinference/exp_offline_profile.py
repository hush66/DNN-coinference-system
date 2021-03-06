import torch.nn as nn
import torch.nn.init
import torch
import pandas as pd
import time

# conv layers
computation = []
exe_time = []

feature_map = [32, 16, 8, 4, 2]
input_channel = [3, 32, 64, 96, 128]
output_f_number = [3, 32, 64, 96, 128]
filter_stride = [(3, 1, 1), (5, 1, 2), (7, 1, 3)]

for d in feature_map:
    for c in input_channel:
        for j in output_f_number:
            for f, s, p in filter_stride:
                if d < f: continue
                conv = nn.Conv2d(c, j, f, stride=s, padding=p)
                input_data = torch.rand(1, c, d, d)

                start_time = time.time()
                input_data = conv(input_data)
                end_time = time.time()

                exe_time.append(end_time - start_time)
                computation.append(d * d * c * ((f / s) ** 2) * j)

conv_dataframe = pd.DataFrame({'total_computation': computation, 'time': exe_time})
conv_dataframe.to_csv("./profile/Conv.csv", index=False, sep=',')

# Relu
input_data_size = []
exe_time = []

width, channel = [32, 16, 64, 128, 227], [3, 32, 64, 128, 192, 96]
for d in width:
    for c in channel:
        input_data = torch.rand(1, c, d, d)
        relu = nn.ReLU()

        start_time = time.time()
        input_data = relu(input_data)
        end_time = time.time()

        input_data_size.append(d*d*c)
        exe_time.append(end_time - start_time)
relu_dataframe = pd.DataFrame({'input_data_size': input_data_size, 'time': exe_time})
relu_dataframe.to_csv('./profile/ReLU.csv', index=False, sep=',')

# Pooling
input_data_size, output_data_size, exe_time = [], [], []

width = [32, 16, 8, 4, 227, 114, 64]
channel = [3, 32, 64, 96, 128]
kernal = [2]
stride = [2]

for w in width:
    for k in kernal:
        for s in stride:
            for c in channel:
                if w < k: continue
                input_size = w * w * c
                output_size = c * ((w - k) // s + 1)**2
                pooling = nn.MaxPool2d(k, stride=s)
                input_data = torch.rand(1, c, w, w)

                start_time = time.time()
                input_data = pooling(input_data)
                end_time = time.time()

                input_data_size.append(input_size)
                output_data_size.append(output_size)
                exe_time.append(end_time - start_time)
pooling_dataframe = pd.DataFrame(
    {'input_data_size': input_data_size, 'output_data_size': output_data_size, 'time': exe_time})
pooling_dataframe.to_csv('./profile/pooling.csv', index=False, sep=',')

# LRN
input_data_size, exe_time = [], []

width = [32, 16, 8, 4, 2, 227, 114]
channel = [3, 32, 64, 96, 128]

for w in width:
    for c in channel:
        input_data = torch.rand(1, c, w, w)
        lrn = nn.LocalResponseNorm(3, alpha=5e-05, beta=0.75)

        start_time = time.time()
        input_data = lrn(input_data)
        end_time = time.time()

        input_data_size.append(c * w * w)
        exe_time.append(end_time - start_time)
lrn_dataframe = pd.DataFrame({'input_data_size': input_data_size, 'time': exe_time})
lrn_dataframe.to_csv('./profile/lrn.csv', index=False, sep=',')

# FC
input_data_size, output_data_size, exe_time = [], [], []

configuration = [(1024, 512), (1024, 256), (512, 256), (256, 128), (128, 10), (64, 10)]

for i, o in configuration:
    input_data = torch.rand(1, i)
    ln = nn.Linear(i, o)
    nn.init.xavier_uniform(ln.weight)
    nn.init.constant(ln.bias, 0.1)

    start_time = time.time()
    input_data = ln(input_data)
    end_time = time.time()

    input_data_size.append(i)
    output_data_size.append(o)
    exe_time.append(end_time - start_time)

fc_dataframe = pd.DataFrame(
    {'input_data_size': input_data_size, 'output_data_size': output_data_size, 'time': exe_time})
fc_dataframe.to_csv('./profile/fc.csv', index=False, sep=',')

# Dropout
input_data_size, exe_time = [1024, 4096, 2048, 256, 512, 128], []

for i in input_data_size:
    input_data = torch.rand(1, i)
    dropout = nn.Dropout(0.5, True)

    start_time = time.time()
    input_data = dropout(input_data)
    end_time = time.time()

    exe_time.append(end_time - start_time)
dropout_dataframe = pd.DataFrame({'input_data_size': input_data_size, 'time': exe_time})
dropout_dataframe.to_csv('./profile/dropout.csv', index=False, sep=',')

