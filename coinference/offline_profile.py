import torch.nn as nn
import numpy as np
import sys

sys.path.append('../')
from model.networks.utils import Flatten


###############################################
# DNN Model structure
###############################################
class ModelConfig:
    """
    A class used to represent specific model configuration, including structure and output of each layer

    notice:
           1.branchId starts from 0
           2.we Suppose that the length and width of the convolution kernel are equal for the input data!
    Parameters:
    width: input width
    channel: input channel
    branches: all branches in branchyNet, including main branch
    branches_info: a list includes all layers information for all branches
    branches_output_data_size: a list show the output data size of specific layer
    ----------------
    """
    def __init__(self, branches, input):
        self.width, self.channel = input

        self.branches = branches
        self.branch_number = len(self.branches)

        self.branches_info = []
        for branch in self.branches:
            self.branches_info.append(self.initFromModel(branch))

        self.branches_output_data_info = []
        self.branches_output_data_size = []
        for branch_info in self.branches_info:
            self.initDataTable(self.width, self.channel, branch_info)

        self.layers_number_of_each_branch = [len(branch_info) for branch_info in self.branches_info]


    def initFromModel(self, model):
        # Ignore flatten layer, layer id starts from 1
        layer_list = model.layers
        model_info = [['input', None]]

        for layer in layer_list:
            if isinstance(layer, nn.Conv2d):
                kernal_size = layer.kernel_size[0]
                stride = layer.stride[0]
                out_channels = layer.out_channels
                padding = layer.padding[0]
                params = {'kernal_size': kernal_size, 'stride': stride, 'out_channels':out_channels, 'padding': padding}
                model_info.append(['conv', params])
            elif isinstance(layer, nn.ReLU):
                model_info.append(['relu', None])
            elif isinstance(layer, nn.MaxPool2d):
                kernal_size = layer.kernel_size
                stride = layer.stride
                params = {'kernal_size':kernal_size, 'stride':stride}
                model_info.append(['pool', params])
            elif isinstance(layer, nn.LocalResponseNorm):
                model_info.append(['lrn', None])
            elif isinstance(layer, nn.Linear):
                in_features = layer.in_features
                out_features = layer.out_features
                params = {'in_features':in_features, 'out_features':out_features}
                model_info.append(['linear', params])
            elif isinstance(layer, nn.Dropout):
                model_info.append(['dropout', None])
            elif isinstance(layer, Flatten):
                model_info.append(['Flatten', None])

        return model_info

    def initDataTable(self, width, channel, branch_info):
        data_info_table = [(width, channel)]
        data_size_table = [width**2*channel]

        for layer_info in branch_info[1:]:
            if layer_info[0] == 'conv':
                params = layer_info[1]
                width = (width + 2*params['padding'] - params['kernal_size']) // params['stride'] + 1
                channel = params['out_channels']
                data_info_table.append( (width, channel) )
                data_size_table.append(width**2*channel)
            elif layer_info[0] == 'pool':
                params = layer_info[1]
                width = (width - params['kernal_size']) // params['stride'] + 1
                data_info_table.append( (width, channel) )
                data_size_table.append(width ** 2 * channel)
            elif layer_info[0] == 'linear':
                params = layer_info[1]
                width = np.sqrt(params['out_features'])
                channel = 1
                data_info_table.append( (width, channel) )
                data_size_table.append(width ** 2 * channel)
            else:
                data_info_table.append( (width, channel) )
                data_size_table.append(width ** 2 * channel)
        self.branches_output_data_info.append(data_info_table)
        self.branches_output_data_size.append(data_size_table)

        return


    def getDataInfoTable(self, branch_id):
        '''
        :param branch_id:
        :return: data information(width, channel) table of specific branch
        '''
        return self.branches_output_data_info[branch_id]

    def getLayerInfo(self, branch_id):
        return self.branches_info[branch_id]

    def getDataSizeTable(self, branc_id):
        '''
        :return: actual data size table of branchynet
        '''
        return self.branches_output_data_size[branc_id]


###############################################
# Base class of time predictor
###############################################
class TimePredictor:

    def __init__(self, model_config):
        self.model_config = model_config

        self.each_branches_exe_time = []
        for i, branch_info in enumerate(self.model_config.branches_info):
            self.each_branches_exe_time.append(self.layerExecutionTime(branch_info, i))

    # time predict function got from regression model
    def lrn(self, data_size):
        pass

    def pool(self, input_data_size, output_data_size):
        pass

    def relu(self, input_data_size):
        pass

    def dropout(self, input_data_size):
        pass

    def linear(self, input_data_size, output_data_size):
        pass

    def conv(self, computition_amount):
        pass

    def getDataInfo(self, branch_id):
        return self.model_config.branches_output_data_info[branch_id]

    def getExeTime(self, branch_id):
        return self.each_branches_exe_time[branch_id]

    def layerExecutionTime(self, branch_info, branch_id):
        exe_time = [0]
        data_info_table = self.getDataInfo(branch_id)

        for i, layer_info in enumerate(branch_info):
            if i == 0:
                continue

            layer_name, params = layer_info
            in_w, in_c = data_info_table[i - 1]
            out_w, out_c = data_info_table[i]
            input_data_size = in_w ** 2 * in_c
            output_data_size = out_w ** 2 * out_c

            if layer_name == 'conv':
                compution_each_pixel = params['out_channels'] * (params['kernal_size'] / params['stride']) ** 2
                exe_time.append(self.conv(input_data_size*compution_each_pixel))
            elif layer_name == 'pool':
                exe_time.append(self.pool(input_data_size, output_data_size))
            elif layer_name == 'lrn':
                exe_time.append(self.lrn(input_data_size))
            elif layer_name == 'relu':
                exe_time.append(self.relu(input_data_size))
            elif layer_name == 'dropout':
                exe_time.append(self.dropout(input_data_size))
            elif layer_name == 'linear':
                exe_time.append(self.linear(input_data_size, output_data_size))
            else:
                exe_time.append(0)  # Ignore calculation latency

        return exe_time


###############################################
# Mobile device side time prediction
###############################################
class DeviceTimePredictor(TimePredictor):

    def __init__(self, model_config):
        super(DeviceTimePredictor, self).__init__(model_config)

    # time predict function got from regression model
    def lrn(self, data_size):
        return 2.41040419e-07 * data_size + 0.00195032

    def pool(self, input_data_size, output_data_size):
        return 1.53823128e-07 * input_data_size - 3.95475651e-06 * output_data_size - 0.0002605685722191359

    def relu(self, input_data_size):
        return 1.10691116e-08 * input_data_size + 0.00045149

    def dropout(self, input_data_size):
        return -6.26868128e-07 * input_data_size +0.00645282

    def linear(self, input_data_size, output_data_size):
        return 5.00007390e-06 * input_data_size + 6.15883023e-05 * output_data_size - 0.015043842514137852

    def conv(self, computition_amount):
        return 1.43990934e-09 * computition_amount + 0.01242811


###############################################
# Edge server side time prediction class
###############################################
class ServerTimePredictor(TimePredictor):
    def __init__(self, model_config):
        super(ServerTimePredictor, self).__init__(model_config)

    # time predict function got from regression model
    def lrn(self, data_size):
        return 1.23390253e-08 * data_size + 7.49573351e-05

    def pool(self, input_data_size, output_data_size):
        return 1.45752786e-08 * input_data_size - 7.00468225e-06 * output_data_size - 5.201232083731043e-05

    def relu(self, input_data_size):
        return 5.57920434e-10 * input_data_size - 1.32500774e-05

    def dropout(self, input_data_size):
        return 1.04186335e-09 * input_data_size + 4.42568936e-05

    def linear(self, input_data_size, output_data_size):
        return 4.11475976e-08 * input_data_size + 5.44218793e-07 * output_data_size - 8.44561771318189e-05

    def conv(self, computition_amount):
        # compution_each_pixel stands for (filter size / stride)^2 * (number of filters)
        return 5.3933432e-12 * computition_amount + 0.00034535

