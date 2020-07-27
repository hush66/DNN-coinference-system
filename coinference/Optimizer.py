from offline_profile import *
from pulp import *
import numpy as np


class Optimizer:

    def __init__(self, branchy_model, Q, input_data_size, quantization_bits):
        self.branches = [model for model in branchy_model.models]
        self.branches.append(branchy_model.main)

        self.model_config = ModelConfig(self.branches, input_data_size)
        self.device_time_predictor = DeviceTimePredictor(self.model_config)
        self.server_time_predictor = ServerTimePredictor(self.model_config)

        # Fixed configuration
        self.interval = 1 / Q
        self.default_branch_id = len(self.branches)-1
        self.quantization_bits = quantization_bits

    def LWO(self, B):
        """
        :param B: current bandwidth
        :return: optimal partition configuration under lightly workload
        partition point can be choosen from -[1,layer_number-1], 0 means the whole model is executed at server, while
        -1 means no optimal partition point, and
        system is in heavy workload.
        """

        main_branch_info = self.model_config.getLayerInfo(self.default_branch_id)
        layer_number = len(main_branch_info)
        device_layers_exe_time = self.device_time_predictor.getExeTime(self.default_branch_id)
        server_layer_exe_time = self.server_time_predictor.getExeTime(self.default_branch_id)
        data_size_table = self.model_config.getDataSizeTable(self.default_branch_id)

        opt_pp = -1
        total_time = 1<<10

        # p stands for partition point
        for pp in range(layer_number):
            device_exe_time = sum(device_layers_exe_time[:pp+1])
            server_exe_time = sum(server_layer_exe_time[pp+1:])
            trans_time = data_size_table[pp] / B
            
            # check if result in congestion
            if device_exe_time > self.interval or server_exe_time > self.interval or trans_time > self.interval:
                continue

            cur_total_time = device_exe_time + server_exe_time + trans_time
            if cur_total_time < total_time:
                total_time = cur_total_time
                opt_pp = pp

        return (self.default_branch_id, opt_pp, 0)

    def ILP(self, B):
        '''
        Integer Linear Programming
        :param B: current bandwidth
        :return: optimal partition configuration (b, p, c)
        '''

        b_list = [str(b) for b in range(self.model_config.branch_number)]
        c_list = [str(c) for c in self.quantization_bits]
        layers_in_each_branch = [len(branch_info) for branch_info in self.model_config.branches_info]
        max_layer_number = max(layers_in_each_branch)
        p_list = [str(p) for p in range(max_layer_number)]

        # problem define
        prob = LpProblem('Partition_problem', LpMinimize)

        # define variable
        choices = LpVariable.dicts("Choice",
                                   ((b, c, p) for b in b_list for c in c_list for p in p_list[:layers_in_each_branch[int(b)]]),
                                    cat='Binary')

        # Define objective function
        device_exe_time = self.device_time_predictor.each_branches_exe_time
        server_exe_time = self.server_time_predictor.each_branches_exe_time
        data_size_table = self.model_config.branches_output_data_size

        prob += lpSum([sum(device_exe_time[int(b)][:int(p)+1]) * choices[(b,c,p)] for b in b_list for c in c_list for p in p_list[:layers_in_each_branch[int(b)]] ]) \
            + lpSum([sum(server_exe_time[int(b)][int(p)+1:]) * choices[(b,c,p)] for b in b_list for c in c_list for p in p_list[:layers_in_each_branch[int(b)]] ]) \
            + lpSum([data_size_table[int(b)][int(p)] / (B * 32 / int(c)) * choices[(b,c,p)] for b in b_list for c in c_list for p in p_list[:layers_in_each_branch[int(b)]] ])

        # define constraints:
        prob += lpSum([choices[(b,c,p)] for b in b_list for c in c_list for p in p_list[:layers_in_each_branch[int(b)]]]) == 1, ""
        prob += lpSum([sum(device_exe_time[int(b)][:int(p)+1]) * choices[(b,c,p)] for b in b_list for c in c_list for p in p_list[:layers_in_each_branch[int(b)]] ]) <= self.interval
        prob += lpSum([sum(server_exe_time[int(b)][int(p)+1:]) * choices[(b,c,p)] for b in b_list for c in c_list for p in p_list[:layers_in_each_branch[int(b)]] ]) <= self.interval
        prob += lpSum([data_size_table[int(b)][int(p)] / (B * 32 / int(c)) * choices[(b,c,p)] for b in b_list for c in c_list for p in p_list[:layers_in_each_branch[int(b)]] ]) <= self.interval

        # slove problem
        status = prob.solve()
        if not LpStatus[prob.status] == "Optimal":
            return (-1, -1, -1)
        for b in b_list:
            for c in c_list:
                for p in p_list[:layers_in_each_branch[int(b)]]:
                    if choices[(b,c,p)].varValue == 1.0:
                        print(b, p, c)
                        print(sum(device_exe_time[int(b)][:int(p)+1]))
                        print(sum(server_exe_time[int(b)]))
                        print(data_size_table[int(b)][int(p)] / (B * 32 / int(c)))
                    return (int(b), int(p), int(c))

    def HWO(self, B):
        """
        Optimization under high workload
        :param B: current network bandwidth
        :return: if feasible partition configuration exist then return it or return (-1, -1, -1) and notice user
        to reduce sampling rate.
        """

        b, p, c = self.ILP(B)
        if b == -1:
            print('You have to reduce sampling rate.')
            return -1, -1, -1
        return b, p, c


    def overallOptimization(self, B):
        main_b, opt_pp, default_c = self.LWO(B)
        if opt_pp == -1:
            return self.HWO(B)
        return main_b, opt_pp, default_c


    def edgent_optimization(self, B):
        """
        optimization algorithm for edgent
        Args:
            B: current bandwidth
        Returns:
            exit point and partition point
        """
        layer_numbers = [len(branch) for branch in self.branches]
        device_exe_time = self.device_time_predictor.each_branches_exe_time
        server_exe_time = self.server_time_predictor.each_branches_exe_time
        data_size_table = self.model_config.branches_output_data_size

        opt_pp = -1
        opt_ep = -1
        total_time = 1 << 10

        # p stands for partition point
        for branch_id in range(len(self.branches)-1, -1, -1):
            for layer_id in range(layer_numbers[branch_id]):
                device_exe_time = sum(device_exe_time[branch_id][:layer_id+1])
                server_exe_time = sum(server_exe_time[branch_id][layer_id+1:])
                trans_time = data_size_table[branch_id][layer_id] / B

                cur_total_time = device_exe_time + server_exe_time + trans_time
                if cur_total_time < total_time:
                    total_time = cur_total_time
                    opt_pp, opt_ep = layer_id, branch_id
        return opt_ep, opt_pp


    def jalad_optimization(self, B):
        """
        optimization algorithm for JALAD
        Args:
            B: current bandwidth
        Returns:
            partition point

        """
        main_branch_info = self.model_config.getLayerInfo(self.default_branch_id)
        layer_number = len(main_branch_info)
        device_layers_exe_time = self.device_time_predictor.getExeTime(self.default_branch_id)
        server_layer_exe_time = self.server_time_predictor.getExeTime(self.default_branch_id)
        data_size_table = self.model_config.getDataSizeTable(self.default_branch_id)

        c_list = [c for c in self.quantization_bits]

        opt_pp = -1
        quantization_bit = 32
        total_time = 1 << 10

        # p stands for partition point
        for qb in c_list:
            for pp in range(layer_number):
                device_exe_time = sum(device_layers_exe_time[:pp + 1])
                server_exe_time = sum(server_layer_exe_time[pp + 1:])
                trans_time = data_size_table[pp] / (B * (32 / qb))

                # check if result in congestion
                if device_exe_time > self.interval or server_exe_time > self.interval or trans_time > self.interval:
                    continue
                cur_total_time = device_exe_time + server_exe_time + trans_time
                if cur_total_time < total_time:
                    total_time = cur_total_time
                    opt_pp = pp
                    quantization_bit = qb
            if opt_pp != -1:
                return opt_pp, quantization_bit

        for pp in range(layer_number):
            device_exe_time = sum(device_layers_exe_time[:pp + 1])
            server_exe_time = sum(server_layer_exe_time[pp + 1:])
            trans_time = data_size_table[pp] / (B * 4)
            cur_total_time = device_exe_time + server_exe_time + trans_time
            if cur_total_time < total_time:
                total_time = cur_total_time
                opt_pp = pp
                quantization_bit = 8
        return opt_pp, quantization_bit

