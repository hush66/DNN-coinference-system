import torch
import numpy as np


def quantization(inputs, qb):
    """
    Quantization every elements of input into given quantization bit range
    :param inputs:
    :param qb: quantization bit
    :return: data after quantization
    """
    if qb == 0:
        return inputs

    inputs = inputs.numpy()
    max_range = 2**qb
    max_value = np.max(inputs)
    min_value = np.min(inputs)
    np_types = {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}
    if max_value > max_range:
        inputs = (max_value - 1) * (inputs - min_value) / (max_value - min_value)
    inputs = inputs.astype(np_types[qb])
    print(inputs.dtype)
    return inputs


def infer(branchy_net, cORs, eb, pp, qb, input):
    """
    Branchy style model inference under specific partition configuration
    :param branchy_net: branchy style model
    :param cORs: client or server
    :param eb: exit branch
    :param pp: partition point
    :param qb: quantization bit, 0 means not to quantization
    :param input: input data
    :return: inference result
    """

    used_model = branchy_net.models[eb] if eb < len(branchy_net.models) else branchy_net.main

    if cORs == 'Client':
        for layer in used_model.layers[:pp+1]:
            input = layer(input)
    else:
        for layer in used_model.layers[pp+1:]:
            input = layer(input)
        return input
    return quantization(input, qb)


def infer_main_branch(branchy_net, inputs):
    """
    main branch model inference
    Args:
        branchy_net: branchy style model
        inuputs: input data

    Returns:
        inference result
    """
    main_branch = branchy_net.main
    for layer in main_branch.layers:
        inputs = layer(inputs)
    return inputs

