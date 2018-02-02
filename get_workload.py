import nnvm.testing
import tvm
import mxnet as mx
import numpy as np
import time
import nnvm
import json
import argparse

from tvm.contrib import graph_runtime
from topi.nn.conv2d import Workload, _WORKLOADS
from mxnet import gluon
from mxnet.gluon.model_zoo.vision import get_model

parser = argparse.ArgumentParser(description='Search convolution workload.')
parser.add_argument('--model', type=str, required=True,
                    help="Pretrained model from gluon model zoo.")

num_classes = 1000
batch_size = 1
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_classes)

def get_conv2d_workload(model, in_dtype='float32', out_dtype='float32'):
    block = get_model(model, pretrained=True)
    net, params = nnvm.frontend.from_mxnet(block)
    g = nnvm.graph.create(net)
    g = nnvm.compiler.graph_attr.set_shape_inputs(g, {'data': data_shape})
    g = g.apply("InferShape")
    g_dict = json.loads(g.json())
    node_list = g_dict["nodes"]
    shape_list = g_dict['attrs']['shape'][1]
    node_map = g_dict["node_row_ptr"]
    workload_list = []
    workload_set = set()
    for workload in _WORKLOADS:
        workload_set.add(workload)

    for node in node_list:
        if node['op'] != 'conv2d':
            continue
        attrs = node["attrs"]
        if int(attrs["groups"]) != 1:
            continue
        input_index = node["inputs"][0][0]
        input_shape = shape_list[node_map[input_index]]
        if attrs["layout"] == "NCHW":
            height, width, in_filter = input_shape[2], input_shape[3], input_shape[1]
        else:
            height, width, in_filter = input_shape[1], input_shape[2], input_shape[3]
        out_filter = attrs["channels"]
        hkernel, wkernel = (attrs["kernel_size"])[1:-1].split(',')
        hpad, wpad = (attrs["padding"])[1:-1].split(',')
        hstride, wstride = (attrs["strides"])[1:-1].split(',')

        workload = Workload(*[in_dtype, out_dtype, height, width, in_filter, int(out_filter),
                              int(hkernel), int(wkernel), int(hpad), int(wpad), int(hstride), int(wstride)])
        if workload not in workload_set:
            workload_set.add(workload)
            workload_list.append(workload)

    return workload_list


if __name__ == "__main__":
    args = parser.parse_args()
    model = args.model
    workload_list = get_conv2d_workload(model)
    for workload in workload_list:
       print('Workload(\'%s\', \'%s\', %d, %d, %d, %d, %d, %d, %d, %d, %d, %d),' % (
           workload.in_dtype, workload.out_dtype, workload.height, workload.width,
           workload.in_filter, workload.out_filter, workload.hkernel, workload.wkernel,
           workload.hpad, workload.wpad, workload.hstride, workload.wstride))