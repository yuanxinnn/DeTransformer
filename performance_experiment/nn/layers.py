import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter
import torch.distributed as dist

import time

class Linear1D(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True):

        super().__init__()

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features

        # Parameters.
        # Initialize weight.
        self.weight = Parameter(torch.randn(self.out_features, self.in_features))

        if bias:
            self.bias = Parameter(torch.randn(self.out_features))
        else:
            self.bias = None

    def forward(self, input):
        # Matrix multiply.
        bias = self.bias
        output = F.linear(input, self.weight)

        return output

class ParallelLinear1D_Col(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 num_parallel: int,
                 bias: bool = True):

        super().__init__()

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features

        self.out_features_per_partition = int(out_features / num_parallel)
        # Parameters.
        # Initialize weight.
        self.weight = Parameter(torch.randn(self.out_features_per_partition, self.in_features))

        if bias:
            self.bias = Parameter(torch.randn(self.out_features_per_partition))
        else:
            self.bias = None

    # def split_weights_to_chunk(self):
    #     self.weights_list = torch.chunk(self.weight, self.num_chunk, dim=0)

    def forward(self, input_parallel):
        # Matrix multiply.
        # start_time = time.perf_counter()
        
        bias = self.bias
        output_parallel = F.linear(input_parallel, self.weight)

        return output_parallel


class ParallelLinear1D_Row(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 num_parallel: int,
                 bias: bool = True):
        super().__init__()

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features

        # Divide the weight matrix along the last dimension.
        self.input_size_per_partition = int(in_features / num_parallel)

        # Parameters.
        # Initialize weight.
        self.weight = Parameter(torch.randn(self.out_features, self.input_size_per_partition))

        if bias:
            self.bias = Parameter(torch.randn(self.out_features))
        else:
            self.bias = None

    def forward(self, input_parallel):
        # Matrix multiply.
        bias = self.bias
        output_parallel = F.linear(input_parallel, self.weight)
        return output_parallel

