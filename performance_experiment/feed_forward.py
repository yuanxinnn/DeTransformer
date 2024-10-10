import torch
import torch.nn as nn
import torch.distributed as dist

import time

# from utils.gelu import GELU
from nn.layers import ParallelLinear1D_Col, ParallelLinear1D_Row, Linear1D

global mlp_time, count
mlp_time = 0
count = 0

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self,
                 hidden_size, 
                 intermediate_size):
        super(PositionwiseFeedForward, self).__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.layer_0 = Linear1D(hidden_size, intermediate_size)
        self.layer_1 = Linear1D(intermediate_size, hidden_size)
        self.activation = nn.GELU()

    def forward(self, hidden_states):
        start_time = time.perf_counter()
        layer_0_output = self.layer_0(hidden_states)
        gelu_output = self.activation(layer_0_output)
        mlp_output = self.layer_1(gelu_output)
        end_time = time.perf_counter()
        mlp_comp_time = end_time-start_time
        return mlp_output, mlp_comp_time


class ParallelPositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self,
                 hidden_size, 
                 FF_hidden_size, 
                 num_parallel):
        super(ParallelPositionwiseFeedForward, self).__init__()

        self.layer_0 = ParallelLinear1D_Col(hidden_size, FF_hidden_size, num_parallel)
        self.layer_1 = ParallelLinear1D_Row(FF_hidden_size, hidden_size, num_parallel)

        self.activation = nn.GELU()


    def forward(self, hidden_states):
        # hidden_states [bs, seq_len, hidden_size]
        rank, size = dist.get_rank(), dist.get_world_size()

        start_time = time.perf_counter()

        # first GEMM [bs, seq_len, 4*hidden_size/num_parallel]  [1,384,4*768/4]
        mlp_output0 = self.layer_0(hidden_states)

        # gelu
        gelu_output = self.activation(mlp_output0) 

        # second GEMM [bs, seq_len, hidden_size]  [1,384,768]
        mlp_output = self.layer_1(gelu_output) 

        end_time = time.perf_counter()
        mlp_comp_time = end_time-start_time


        global mlp_time
        mlp_time += mlp_comp_time
        # print("total mlp time = " + str(mlp_time))


        global comm_total_time, count
        comm_total_time = 0
        count = 0

        dist.barrier()

        start_time = time.perf_counter()

        # all reduce 
        dist.all_reduce(mlp_output, dist.ReduceOp.SUM)

        end_time = time.perf_counter()
        mlp_comm_time = end_time-start_time
        
        # Output size: [bs, seq_len, hidden_size]
        return mlp_output, mlp_comm_time, mlp_comp_time