import torch
import torch.nn as nn
import torch.distributed as dist

import math
import time

from nn.layers import ParallelLinear1D_Col, ParallelLinear1D_Row, Linear1D

global attention_time
attention_time = 0

class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, 
                 hidden_size: int,
                 num_all_heads: int):
        super().__init__()
        # if hidden_size % num_all_heads != 0:
        #     raise ValueError(
        #         f"The hidden size ({hidden_size}) is not a multiple of the number of attention ")
        self.hidden_size = hidden_size 
        self.num_all_heads = num_all_heads
        self.head_size = hidden_size // num_all_heads

        # First GEMM: Calculate qkv 
        self.query_key_value = Linear1D(hidden_size, 3 * hidden_size, bias=True)
        # Second GEMM: Linear Operation
        self.dense = Linear1D(hidden_size, hidden_size, bias=True)

    def forward(self, hidden_states):
        start_time = time.perf_counter()
        #  hidden_states : (bs, seq_len, hidden)  

        # (bs, seq_len, 3*hidden_size)
        qkv_linear = self.query_key_value(hidden_states)

        # (bs, seq_len, num_all_heads, 3*head_size)
        new_qkv_shape = qkv_linear.shape[:-1] + (self.num_all_heads, 3 * self.head_size)
        attention_output = qkv_linear.view(new_qkv_shape)

        # (bs, num_all_heads, seq_len, 3*head_size)
        attention_output = attention_output.permute(0, 2, 1, 3)

        # (bs, num_all_heads, seq_len, head_size)
        q, k, v = torch.chunk(attention_output, 3, dim=-1)

        # Calculate q*k
        # (bs, num_all_heads, seq_len, seq_len)
        attention_output = torch.matmul(q, k.transpose(-1, -2))
        attention_output = attention_output / math.sqrt(self.head_size)
        attention_output = nn.functional.softmax(attention_output, dim=-1)  

        # Calculate (qk)*v
        # (bs, num_all_heads, seq_len, head_size)
        attention_output = torch.matmul(attention_output, v)

        # (bs, seq_len, num_all_heads, head_size)
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()

        # (bs, seq_len, hidden_size)
        new_context_layer_shape = attention_output.size()[:-2] + (self.num_all_heads*self.head_size,)
        attention_output = attention_output.reshape(new_context_layer_shape)
        context_layer = self.dense(attention_output)  

        end_time = time.perf_counter()
        attention_comp_time = end_time-start_time

        # Output size: [bs, seq_len, hidden_size]
        return context_layer, attention_comp_time



class ParallelMultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, 
                 hidden_size: int,
                 num_all_heads: int,
                 num_parallel: int,
                 num_par_heads: int):
        super().__init__()
        if hidden_size % num_all_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention ")
        self.hidden_size = hidden_size
        self.num_all_heads = num_all_heads
        self.num_parallel = num_parallel
        self.head_size = hidden_size // num_all_heads
        self.num_par_heads = num_par_heads

        # First GEMM: Calculate qkv 
        self.query_key_value = ParallelLinear1D_Col(hidden_size, 3 * hidden_size, self.num_parallel)

        # Second GEMM: Linear Operation
        self.dense = ParallelLinear1D_Row(hidden_size, hidden_size, self.num_parallel)

    def forward(self, hidden_states): 
        #  hidden_states : (bs, seq_len, hidden)

        rank, size = dist.get_rank(), dist.get_world_size()
        
        start_time = time.perf_counter()

        # (bs, seq_len, 3*hidden_size/num_parallel) [1, 384, 576]
        qkv_linear = self.query_key_value(hidden_states)


        # (bs, seq_len, num_par_heads, 3*head_size)
        new_qkv_shape = qkv_linear.shape[:-1] + (self.num_par_heads, 3 * self.head_size)
        attention_output = qkv_linear.view(new_qkv_shape)

        # (bs, num_par_heads, seq_len, 3*head_size)
        attention_output = attention_output.permute(0, 2, 1, 3)

        # (bs, num_par_heads, input_len, head_size)
        q, k, v = torch.chunk(attention_output, 3, dim=-1)

        # Calculate q*k
        # (bs, num_par_heads, seq_len, seq_len)
        attention_output = torch.matmul(q, k.transpose(-1, -2))
        attention_output = attention_output / math.sqrt(self.head_size)
        attention_output = nn.functional.softmax(attention_output, dim=-1)

        # Calculate (qk)*v
        # (bs, num_par_heads, seq_len, head_size)
        attention_output = torch.matmul(attention_output, v)

        # (bs, seq_len, num_par_heads, head_size)
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()

        # (bs, seq_len, hidden_size/num_parallel) 
        new_context_layer_shape = attention_output.size()[:-2] + (self.num_par_heads*self.head_size,)
        attention_output = attention_output.reshape(new_context_layer_shape)
        
        # second GEMM:  Output size: [bs, seq_len, hidden_size]  
        dense_output = self.dense(attention_output)

        end_time = time.perf_counter()
        attention_comp_time = end_time-start_time

        global attention_time
        attention_time += attention_comp_time
        
        dist.barrier()

        start_time = time.perf_counter()

        # all reduce
        dist.all_reduce(dense_output, dist.ReduceOp.SUM)

        end_time = time.perf_counter()
        attention_comm_time = end_time-start_time

        # Output size: [bs, seq_len, hidden_size]
        return dense_output, attention_comm_time, attention_comp_time
