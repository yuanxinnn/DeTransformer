import os
import torch
import torch.distributed as dist
import time

import argparse

from bert_model import MegatronBERT, DeBERT, BERT

parser = argparse.ArgumentParser(description='rank: device id')
parser.add_argument('--rank', default=0, type=int)
parser.add_argument('--world', default=1, type=int)
parser.add_argument('--bs', default=1, type=int)
args = parser.parse_args()


rank = int(os.environ['RANK'])
world = int(os.environ['WORLD_SIZE'])
bs = args.bs
dist.init_process_group(backend="gloo", rank=rank, world_size=world)


print("Start Running")
hidden_size=768
attn_head=12
n_layers=12   
seq_len = 128


megatron_bert_model = MegatronBERT(50000, hidden=hidden_size, 
                     n_layers = n_layers,
                     attn_heads = attn_head,
                     num_parallel=world)


# assert seq_len % args.world == 0
input = torch.randint(low=0, high=49999, size=(bs,seq_len), dtype=torch.long)
segment_label = torch.zeros(bs, seq_len, dtype=torch.long)

total_comm_time = 0
total_comp_time = 0
total_time = 0
count = 0


print("\nwarm up communicate--------")
a = torch.randn(1,seq_len,hidden_size)
for i in range(10):
    start_time = time.perf_counter()
    dist.all_reduce(a, dist.ReduceOp.SUM)
    end_time = time.perf_counter()
    print(f"{i}-th comm warp up time: {(end_time-start_time):.6f} seconds.")

print("\nwarm up compute--------")
for i in range(2):
    with torch.no_grad():
        megatron_bert_res, comm_time, comp_time = megatron_bert_model(input, segment_label)
    print(f"Megatron {i}-th: comm time:  {(comm_time):.6f}. comp time:  {(comp_time):.6f}. total time per inference {(comm_time + comp_time):.6f}.")

print("\nbegin:---------")
for i in range(20):
    with torch.no_grad():
        megatron_bert_res, comm_time, comp_time = megatron_bert_model(input, segment_label)
    total_comm_time += comm_time
    total_comp_time += comp_time
    count += 1
    print(f"Megatron {i}-th: Avg comm time:  {(total_comm_time/count):.6f}. Avg comp time:  {(total_comp_time/count):.6f}. Avg total time per inference {((total_comm_time + total_comp_time)/count):.6f}.")




