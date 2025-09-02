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
parser.add_argument('--num_debert_blocks', default=4, type=int)
parser.add_argument('--num_division', default=4, type=int)
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

debert_model = DeBERT(50000, hidden=hidden_size, 
                     n_layers = n_layers,
                     attn_heads = attn_head,
                     num_parallel = world,
                     num_debert_blocks = args.num_debert_blocks,
                     num_division = args.num_debert_blocks)

input = torch.randint(low=0, high=49999, size=(bs,seq_len), dtype=torch.long)
segment_label = torch.zeros(bs, seq_len, dtype=torch.long)

total_comm_time = 0
total_comp_time = 0
total_time = 0
count = 0


print("\nwarm up communicate--------")
a = torch.randn(1,seq_len,hidden_size)
start_time = time.perf_counter()
for i in range(10):
    dist.all_reduce(a, dist.ReduceOp.SUM)
end_time = time.perf_counter()
print(f"{i}-th comm warp up time: {(end_time-start_time)/10:.6f} seconds.")


print("\nwarm up compute--------")
for i in range(2):
    with torch.no_grad():
        debert_res, comm_time, comp_time = debert_model(input, segment_label)
    print(f"DeBert {i}-th: comm time:  {(comm_time):.6f}. comp time:  {(comp_time):.6f}. total time per inference {(comm_time + comp_time):.6f}.")


print("\nbegin:---------")
for i in range(20):
    with torch.no_grad():
        debert_res, comm_time, comp_time = debert_model(input, segment_label)
    total_comm_time += comm_time
    total_comp_time += comp_time
    count += 1
    print(f"DeBert {i}-th: Avg comm time:  {(total_comm_time/count):.6f}. Avg comp time:  {(total_comp_time/count):.6f}. Avg total time per inference {((total_comm_time + total_comp_time)/count):.6f}.")



