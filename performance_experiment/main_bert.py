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


bs = args.bs

print("Start Running")
hidden_size=768
attn_head=12
n_layers=12   
seq_len = 128


bert_model = BERT(50000, hidden=hidden_size, 
                     n_layers = n_layers,
                     attn_heads = attn_head)


# assert seq_len % args.world == 0
input = torch.randint(low=0, high=49999, size=(bs,seq_len), dtype=torch.long)
segment_label = torch.zeros(bs, seq_len, dtype=torch.long)

total_comm_time = 0
total_comp_time = 0
total_time = 0
count = 0


print("\nwarm up compute--------")
for i in range(2):
    with torch.no_grad():
        megatron_bert_res, comp_time = bert_model(input, segment_label)
    print(f"Megatron {i}-th: comp time:  {(comp_time):.6f}.")

print("\nbegin:---------")
for i in range(20):
    with torch.no_grad():
        megatron_bert_res, comp_time = bert_model(input, segment_label)

    total_comp_time += comp_time
    count += 1
    print(f"Megatron {i}-th: Avg comp time:  {(total_comp_time/count):.6f}. Avg total time per inference {(total_comp_time/count):.6f}.")
