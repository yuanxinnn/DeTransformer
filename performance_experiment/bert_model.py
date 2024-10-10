import torch
import torch.nn as nn

import time

from transformer_layer import BertLayer, MegatronBertLayer
from embedding import BERTEmbedding

import torch.distributed as dist


class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden=768, n_layers=6, attn_heads=12, dropout=0.1, fp16=False):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden, fp16=fp16)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [BertLayer(hidden, attn_heads, self.feed_forward_hidden, 1, dropout) for _ in range(n_layers)])

    def forward(self, x, segment_info):

        # embedding the indexed sequence to sequence of vectors   [bz,seq,hidden]
        
        hidden_states = x
        
        global total_time, count
        total_time = 0
        count = 0

        for i, transformer in enumerate(self.transformer_blocks):
            start_time = time.perf_counter()
            hidden_states, _ = transformer.forward(hidden_states)
            end_time = time.perf_counter()
            total_time += (end_time-start_time)
            count += 1

        return hidden_states, total_time


class MegatronBERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden=768, n_layers=6, attn_heads=12, dropout=0.1, 
                num_parallel=1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.num_parallel = num_parallel
        

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = self.hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        self.num_par_heads = attn_heads // num_parallel

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [MegatronBertLayer(hidden, attn_heads, self.feed_forward_hidden, dropout, num_par_heads = self.num_par_heads, 
                                      num_parallel=self.num_parallel) for _ in range(n_layers)])

    def forward(self, x, segment_info):

        # embedding the indexed sequence to sequence of vectors   [bz,seq,hidden]
        # hidden_states = self.embedding(x, segment_info)
        
        hidden_states = x
        
        global total_comm_time, total_comp_time, count
        total_comm_time = 0
        total_comp_time = 0
        count = 0

        for i, transformer in enumerate(self.transformer_blocks):
            hidden_states, comm_time, comp_time = transformer.forward(hidden_states)
            total_comm_time += comm_time
            total_comp_time += comp_time  
            count += 1

        return hidden_states, total_comm_time, total_comp_time
    

class DeBERT(nn.Module):
    """
    DeBERT model : DeCoupled Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden=768, n_layers=6, attn_heads=12, dropout=0.1, 
                num_parallel=1, num_debert_blocks=4, num_division = 4):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.num_parallel = num_parallel
        self.debert_division = num_division
        self.num_debert_blocks = num_debert_blocks

        self.num_layers_per_block = n_layers // self.num_debert_blocks # eg. 3
        self.num_debert_layers_per_block = self.num_layers_per_block - 1

        self.num_par_heads = attn_heads // self.debert_division

        self.intermediate_size = self.hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # multi-layers debert blocks, deep network
        DeBert_blocks = []

        for i in range(self.num_debert_blocks):
            debert_layers = [BertLayer(hidden // self.debert_division, attn_heads, self.intermediate_size, self.num_parallel,
                                       dropout, self.debert_division) for _ in range(self.num_debert_layers_per_block)] 
            bert_layer = BertLayer(hidden, attn_heads, hidden * 4, 1 ,dropout) 

            # uncomment here for Plan BP+TP
            # bert_layer = MegatronBertLayer(hidden, attn_heads, self.intermediate_size, dropout, num_par_heads = self.num_par_heads, 
            #                           num_parallel=self.debert_division)  
            
            DeBert_blocks.extend(debert_layers + [bert_layer]) 

        self.layers = nn.ModuleList(DeBert_blocks)

    def forward(self, x, segment_info):
        rank, world_size = dist.get_rank(), dist.get_world_size()


        # embedding the indexed sequence to sequence of vectors   [bz,seq,hidden]
        hidden_states = x

        # split the embedding output
        embedding_output_split = torch.chunk(hidden_states, self.debert_division, dim = -1)

        # each process handle the corresponding data   [bz,seq,hidden//self.debert_division]
        hidden_states = embedding_output_split[rank]

        bs, seq_len = x.shape[0], x.shape[1]
        gather_list = [torch.zeros(bs, seq_len, self.hidden // self.num_parallel) for _ in range(self.num_parallel)]
        scatter_recv = torch.zeros(bs, seq_len, self.hidden // self.debert_division)

        global total_comm_time, total_comp_time, count
        total_comm_time = 0
        total_comp_time = 0
        count = 0

        time_decoupled = 0
        time_original = 0

        for i, layer_module in enumerate(self.layers):

            comp_time_deblock = 0
            comp_time_block = 0
            comm_time_block = 0
            comm_time_gather = 0


            if((i+1) % self.num_layers_per_block == 0):   # BertLayer
                # the optimal plan depends on bandwidth, compute capability and memory budget

                # Plan ：BP+TP
                # start_time = time.perf_counter()
                # dist.all_gather(gather_list, hidden_states)
                # end_time = time.perf_counter()
                # comm_time_gather = end_time-start_time
                # hidden_states = torch.cat(gather_list, dim=-1)
                # hidden_states, comm_time_block, comp_time_block = layer_module.forward(hidden_states)

                # Plan ：BP+Allgather  
                start_time = time.perf_counter()
                dist.all_gather(gather_list, hidden_states)
                end_time = time.perf_counter()
                comm_time_gather = end_time-start_time
                hidden_states = torch.cat(gather_list, dim=-1)
                hidden_states, comp_time_block = layer_module.forward(hidden_states)

                if(i != self.n_layers - 1):
                    hidden_states_list = list(hidden_states.chunk(self.debert_division, dim=-1))
                    hidden_states = hidden_states_list[rank]
                
            else:   # DeBertLayer 
                hidden_states, comp_time_deblock = layer_module.forward(hidden_states)  

            time_decoupled += comp_time_deblock
            time_original += comm_time_block

            total_comm_time += comm_time_block  + comm_time_gather
            total_comp_time += comp_time_block + comp_time_deblock
            # print("comp_time_block = " + str(comp_time_block) + " comp_time_deblock = " + str(comp_time_deblock) )

            count += 1

        return hidden_states, total_comm_time, total_comp_time

