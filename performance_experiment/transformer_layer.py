import torch.nn as nn
import time

from multi_head import ParallelMultiHeadedAttention, MultiHeadedAttention
from feed_forward import ParallelPositionwiseFeedForward, PositionwiseFeedForward
from seq import Seq

total_sum = 0
count = 0
 
class BertLayer(nn.Module):  
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, num_parallel, dropout, num_division=1):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.num_parallel = num_parallel
        self.num_division = num_division
        self.attention = MultiHeadedAttention(hidden_size=hidden, num_all_heads=attn_heads)
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden, intermediate_size=feed_forward_hidden)
        self.Seq_1 = Seq(size=hidden, dropout=dropout)
        self.Seq_2 = Seq(size=hidden, dropout=dropout)
        
    def forward(self, x):
        comp_time = 0
        for i in range(self.num_division // self.num_parallel):

            x, attention_comp_time = self.attention(x) 
            x = self.Seq_1(x)
            x, mlp_comp_time = self.feed_forward(x)
            x = self.Seq_2(x)
            
            comp_time += attention_comp_time + mlp_comp_time

        return x, comp_time


class MegatronBertLayer(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout, num_par_heads, num_parallel=1):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = ParallelMultiHeadedAttention(hidden_size=hidden,num_all_heads=attn_heads, 
                                                      num_parallel=num_parallel, num_par_heads=num_par_heads)
        self.feed_forward = ParallelPositionwiseFeedForward(hidden_size=hidden, FF_hidden_size=feed_forward_hidden,
                                                            num_parallel=num_parallel)
        self.Seq_1 = Seq(size=hidden, dropout=dropout)
        self.Seq_2 = Seq(size=hidden, dropout=dropout)
        
    def forward(self, x):

        comp_time = 0
        comm_time = 0

        x, attention_comm_time, attention_comp_time = self.attention(x) # Output size: [bs, seq_len, hidden_size]
        x = self.Seq_1(x)
        x, mlp_comm_time, mlp_comp_time = self.feed_forward(x)
        x = self.Seq_2(x)

        comm_time = attention_comm_time + mlp_comm_time
        comp_time += attention_comp_time + mlp_comp_time

        return x, comm_time, comp_time
