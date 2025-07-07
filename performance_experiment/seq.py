import torch.nn as nn


class Seq(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(Seq, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        "Apply residual connection to any sublayer with the same size."
        # x is an input after sequence partitioning 
        y = x + self.dropout(x)
        v = self.norm(y)

        return v