import torch
from torch import nn, Tensor
from torch import distributions as D
from torch.nn import init
from typing import Union
from sys import stderr


class BiAffine(nn.Module):

    def __init__(self, in_dim: int, mid_dim: int = None, classes: int = 1, with_left_edge_bias: bool = True, with_right_edge_bias: bool = True):
        super().__init__()
        if mid_dim is None:
            mid_dim = in_dim
        self.mlp_left = nn.Linear(in_dim, mid_dim)
        self.mlp_right = nn.Linear(in_dim, mid_dim)
        self.weight = nn.Parameter(torch.empty((classes, mid_dim, mid_dim)))
        init.kaiming_uniform_(self.weight)
        if with_right_edge_bias:
            self.Uright = nn.Parameter(torch.empty((mid_dim, classes)))
            init.kaiming_uniform_(self.Uright)
        if with_left_edge_bias:
            self.Uleft = nn.Parameter(torch.empty((mid_dim, classes)))
            init.kaiming_uniform_(self.Uleft)

    def forward(self, x: Tensor, mask: Tensor, diagonal: int = 1) -> Tensor:
        """
        x -- (b,n,d)
        mask -- (b,n)
        returns -- (b,n,c,n)
        check out original paper https://arxiv.org/pdf/1611.01734.pdf
        """
        a = self.mlp_left(x).unsqueeze(1)   # (b, 1, l, d)
        b = self.mlp_right(x).unsqueeze(1).transpose(-1, -2)  # (b, 1, d, l)
        t = (a @ self.weight) @ b  # (b, c, l, l)

        if hasattr(self, 'Uleft'):
            t_add = (a @ self.Uleft).transpose(-1, -3)  # (b, c, l, 1)
            t += t_add

        if hasattr(self, 'Uright'):
            t_add = (b.transpose(-1, -2) @ self.Uright).permute(0, 3, 1, 2)  # (b, c, 1, l)
            t += t_add

        # m = mask.to(torch.float32)  # (b, l)
        # m = m.unsqueeze(-1) @ m.unsqueeze(-2)  # (b, l, l)
        # m = torch.triu(m, diagonal=diagonal).unsqueeze(1)  # (b, 1, l, l)
        # t *= m

        d1, d2 = mask.shape
        m = mask.unsqueeze(-2).expand(d1, d2, d2).triu(diagonal).unsqueeze(1)
        t = t.masked_fill(~m, 0.)

        return t


class LSTMSmoother(nn.Module):

    def __init__(self, dim: int, layers: int, dropout: float):
        super(LSTMSmoother, self).__init__()
        self.dim = dim
        self.layers = layers
        self.dropout = dropout
        if self.layers > 0:
            assert dim % 2 == 0, "dimension for bi-lstm must be an even number"
            self.lstm = nn.LSTM(dim, dim//2, layers, dropout=dropout, bidirectional=True)

    def forward(self, batch):
        """
        batch: is [sent_id, word_id, embedding]
        """
        if self.layers == 0:
            return batch
        h0 = torch.zeros([2 * self.layers, len(batch), self.dim // 2], device=batch.device)
        c0 = torch.zeros([2 * self.layers, len(batch), self.dim // 2], device=batch.device)
        batch = batch.transpose(0, 1)
        output, _ = self.lstm(batch, (h0, c0))
        output = output.transpose(0, 1)
        return output


def optimizer_class_by_name(optimizer: str):
    if optimizer == "Adam":
        optim_class = torch.optim.Adam
    elif optimizer == "AdamW":
        optim_class = torch.optim.AdamW
    elif optimizer == "Adagrad":
        optim_class = torch.optim.Adagrad
    elif optimizer == "RMSProp":
        optim_class = torch.optim.RMSprop
    elif optimizer == "SGD":
        optim_class = torch.optim.SGD
    else:
        raise Exception("Unsupported optimizer")
    return optim_class
