import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import fix_length


class TAPE(nn.Module):
    def __init__(self, dropout):
        super(TAPE, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, time, data_size):
        b, n, d = x.shape
        # (b, n)
        time_ = torch.clone(time)
        time_[:, 1:] = time[:, :-1]
        mask = [torch.ones(e, dtype=torch.float32) for e in data_size]
        # (b, n)
        mask = fix_length(mask, 1, n, "exclude padding term").to('cuda:0')

        # (b, n)
        interval = time - time_
        interval = interval.masked_fill(mask == 0, 0.0)
        sum_interval = (interval.sum(dim=-1)).reshape(b, -1)
        sum_interval = sum_interval.masked_fill(sum_interval == 0, 1)
        num_interval = (mask.sum(dim=-1) - 1).reshape(b, -1)
        num_interval = num_interval.masked_fill(num_interval == 0, 1)
        avg_interval = sum_interval / num_interval
        interval /= avg_interval

        # (b, n)
        pos = torch.zeros_like(time)
        pos[:, 0] = 1.
        for k in range(1, n):
            pos[:, k] = pos[:, k - 1] + interval[:, k] + 1
        pos = pos.masked_fill(mask == 0, 0.0)
        div_term = torch.exp(torch.arange(0, d, 2) * -(math.log(10000.0) / d)).to('cuda:0')
        # (b, n, d)
        tape = torch.zeros_like(x)
        tape[:, :, 0::2] = torch.sin(pos[:].unsqueeze(-1) * div_term.unsqueeze(0))
        tape[:, :, 1::2] = torch.cos(pos[:].unsqueeze(-1) * div_term.unsqueeze(0))
        x += Variable(tape, requires_grad=False)
        return self.dropout(x)