import math
import torch
import torch.nn as nn
from torch.autograd import Variable


class Time_Aware_Positional_Encoding(nn.Module):
    def __init__(self, dim, dropout, max_len, device):
        super(Time_Aware_Positional_Encoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        self.max_len = max_len

    def forward(self, x, time, data_size):
        tape = torch.zeros(x.size(0), x.size(1), x.size(2))
        position = torch.zeros(time.size(0), time.size(1))
        interval = torch.zeros(time.size(0), time.size(1))
        for i in range(len(data_size)):
            length = data_size[i]
            start_idx = self.max_len - length
            position[i][start_idx] = 1.
            mean = 0.
            for j in range(start_idx + 1, self.max_len):
                interval[i][j] = time[i][j] - time[i][j - 1]
                mean += interval[i][j]
            if data_size[i] > 1:
                mean /= (data_size[i] - 1)
            interval[i] /= mean
            for k in range(start_idx + 1, self.max_len):
                position[i][k] = position[i][k - 1] + interval[i][k] + 1
        for i in range(len(data_size)):
            tape[i][:, 0::2] = torch.sin(position[i].unsqueeze(1) * self.div_term)
            tape[i][:, 1::2] = torch.cos(position[i].unsqueeze(1) * self.div_term)
        tape = tape.to(self.device)
        x += Variable(tape, requires_grad=False)
        return self.dropout(x)