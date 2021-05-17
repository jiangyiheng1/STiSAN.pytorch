import math
import torch
import torch.nn as nn
from torch.autograd import Variable


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:x.size(-2), :], requires_grad=False)
        return self.dropout(x)


class Time_Aware_Positional_Encoding(nn.Module):
    def __init__(self, batch_size, max_len, d_model, dropout):
        super(Time_Aware_Positional_Encoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe_cos = torch.zeros([batch_size, max_len, d_model])
        self.pe_sin = self.pe_cos
        self.div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)).repeat_interleave(2)

    def forward(self, x, t):
        batch_size = x.size(0)
        max_len = x.size(1)
        tpe = t + torch.arange(0, max_len)
        for i in range(batch_size):
            for j in range(max_len):
                self.pe_cos[i][j] = tpe[i][j]
        # self.pe_cos.to(device='cuda:0')
        self.pe_sin = self.pe_cos
        self.pe_cos = torch.cos(self.div_term * self.pe_cos)
        self.pe_sin = torch.sin(self.div_term * self.pe_sin)
        y = torch.zeros_like(x)
        y[:, :, 0::2] = x[:, :, 1::2] * -1
        y[:, :, 1::2] = x[:, :, 0::2]
        x = x * self.pe_cos + y * self.pe_sin

        return self.dropout(x)

