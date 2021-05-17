import copy
import torch.nn.functional as F
from Modules_Model import *


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class GeoEncoder(nn.Module):
    def __init__(self, layer, N):
        super(GeoEncoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        x = torch.mean(x, dim=-2)
        return self.norm(x)


class GeoEncoderLayer(nn.Module):
    def __init__(self, size, attn, ffn, dropout):
        super(GeoEncoderLayer, self).__init__()
        self.size = size
        self.sublayer = clones(SublayerConnection(self.size, dropout), N=2)
        self.self_attn = attn
        self.ffn = ffn
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer[1](x, self.ffn)
        return self.dropout(x)


class GeoAttention(nn.Module):
    def __init__(self, dropout):
        super(GeoAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        x = attention(query, key, value, mask=mask, dropout=self.dropout)
        return x


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / d_k
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value)


class MultiEncoder(nn.Module):
    def __init__(self, layer, N):
        super(MultiEncoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, time_interval_matrix, geo_interval_matrix, mask=None):
        for layer in self.layers:
            x = layer(x, time_interval_matrix, geo_interval_matrix, mask)
        return self.norm(x)


class MultiEncoderLayer(nn.Module):
    def __init__(self, size, interval_attn, ffn, dropout):
        super(MultiEncoderLayer, self).__init__()
        self.size = size
        self.sublayer = clones(SublayerConnection(self.size, dropout), 2)
        self.interval_attn = interval_attn
        self.ffn = ffn
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, time_interval_matrix, geo_interval_matrix, mask=None):
        x = self.sublayer[0](x, lambda x: self.interval_attn(x, x, x,
                                                             time_interval_matrix, geo_interval_matrix, mask))
        x = self.sublayer[1](x, self.ffn)
        return self.dropout(x) / x.size(-1)


class SequentialAttention(nn.Module):
    def __init__(self, dropout):
        super(SequentialAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(0)
        x = attention(query, key, value, mask=mask, dropout=self.dropout)
        return x


class IntervalAttention(nn.Module):
    def __init__(self, dropout):
        super(IntervalAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, t_m, g_m, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(0)
        x = interval_attn(query, key, value, t_m, g_m, mask=mask, dropout=self.dropout)
        return x


def interval_attn(query, key, value, t_m, g_m, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / d_k
    t_m = t_m.view(query.size(0), -1)
    max_t, idx = t_m.max(dim=1)
    max_t = max_t.view(query.size(0), 1)
    t_m = (abs(t_m - max_t)).view(query.size(0), query.size(1), query.size(1))
    g_m = g_m.view(query.size(0), -1)
    max_g, idx = g_m.max(dim=1)
    max_g = max_g.view(query.size(0), 1)
    g_m = (abs(g_m - max_g)).view(query.size(0), query.size(1), query.size(1))
    scores = scores + t_m + g_m
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value)




class TgtDeocder(nn.Module):
    def __init__(self, layer, N):
        super(TgtDeocder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class TgtDecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, ffn, dropout):
        super(TgtDecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.ffn = ffn
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, tgt_mask, mem_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, mem_mask))
        x = self.sublayer[2](x, self.ffn)
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_2(self.dropout(F.relu(self.w_1(x))))
        return x