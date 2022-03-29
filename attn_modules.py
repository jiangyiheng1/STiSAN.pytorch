import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import math


def clones(module, num_sub_layer):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_sub_layer)])


class Embedding(nn.Module):
    def __init__(self, vocab_size, num_units, zeros_pad=True, scale=False):
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.num_units = num_units
        self.zeros_pad = zeros_pad
        self.scale = scale
        self.lookup_table = nn.Parameter(torch.Tensor(vocab_size, num_units))
        nn.init.xavier_normal_(self.lookup_table.data)
        if self.zeros_pad:
            self.lookup_table.data[0, :].fill_(0)

    def forward(self, inputs):
        if self.zeros_pad:
            self.padding_idx = 0
        else:
            self.padding_idx = -1
        outputs = F.embedding(inputs, self.lookup_table, self.padding_idx, None, 2, False, False)
        if self.scale:
            outputs = outputs * (self.num_units ** 0.5)

        return outputs


class SubLayerConnect(nn.Module):
    def __init__(self, features):
        super(SubLayerConnect, self).__init__()
        self.norm = nn.LayerNorm(features)

    def forward(self, x, sublayer):
        # (*, d)
        return x + sublayer(self.norm(x))


class FFN(nn.Module):
    def __init__(self, features, exp_factor, dropout):
        super(FFN, self).__init__()
        self.w_1 = nn.Linear(features, exp_factor * features)
        self.act = nn.ReLU()
        self.w_2 = nn.Linear(exp_factor * features, features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        return x


class SA(nn.Module):
    def __init__(self, dropout):
        super(SA, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask, pad_mask):
        scale_term = math.sqrt(x.size(-1))
        scores = torch.matmul(x, x.transpose(-2, -1)) / scale_term
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask == 0.0, -1e9)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            scores = scores.masked_fill(attn_mask == 0.0, -1e9)
        prob = F.softmax(scores, dim=-1)
        prob = self.dropout(prob)
        return torch.matmul(prob, x)


class GeoEncoderLayer(nn.Module):
    def __init__(self, features, exp_factor, dropout):
        super(GeoEncoderLayer, self).__init__()
        self.sa_layer = SA(dropout)
        self.ffn_layer = FFN(features, exp_factor, dropout)
        self.sublayer = clones(SubLayerConnect(features), 2)

    def forward(self, x):
        # (b ,n, l, d)
        x = self.sublayer[0](x, lambda x:self.sa_layer(x, None, None))
        x = self.sublayer[1](x, self.ffn_layer)
        return x


class GeoEncoder(nn.Module):
    def __init__(self, features, layer, depth):
        super(GeoEncoder, self).__init__()
        self.layers = clones(layer, depth)
        self.norm = nn.LayerNorm(features)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=-2)
        return self.norm(x)


class InrAwaSA(nn.Module):
    def __init__(self, dropout):
        super(InrAwaSA, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, r_mat, attn_mask, pad_mask):
        scale_term = math.sqrt(x.size(-1))
        scores = torch.matmul(x, x.transpose(-2, -1)) / scale_term
        mask = pad_mask
        r_mat = r_mat.masked_fill(mask == 0.0, -1e9)
        mask = attn_mask.unsqueeze(0)
        r_mat = r_mat.masked_fill(mask == 0.0, -1e9)
        r_mat = F.softmax(r_mat, dim=-1)
        scores += r_mat
        if pad_mask is not None:
            scores.masked_fill(pad_mask == 0.0, -1e9)
        if attn_mask is not None:
            attn_mask.unsqueeze(0)
            scores.masked_fill(attn_mask == 0.0, -1e9)
        prob = F.softmax(scores, dim=-1)
        prob = self.dropout(prob)
        return torch.matmul(prob, x)


class InrEncoderLayer(nn.Module):
    def __init__(self, features, exp_factor, dropout):
        super(InrEncoderLayer, self).__init__()
        self.inr_sa_layer = InrAwaSA(dropout)
        self.ffn_layer = FFN(features, exp_factor, dropout)
        self.sublayer = clones(SubLayerConnect(features), 2)

    def forward(self, x, r_mat, attn_mask, pad_mask):
        x = self.sublayer[0](x, lambda x:self.inr_sa_layer(x, r_mat, attn_mask, pad_mask))
        x = self.sublayer[1](x, self.ffn_layer)
        return x


class InrEncoder(nn.Module):
    def __init__(self, features, layer, depth):
        super(InrEncoder, self).__init__()
        self.layers = clones(layer, depth)
        self.norm = nn.LayerNorm(features)

    def forward(self, x, r_mat, attn_mask, pad_mask):
        for layer in self.layers:
            x = layer(x, r_mat, attn_mask, pad_mask)
        return self.norm(x)


class TrgAwaDecoder(nn.Module):
    def __init__(self, features, dropout):
        super(TrgAwaDecoder, self).__init__()
        self.norm = nn.LayerNorm(features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, trg, key_pad_mask, mem_mask):
        res = src
        src = self.norm(src)
        trg = self.norm(trg)
        scale_term = src.size(-1)
        scores = torch.matmul(trg, src.transpose(-2, -1)) / scale_term
        if key_pad_mask is not None:
            scores = scores.masked_fill(key_pad_mask == 0.0, -1e9)
        if mem_mask is not None:
            mem_mask.unsqueeze(0)
            scores = scores.masked_fill(mem_mask == 0.0, -1e9)
        prob = F.softmax(scores, dim=-1)
        prob = self.dropout(prob)
        x = res + torch.matmul(prob, src)
        return self.norm(x)