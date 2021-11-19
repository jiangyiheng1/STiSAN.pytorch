import copy
import math
from modules import *


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PointWise_FFN(nn.Module):
    def __init__(self, d_attn, d_ffn, dropout):
        super(PointWise_FFN, self).__init__()
        self.w_1 = nn.Linear(d_attn, d_ffn)
        self.w_2 = nn.Linear(d_ffn, d_attn)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_2(self.dropout(F.relu(self.w_1(x))))

        return x


class Geo_Attn(nn.Module):
    def __init__(self, dropout):
        super(Geo_Attn, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, value, key):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        p_attns = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            p_attns = self.dropout(p_attns)

        return torch.matmul(p_attns, value)


class Geo_Encoder_Layer(nn.Module):
    def __init__(self, size, attn, ffn, dropout):
        super(Geo_Encoder_Layer, self).__init__()
        self.size = size
        self.sublayer = clones(SubLayerConnection(self.size, dropout), N=2)
        self.attn = attn
        self.ffn = ffn
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.sublayer[0](x, lambda x: self.attn(x, x, x))
        x = self.sublayer[1](x, self.ffn)

        return x


class Geo_Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Geo_Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = torch.mean(x, dim=-2)

        return self.norm(x)


class Interval_Attn(nn.Module):
    def __init__(self, dropout):
        super(Interval_Attn, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, relation_matrix, padding_mask, attn_mask):
        d_k = query.size(-1)
        mask = attn_mask.unsqueeze(0)
        relation_matrix = relation_matrix.masked_fill(mask == 0.0, -1e9)
        relation_matrix = F.softmax(relation_matrix, dim=-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        scores += relation_matrix
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask == 0.0, -1e9)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            scores = scores.masked_fill(attn_mask == 0.0, -1e9)
        p_attns = F.softmax(scores, dim=-1)
        p_attns = self.dropout(p_attns)

        return torch.matmul(p_attns, value)



class Interval_Aware_Attention_Block(nn.Module):
    def __init__(self, size, attn, ffn, dropout):
        super(Interval_Aware_Attention_Block, self).__init__()
        self.size = size
        self.sublayer = clones(SubLayerConnection(self.size, dropout), 2)
        self.attn = attn
        self.ffn = ffn
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, relation_matrix, padding_mask, attn_mask):
        x = self.sublayer[0](x, lambda x: self.attn(x, x, x, relation_matrix, padding_mask, attn_mask))
        x = self.sublayer[1](x, self.ffn)
        return x


class Stacking_IAABs(nn.Module):
    def __init__(self, block, N):
        super(Stacking_IAABs, self).__init__()
        self.blocks = clones(block, N)
        self.norm = LayerNorm(block.size)

    def forward(self, x, relation_matrix, padding_mask, attn_mask):
        for block in self.blocks:
            x = block(x, relation_matrix, padding_mask, attn_mask)
        return self.norm(x)


class Target_Aware_Attention(nn.Module):
    def __init__(self, dropout):
        super(Target_Aware_Attention, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, key_padding_mask, mem_mask):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask == 0.0, -1e9)
        if mem_mask is not None:
            mem_mask = mem_mask.unsqueeze(0)
            scores = scores.masked_fill(mem_mask == 0.0, -1e9)
        p_attns = F.softmax(scores, dim=-1)
        return torch.matmul(p_attns, value)


class Target_Aware_Attention_Decoder_Layer(nn.Module):
    def __init__(self, size, attn, dropout):
        super(Target_Aware_Attention_Decoder_Layer, self).__init__()
        self.size = size
        self.sublayer = clones(SubLayerConnection(self.size, dropout), 1)
        self.attn = attn

    def forward(self, src, tgt, mem_mask, key_padding_mask):
        x = self.sublayer[0](src, lambda src: self.attn(tgt, src, src, key_padding_mask, mem_mask))
        return x


class Target_Aware_Attention_Decoder(nn.Module):
    def __init__(self, layer, N=1):
        super(Target_Aware_Attention_Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, src, tgt, key_padding_mask, mem_mask):
        for layer in self.layers:
            x = layer(src, tgt, key_padding_mask, mem_mask)
        return x
