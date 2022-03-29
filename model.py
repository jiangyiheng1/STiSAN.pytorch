import torch

from time_aware_pe import TAPE
from attn_modules import *


class STiSAN(nn.Module):
    def __init__(self, n_loc, n_quadkey, features, exp_factor, k_t, k_g, depth, dropout):
        super(STiSAN, self).__init__()
        self.emb_loc = Embedding(n_loc, features, True, True)
        self.emb_quadkey = Embedding(n_quadkey, features, True, True)

        self.geo_encoder_layer = GeoEncoderLayer(features, exp_factor, dropout)
        self.geo_encoder = GeoEncoder(features, self.geo_encoder_layer, depth=2)

        self.tape = TAPE(dropout)

        self.k_t = torch.tensor(k_t)
        self.k_g = torch.tensor(k_g)

        self.inr_awa_attn_layer = InrEncoderLayer(features * 2, exp_factor, dropout)
        self.inr_awa_attn_block = InrEncoder(features * 2, self.inr_awa_attn_layer, depth)

        self.trg_awa_attn_decoder = TrgAwaDecoder(features * 2, dropout)

    def forward(self, src_loc, src_quadkey, src_time, t_mat, g_mat, pad_mask, attn_mask,
                trg_loc, trg_quadkey, key_pad_mask, mem_mask, ds):
        # (b, n, d)
        src_loc_emb = self.emb_loc(src_loc)
        # (b, n * (1 + k), d)
        trg_loc_emb = self.emb_loc(trg_loc)

        # (b, n, l, d)
        src_quadkey_emb = self.emb_quadkey(src_quadkey)
        # (b, n, d)
        src_quadkey_emb = self.geo_encoder(src_quadkey_emb)
        # (b, n * (1 + k), d)
        trg_quadkey_emb = self.emb_quadkey(trg_quadkey)
        # (b, n * (1 + k), d)
        trg_quadkey_emb = self.geo_encoder(trg_quadkey_emb)
        # (b, n, 2 * d)
        src = torch.cat([src_loc_emb, src_quadkey_emb], dim=-1)
        # (b, n * (1 + k), 2 * d)
        trg = torch.cat([trg_loc_emb, trg_quadkey_emb], dim=-1)
        # (b, n, 2 * d)
        src = self.tape(src, src_time, ds)
        # (b, n, n)
        for i in range(src.size(0)):
            mask = torch.gt(t_mat[i], self.k_t)
            t_mat[i] = t_mat[i].masked_fill(mask == True, self.k_t)
            t_mat[i] = t_mat[i].max() - t_mat[i]
            mask = torch.gt(t_mat[i], self.k_t)
            g_mat[i] = g_mat[i].masked_fill(mask == True, self.k_g)
            g_mat[i] = g_mat[i].max() - g_mat[i]
        # (b, n, n)
        r_mat = t_mat + g_mat
        # (b, n, 2 * d)
        src = self.inr_awa_attn_block(src, r_mat, attn_mask, pad_mask)

        if self.training:
            # (b, n * (1 + k), 2 * d)
            src = src.repeat(1, trg.size(1)//src.size(1), 1)
            # (b, n * (1 + k), 2 * d)
            src = self.trg_awa_attn_decoder(src, trg, key_pad_mask, mem_mask)
        else:
            # (b, 2 * d)
            src = src[torch.arange(len(ds)), torch.tensor(ds) - 1, :]
            # (b, 1 + k, 2 * d)
            src = src.unsqueeze(1).repeat(1, trg.size(1), 1)
            src = self.trg_awa_attn_decoder(src, trg, key_pad_mask, mem_mask)
        # (b, 1 + k)
        output = torch.sum(src * trg, dim=-1)
        return output

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))