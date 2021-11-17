import torch
from attn_modules import *
from modules import *
from pos_encoder import *


class STiSAN(nn.Module):
    def __init__(self, n_poi, n_quadkey, geo_dim, n_geo_layer, emb_dim, n_IAAB, n_TAAD, kt, kg, device, dropout):
        super(STiSAN, self).__init__()
        "Dimension Setting"
        geo_attn_dim = geo_dim
        geo_ffn_dim = geo_attn_dim * 2
        attn_dim = emb_dim + geo_dim
        ffn_dim = attn_dim * 2

        "Embedding"
        self.emb_poi = Embedding(n_poi, emb_dim)
        self.emb_quadkey = Embedding(n_quadkey, geo_dim)

        "Geography Encoder"
        self.geo_pos = Positional_Encoding(geo_dim, dropout, max_len=12)
        self.geo_attn = Geo_Attn(dropout)
        self.geo_ffn = PointWise_FFN(geo_attn_dim, geo_ffn_dim, dropout)
        self.geo_enc_layer = Geo_Encoder_Layer(geo_attn_dim, self.geo_attn, self.geo_ffn, dropout)
        self.geo_encoder = Geo_Encoder(self.geo_enc_layer, n_geo_layer)

        self.device = device

        "Threshold for time interval and geogaphy distance interval"
        self.kt = kt
        self.kg = kg
        self.kt_matrix = torch.tensor(self.kt).to(self.device)
        self.kt_matrix = self.kt_matrix.repeat(100, 100)
        self.kg_matrix = torch.tensor(self.kg).to(self.device)
        self.kg_matrix = self.kg_matrix.repeat(100, 100)

        "Interval Aware Attention Blocks"
        self.seq_pos = Time_Aware_Positional_Encoding(attn_dim, dropout, max_len=100, device=device)
        self.seq_attn = Interval_Attn(dropout)
        self.seq_ffn = PointWise_FFN(attn_dim, ffn_dim, dropout)
        self.IAAB = Interval_Aware_Attention_Block(attn_dim, self.seq_attn, self.seq_ffn, dropout)
        self.seq_encoder = Stacking_IAABs(self.IAAB, n_IAAB)

        "Target Aware Attention Decoder"
        self.tgt_attn = Target_Aware_Attention(dropout)
        self.dec_layer = Target_Aware_Attention_Decoder_Layer(attn_dim, self.tgt_attn, dropout)
        self.decoder = Target_Aware_Attention_Decoder(self.dec_layer, n_TAAD)

        "Layer Normalization"
        self.norm = LayerNorm(attn_dim)

    def forward(self, src_poi, src_quadkey, src_time, tm, gm, padding_mask, attn_mask, tgt_poi, tgt_quadkey, key_padding_mask, mem_mask, data_size):
        "Embedding poi & quakey"
        # (N, T, d)
        src_poi_emb = self.emb_poi(src_poi)
        # (N, T * (1 + k), d)
        tgt_poi_emb = self.emb_poi(tgt_poi)
        # (N, T, L, d)
        src_quadkey_emb = self.emb_quadkey(src_quadkey)
        # (N, T * (1 + k), L, d)
        tgt_quadkey_emb = self.emb_quadkey(tgt_quadkey)

        "Geography Encoder"
        src_quadkey_emb = self.geo_pos(src_quadkey_emb)
        # (N, T, d)
        src_quadkey_emb = self.geo_encoder(src_quadkey_emb)
        tgt_quadkey_emb = self.geo_pos(tgt_quadkey_emb)
        # (N, T * (1 + k), d)
        tgt_quadkey_emb = self.geo_encoder(tgt_quadkey_emb)

        "Generating src & tgt"
        # (N, T, 2d)
        src = torch.cat([src_poi_emb, src_quadkey_emb], dim=-1)
        # (N, T * (1 + k), 2d)
        tgt = torch.cat([tgt_poi_emb, tgt_quadkey_emb], dim=-1)

        "Interval Aware Attention Block"
        # (N, T, 2d)
        src = self.seq_pos(src, src_time, data_size)

        for i in range(src.size(0)):
            bool_mask = torch.gt(tm[i], self.kt_matrix)
            tm[i] = tm[i].masked_fill(bool_mask == True, self.kt)
            bool_mask = torch.gt(gm[i], self.kg_matrix)
            gm[i] = gm[i].masked_fill(bool_mask == True, self.kg)
            tm[i] = tm[i].max() - tm[i]
            gm[i] = gm[i].max() - gm[i]

        # (N, T, T)
        relation_matrix = tm + gm
        # (N, T, 2d)
        src = self.seq_encoder(src, relation_matrix, padding_mask, attn_mask)

        "Target Aware Attention Decoder"
        if self.training:
            # (N, T * (1 + k), 2d)
            src = src.repeat(1, tgt.size(1) // src.size(1), 1)
            # (N, T * (1 + k), 2d)
            output = self.decoder(src, tgt, key_padding_mask, mem_mask)
        else:
            src = src[torch.arange(len(data_size)), -1, :]
            src = src.unsqueeze(1)
            # (N, 101, 2d)
            src = src.repeat(1, tgt.size(1), 1)
            # (N, 101, 2d)
            output = self.decoder(src, tgt, key_padding_mask, mem_mask)
        
        output += src
        output = self.norm(output)
        # (N, 101)
        output = torch.sum(output * tgt, dim=-1)
        return output

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))