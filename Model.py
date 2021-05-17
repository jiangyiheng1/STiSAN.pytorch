from Modules_Attention import *
from Position_Encoder import *

"Dim Expression:-------------------------------------------------------------------------------------------------------"
"N: batch_size"
"T: max_sequence_length"
"L: max_QUADKEY_length"
"K: num_neg_samples"
"d: embedding_dim"
"----------------------------------------------------------------------------------------------------------------------"


class STAPT_PE(nn.Module):
    def __init__(self, n_poi, n_quadkey, emb_dim, n_geo_enc_layer, n_multi_enc_layer, n_dec_layer,
                 len_quadkey, len_seq, dropout):
        super(STAPT_PE, self).__init__()
        "Embedding Layer"
        self.emb_poi = Embeddings(emb_dim, n_poi)
        self.emb_quadkey = Embeddings(emb_dim, n_quadkey)

        "Dimension Settings"
        geo_attn_dim = emb_dim
        geo_ffn_dim = emb_dim * 2
        src_attn_dim = emb_dim * 2
        src_ffn_dim = emb_dim * 4
        tgt_attn_dim = src_attn_dim
        tgt_ffn_dim = src_ffn_dim

        self.norm = LayerNorm(src_attn_dim)

        "GeoEncoder"
        self.geo_pos = Geo_PE(emb_dim, dropout, len_quadkey)
        self.geo_attn = GeoAttention(dropout)
        self.geo_ffn = PositionwiseFeedForward(geo_attn_dim, geo_ffn_dim, dropout)
        self.geo_enc_layer = GeoEncoderLayer(geo_attn_dim, self.geo_attn, self.geo_ffn, dropout)
        self.geo_enc = GeoEncoder(self.geo_enc_layer, n_geo_enc_layer)

        "MultiEncoder"
        self.multi_pos = Src_PE(emb_dim * 2, dropout, len_seq)
        self.multi_attn = IntervalAttention(dropout)
        self.multi_ffn = PositionwiseFeedForward(src_attn_dim, src_ffn_dim, dropout)
        self.multi_enc_layer = MultiEncoderLayer(src_attn_dim, self.multi_attn, self.multi_ffn, dropout)
        self.multi_enc = MultiEncoder(self.multi_enc_layer, n_multi_enc_layer)

        "Decoder"
        self.tgt_attn = SequentialAttention(dropout)
        self.mem_attn = SequentialAttention(dropout)
        self.dec_ffn = PositionwiseFeedForward(tgt_attn_dim, tgt_ffn_dim, dropout)
        self.dec_layer = TgtDecoderLayer(tgt_attn_dim, self.tgt_attn, self.mem_attn, self.dec_ffn, dropout)
        self.dec = TgtDeocder(self.dec_layer, n_dec_layer)

    def forward(self, src_poi, src_quadkey, t_m, g_m, tgt_poi, tgt_quadkey, src_mask=None, tgt_mask=None, mem_mask=None,
                data_size=None):
        # (N, T, d)
        poi_emb = self.emb_poi(src_poi)
        # (N, T, L, d)
        quadkey_emb = self.emb_quadkey(src_quadkey)
        quadkey_emb = self.geo_pos(quadkey_emb)
        # (N, T, d)
        quadkey_emb = self.geo_enc(quadkey_emb, mask=None)
        # (N, T, 2 * d)
        src = torch.cat([poi_emb, quadkey_emb], dim=-1)
        src = self.multi_pos(src)

        # (N, T, 2 * d)
        src = self.multi_enc(src, t_m, g_m, src_mask)

        # (N, (1 + K) * T, d)
        tgt_poi_emb = self.emb_poi(tgt_poi)
        # (N, (1 + K) * T, L, d)
        tgt_quadkey_emb = self.emb_quadkey(tgt_quadkey)
        tgt_quadkey_emb = self.geo_pos(tgt_quadkey_emb)
        # (N, (1 + K) * T, d)
        tgt_quadkey_emb = self.geo_enc(tgt_quadkey_emb)
        # (N, (1 + K) * T, 2 * d)
        tgt = torch.cat([tgt_poi_emb, tgt_quadkey_emb], dim=-1)
        tgt = self.multi_pos(tgt)
        # (N, (1 + K) * T, 2 * d)
        memory = src.repeat(1, tgt.size(1) // src.size(1), 1)
        # (N, (1 + K) * T, 2 * d)
        output = self.dec(tgt, memory, tgt_mask, mem_mask)

        if self.training:
            # (N, (1 + K) * T, 2 * d)
            src = src.repeat(1, tgt.size(1)//src.size(1), 1)
        else:
            # (N, 2 * d)
            src = src[torch.arange(len(data_size)), len(data_size)-1, :]
            # (N, (1 + k) * T, 2 * d)
            src = src.unsqueeze(1).repeat(1, tgt.size(1), 1)

        # (N, (1 + k) * T, 2 * d)
        output = output + src
        output = self.norm(output)
        # (N, (1 + k) * T)
        output = torch.sum(output * tgt, dim=-1)

        return output

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))