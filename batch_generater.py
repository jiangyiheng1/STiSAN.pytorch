import torch
from utils import fix_length
from einops import rearrange


def cf_train_quadkey(batch, data_source, max_len, sampler, quadkey_processor, loc2quadkey, num_neg):
    src_seq, trg_seq, t_mat, g_mat = zip(*batch)
    t_mat_ = torch.stack(t_mat)
    g_mat_ = torch.stack(g_mat)
    src_locs_, src_quadkeys_, src_times_ = [], [], []
    data_size = []
    for e in src_seq:
        u_, l_, q_, t_, _ = zip(*e)
        data_size.append(len(u_))
        src_locs_.append(torch.tensor(l_))
        q_ = quadkey_processor.numericalize(list(q_))
        src_quadkeys_.append(q_)
        src_times_.append(torch.tensor(t_))
    src_locs_ = fix_length(src_locs_, 1, max_len, 'train src seq')
    src_quadkeys_ = fix_length(src_quadkeys_, 2, max_len, 'train src seq')
    src_times_ = fix_length(src_times_, 1, max_len, 'train src seq')

    trg_locs_ = []
    trg_quadkeys_ = []
    for i, seq in enumerate(trg_seq):
        pos = torch.tensor([[e[1]] for e in seq])
        neg = sampler(seq, num_neg, user=seq[0][0])
        pos_neg_locs = torch.cat([pos, neg], dim=-1)
        trg_locs_.append(pos_neg_locs)
        pos_neg_quadkey = []
        for l in range(pos_neg_locs.size(0)):
            q_key = []
            for loc_idx in pos_neg_locs[l]:
                q_key.append(loc2quadkey[loc_idx])
            pos_neg_quadkey.append(quadkey_processor.numericalize(q_key))
        trg_quadkeys_.append(torch.stack(pos_neg_quadkey))

    trg_locs_ = fix_length(trg_locs_, n_axies=2, max_len=max_len, dtype='train trg seq')
    trg_locs_ = rearrange(rearrange(trg_locs_, 'b n k -> k n b').contiguous(), 'k n b -> b (k n)')
    trg_quadkeys_ = fix_length(trg_quadkeys_, n_axies=3, max_len=max_len, dtype='train trg seq')
    trg_quadkeys_ = rearrange(rearrange(trg_quadkeys_, 'b n k l -> k n b l').contiguous(), 'k n b l -> b (k n) l')

    return src_locs_, src_quadkeys_, src_times_, t_mat_, g_mat_, trg_locs_, trg_quadkeys_, data_size

def cf_eval_quadkey(batch, data_source, max_len, sampler, quadkey_processor, loc2quadkey, num_neg):
    src_seq, trg_seq, t_mat, g_mat = zip(*batch)
    t_mat_ = torch.stack(t_mat)
    g_mat_ = torch.stack(g_mat)
    src_locs_, src_quadkeys_, src_times_ = [], [], []
    data_size = []
    for e in src_seq:
        u_, l_, q_, t_, _ = zip(*e)
        data_size.append(len(u_))
        src_locs_.append(torch.tensor(l_))
        q_ = quadkey_processor.numericalize(list(q_))
        src_quadkeys_.append(q_)
        src_times_.append(torch.tensor(t_))
    src_locs_ = fix_length(src_locs_, 1, max_len, 'eval src seq')
    src_quadkeys_ = fix_length(src_quadkeys_, 2, max_len, 'eval src seq')
    src_times_ = fix_length(src_times_, 1, max_len, 'eval src seq')

    trg_locs_ = []
    trg_quadkeys_ = []
    for i, seq in enumerate(trg_seq):
        pos = torch.tensor([[e[1]] for e in seq])
        neg = sampler(seq, num_neg, user=seq[0][0])
        pos_neg_locs = torch.cat([pos, neg], dim=-1)
        trg_locs_.append(pos_neg_locs)
        pos_neg_quadkey = []
        for l in range(pos_neg_locs.size(0)):
            q_key = []
            for loc_idx in pos_neg_locs[l]:
                q_key.append(loc2quadkey[loc_idx])
            pos_neg_quadkey.append(quadkey_processor.numericalize(q_key))
        trg_quadkeys_.append(torch.stack(pos_neg_quadkey))

    trg_locs_ = fix_length(trg_locs_, n_axies=2, max_len=max_len, dtype='eval trg loc')
    trg_locs_ = rearrange(rearrange(trg_locs_, 'b n k -> k n b').contiguous(), 'k n b -> b (k n)')
    trg_quadkeys_ = fix_length(trg_quadkeys_, n_axies=3, max_len=max_len, dtype='eval trg loc')
    trg_quadkeys_ = rearrange(rearrange(trg_quadkeys_, 'b n k l -> k n b l').contiguous(), 'k n b l -> b (k n) l')

    return src_locs_, src_quadkeys_, src_times_, t_mat_, g_mat_, trg_locs_, trg_quadkeys_, data_size