from utils import *


def generate_train_batch(batch, data_source, train_neg_sampler, train_num_neg, max_len, region_processer, poi2quadkey):
    src, tgt, tm, gm = zip(*batch)
    tm_ = []
    gm_ = []
    for m in tm:
        # (N, T, T)
        tm_.append(m)
    for m in gm:
        # (N, T, T)
        gm_.append(m)

    poi, quadkey, timestamp = [], [], []
    data_size = []
    tgt_ = []
    tgt_probs_ = []

    for e in src:
        u_, p_, t_, q_, _, b_ = zip(*e)
        data_size.append(len(u_))
        poi.append(torch.tensor(p_))
        timestamp.append(torch.tensor(t_))
        q_ = region_processer.numericalize(list(q_))
        # (T, L)
        q_.clone().detach().requires_grad_(False)
        quadkey.append(q_)
    # (N, T)
    poi_ = fix_length(poi, max_len)
    time_ = fix_length(timestamp, max_len)
    # (N, T, L)
    quadkey_ = fix_length(quadkey, max_len)
    batch_tgt_quadkeys = []
    for i, seq in enumerate(tgt):
        pos = torch.tensor([[e[1]] for e in seq])
        neg, probs = train_neg_sampler(seq, train_num_neg, user=seq[0][0])
        # (T, k + 1)
        tgt_seq = torch.cat([pos, neg], dim=-1)
        tgt_.append(tgt_seq)
        tgt_quadkeys = []
        for l in range(tgt_seq.size(0)):
            qks = []
            for poi in tgt_seq[l]:
                qks.append(poi2quadkey[poi])
            tgt_quadkeys.append(region_processer.numericalize(qks))
        batch_tgt_quadkeys.append(torch.stack(tgt_quadkeys))
        tgt_probs_.append(probs)
    # (N, T, k + 1, L)
    batch_tgt_quadkeys = fix_length(batch_tgt_quadkeys, max_len)
    # (N, T * (k + 1), L)
    batch_tgt_quadkeys = batch_tgt_quadkeys.permute(2, 1, 0, 3).contiguous().view(-1, batch_tgt_quadkeys.size(0), batch_tgt_quadkeys.size(3)).transpose(0, 1)
    # (N, T, k + 1)
    tgt_ = fix_length(tgt_, max_len)
    # (N, T * (k + 1))
    tgt_ = tgt_.permute(2, 1, 0).contiguous().view(-1, tgt_.size(0)).t()
    tgt_probs_ = fix_length(tgt_probs_, max_len, batch_first=True, padding_value=1.0)
    tgt_nov_ = [[not e[-1] for e in seq] for seq in tgt]

    return poi_, time_, quadkey_, tm_, gm_, tgt_, batch_tgt_quadkeys, tgt_probs_, tgt_nov_, data_size


def generate_eval_batch(batch, data_source, eval_neg_sampler, eval_num_neg, max_len, region_processer, poi2quadkey):
    src, tgt, tm, gm = zip(*batch)
    tm_ = []
    gm_ = []
    for m in tm:
        tm_.append(m)
    for m in gm:
        gm_.append(m)

    poi, quadkey, timestamp = [], [], []
    data_size = []
    tgt_ = []

    for e in src:
        u_, p_, t_, q_, _, b_ = zip(*e)
        data_size.append(len(u_))
        poi.append(torch.tensor(p_))
        timestamp.append(torch.tensor(t_))
        q_ = region_processer.numericalize(list(q_))
        q_.clone().detach().requires_grad_(False)
        quadkey.append(q_)
    poi_ = fix_length(poi, max_len)
    time_ = fix_length(timestamp, max_len)
    quadkey_ = fix_length(quadkey, max_len)
    batch_tgt_quadkeys = []
    for i, seq in enumerate(tgt):
        pos = torch.tensor([[e[1]] for e in seq])
        neg, _ = eval_neg_sampler(seq, eval_num_neg, user=seq[0][0])
        tgt_seq = torch.cat([pos, neg], dim=-1)
        tgt_.append(tgt_seq)
        tgt_quadkeys = []
        for l in range(tgt_seq.size(0)):
            qks = []
            for poi in tgt_seq[l]:
                qks.append(poi2quadkey[poi])
            tgt_quadkeys.append(region_processer.numericalize(qks))
        batch_tgt_quadkeys.append(torch.stack(tgt_quadkeys))

    batch_tgt_quadkeys = pad_sequence(batch_tgt_quadkeys)
    batch_tgt_quadkeys = batch_tgt_quadkeys.permute(2, 1, 0, 3).contiguous().view(-1, batch_tgt_quadkeys.size(0), batch_tgt_quadkeys.size(3)).transpose(0, 1)
    tgt_ = pad_sequence(tgt_)
    tgt_ = tgt_.permute(2, 1, 0).contiguous().view(-1, tgt_.size(0)).t()

    return poi_, time_, quadkey_, tm_, gm_, tgt_, batch_tgt_quadkeys, data_size