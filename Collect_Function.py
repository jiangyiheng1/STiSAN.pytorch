import torch


def fix_length(sequences, max_len, batch_first=True, padding_value=0):
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        if batch_first:
            out_tensor[i, max_len - length:, ...] = tensor
        else:
            out_tensor[max_len - length:, i, ...] = tensor

    return out_tensor


def pad_sequence(sequences, batch_first=False, padding_value=0):
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        if batch_first:
            out_tensor[i, max_len - length:, ...] = tensor
        else:
            out_tensor[max_len - length:, i, ...] = tensor

    return out_tensor


def collect_func_pe_train(batch, data_source, sampler, region_processer, poi2quadkey=None, k=5):
    src, tgt, time_interval_matrix, geo_interval_matrix = zip(*batch)
    time_interval_matrix_ = list()
    geo_interval_matrix_ = list()

    for matrix in time_interval_matrix:
        # (N, T, T)
        time_interval_matrix_.append(matrix)
    for matrix in geo_interval_matrix:
        # (N, T, T)
        geo_interval_matrix_.append(matrix)

    poi, quadkey = [], []
    data_size = []
    tgt_ = []
    tgt_probs_ = []

    for e in src:
        # user, poi, time_idx, region, lat, lon, time, bool
        _, p_, _, r_, _, _, _, b_ = zip(*e)
        data_size.append(len(p_))
        poi.append(torch.tensor(p_))
        r_ = region_processer.numericalize(list(r_))
        quadkey.append(r_)
    poi_ = fix_length(poi, max_len=100, batch_first=True)
    quadkey_ = fix_length(quadkey, max_len=100, batch_first=True)

    batch_tgt_quadkeys = []
    for i, seq in enumerate(tgt):
        pos = torch.tensor([[e[1]] for e in seq])
        neg, probs = sampler(seq, k, user=seq[0][0])
        tgt_seq = torch.cat([pos, neg], dim=-1)
        tgt_.append(tgt_seq)
        tgt_quadkeys = []
        for l in range(tgt_seq.size(0)):
            quadkeys = []
            for poi in tgt_seq[l]:
                quadkeys.append(poi2quadkey[poi])
            tgt_quadkeys.append(region_processer.numericalize(quadkeys))
        batch_tgt_quadkeys.append(torch.stack(tgt_quadkeys))
        tgt_probs_.append(probs)

    batch_tgt_quadkeys = fix_length(batch_tgt_quadkeys, max_len=100, batch_first=True)
    batch_tgt_quadkeys = batch_tgt_quadkeys.permute(2, 1, 0, 3).contiguous().view(-1, batch_tgt_quadkeys.size(0),
                                                                                  batch_tgt_quadkeys.size(3))
    batch_tgt_quadkeys = batch_tgt_quadkeys.transpose(0, 1)
    tgt_ = fix_length(tgt_, max_len=100, batch_first=True)
    tgt_probs_ = fix_length(tgt_probs_, max_len=100, batch_first=True, padding_value=1.0)
    tgt_ = tgt_.permute(2, 1, 0).contiguous().view(-1, tgt_.size(0))
    tgt_ = tgt_.t()
    tgt_nov_ = [[not e[-1] for e in seq] for seq in tgt]

    return poi_, quadkey_, time_interval_matrix_, geo_interval_matrix_, tgt_, batch_tgt_quadkeys, tgt_nov_, tgt_probs_, data_size


def collect_func_pe_test(batch, data_source, sampler, region_processer, poi2quadkey=None, k=100):
    src, tgt, time_interval_matrix, geo_interval_matrix = zip(*batch)
    time_interval_matrix_ = list()
    geo_interval_matrix_ = list()

    for matrix in time_interval_matrix:
        # (N, T, T)
        time_interval_matrix_.append(matrix)
    for matrix in geo_interval_matrix:
        # (N, T, T)
        geo_interval_matrix_.append(matrix)

    poi, quadkey = [], []
    data_size = []
    tgt_ = []
    for e in src:
        _, p_, t_, r_, _, _, _, b_ = zip(*e)
        data_size.append(len(p_))
        poi.append(torch.tensor(p_))
        r_ = region_processer.numericalize(list(r_))
        quadkey.append(r_)
    poi_ = fix_length(poi, max_len=100, batch_first=True)
    quadkey_ = fix_length(quadkey, max_len=100, batch_first=True)

    batch_tgt_quadkeys = []
    for i, seq in enumerate(tgt):
        pos = torch.tensor([[e[1]] for e in seq])
        neg, _ = sampler(seq, k, user=seq[0][0])
        tgt_seq = torch.cat([pos, neg], dim=-1)
        # (N, 1 + K)
        tgt_.append(tgt_seq)
        tgt_quadkeys = []
        for l in range(tgt_seq.size(0)):
            quadkeys = []
            for poi in tgt_seq[l]:
                quadkeys.append(poi2quadkey[poi])
            tgt_quadkeys.append(region_processer.numericalize(quadkeys))
        # (N, 1 + K, L)
        batch_tgt_quadkeys.append(torch.stack(tgt_quadkeys))
    # (N, T, 1 + K, L)
    batch_tgt_quadkeys = pad_sequence(batch_tgt_quadkeys, batch_first=True)
    # ((1+k)*T, N, L)
    batch_tgt_quadkeys = batch_tgt_quadkeys.permute(2, 1, 0, 3).contiguous().view(-1, batch_tgt_quadkeys.size(0),
                                                                                  batch_tgt_quadkeys.size(3))
    # (N, (1+k)*T, L)
    batch_tgt_quadkeys = batch_tgt_quadkeys.transpose(0, 1)
    # (N, T, k+1)
    tgt_ = pad_sequence(tgt_, batch_first=True)
    tgt_ = tgt_.permute(2, 1, 0).contiguous().view(-1, tgt_.size(0))
    tgt_ = tgt_.t()

    return poi_, quadkey_, time_interval_matrix_, geo_interval_matrix_, tgt_, batch_tgt_quadkeys, data_size

