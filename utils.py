import os
import json
import torch
import numpy as np
from math import radians, cos, sin, asin, sqrt, floor
from torch.utils.data import Sampler
from torch.nn.utils.rnn import pad_sequence
from einops import reduce, repeat

try:
    import cPickle as _pickle
except ImportError:
    import pickle as _pickle


def serialize(obj, path, in_json=False):
    if in_json:
        with open(path, "w") as file:
            json.dump(obj, file, indent=2)
    else:
        with open(path, 'wb') as file:
            _pickle.dump(obj, file)


def get_attn_mask(max_len, device):
    mask = (torch.triu(torch.ones(max_len, max_len))).transpose(0, 1)
    return mask.to(device)

def get_pad_mask(data_size, max_len, device):
    mask = torch.zeros([len(data_size), max_len, max_len])
    for i in range(len(data_size)):
        mask[i][0: data_size[i], 0: data_size[i]] = torch.ones(data_size[i], data_size[i])
    return mask.to(device)

def get_key_pad_mask(data_size, max_len, num_neg, device):
    mask = torch.zeros([len(data_size), max_len, max_len])
    for i in range(len(data_size)):
        mask[i][0: data_size[i], 0: data_size[i]] = torch.ones(data_size[i], data_size[i])
    mask = mask.repeat(1, num_neg + 1, num_neg + 1).to(device)
    return mask

def get_mem_mask(max_len, num_neg, device):
    mask = torch.zeros((1 + num_neg) * max_len, (1 + num_neg) * max_len)
    attend_items = (torch.triu(torch.ones(max_len, max_len))).transpose(0, 1)
    for i in range(0, (1 + num_neg) * max_len, max_len):
        mask[i:i + max_len, i:i + max_len] = attend_items
    mask = mask.to(device)
    return mask

def unserialize(path):
    suffix = os.path.basename(path).split(".")[-1]
    if suffix == "json":
        with open(path, "r") as file:
            return json.load(file)
    else:
        with open(path, 'rb') as file:
            return _pickle.load(file)


class LadderSampler(Sampler):
    def __init__(self, data_source, batch_sz, fix_order=False):
        super(LadderSampler, self).__init__(data_source)
        self.data = [len(e[0]) for e in data_source]
        self.batch_size = batch_sz * 100
        self.fix_order = fix_order

    def __iter__(self):
        if self.fix_order:
            d = zip(self.data, np.arange(len(self.data)), np.arange(len(self.data)))
        else:
            d = zip(self.data, np.random.permutation(len(self.data)), np.arange(len(self.data)))
        d = sorted(d, key=lambda e: (e[1] // self.batch_size, e[0]), reverse=True)
        return iter(e[2] for e in d)

    def __len__(self):
        return len(self.data)


def fix_length(sequences, n_axies, max_len, dtype='source'):
    if dtype != 'eval trg loc':
        padding_term = torch.zeros_like(sequences[0])
        length = padding_term.size(0)
        # (l, any) -> (1, any) -> (max_len any)
        if n_axies == 1:
            padding_term = reduce(padding_term, '(h l) -> h', 'max', l=length)
            padding_term = repeat(padding_term, 'h -> (repeat h)', repeat=max_len)
        elif n_axies == 2:
            padding_term = reduce(padding_term, '(h l) any -> h any', 'max', l=length)
            padding_term = repeat(padding_term, 'h any -> (repeat h) any', repeat=max_len)
        else:
            padding_term = reduce(padding_term, '(h l) any_1 any_2 -> h any_1 any_2', 'max', l=length)
            padding_term = repeat(padding_term, 'h any_1 any_2 -> (repeat h) any_1 any_2', repeat=max_len)

        sequences.append(padding_term)
        tensor = pad_sequence(sequences, True)
        return tensor[:-1]
    else:
        tensor = pad_sequence(sequences, True)
        return tensor


def get_visited_locs(dataset):
    user_visited_locs = {}
    for u in range(len(dataset.user_seq)):
        seq = dataset.user_seq[u]
        user = seq[0][0]
        user_visited_locs[user] = set()
        for i in reversed(range(len(seq))):
            if not seq[i][-1]:
                break
        user_visited_locs[user].add(seq[i][1])
        seq = seq[:i]
        for check_in in seq:
            user_visited_locs[user].add(check_in[1])
    return user_visited_locs


def build_st_matrix(dataset, sequence, max_len):
    seq_len = len(sequence)
    temporal_matrix = torch.zeros(max_len, max_len)
    spatial_matrix = torch.zeros(max_len, max_len)
    seq_loc = [r[1] for r in sequence]
    seq_time = [r[3] for r in sequence]
    for x in range(seq_len):
        for y in range(x):
            temporal_matrix[x, y] = floor((seq_time[x] - seq_time[y]) / 3600)
            spatial_matrix[x, y] = haversine(dataset.idx2gps[seq_loc[x]], dataset.idx2gps[seq_loc[y]])
    return temporal_matrix, spatial_matrix

def haversine(point_1, point_2):
    lat_1, lng_1 = point_1
    lat_2, lng_2 = point_2
    lat_1, lng_1, lat_2, lng_2 = map(radians, [lat_1, lng_1, lat_2, lng_2])

    d_lon = lng_2 - lng_1
    d_lat = lat_2 - lat_1
    a = sin(d_lat / 2) ** 2 + cos(lat_1) * cos(lat_2) * sin(d_lon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return floor(c * r)

def extract_sub_matrix(start_idx, end_idx, seq_len, max_len, temporal_matrix, spatial_matrix):
    sub_t_mat = torch.zeros(max_len, max_len)
    sub_s_mat = torch.zeros(max_len, max_len)
    if seq_len == max_len:
        sub_t_mat = temporal_matrix[start_idx: end_idx, start_idx: end_idx]
        sub_s_mat = spatial_matrix[start_idx: end_idx, start_idx: end_idx]
    else:
        sub_t_mat[start_idx - start_idx: end_idx - start_idx, start_idx - start_idx: end_idx - start_idx] = temporal_matrix[start_idx: end_idx, start_idx: end_idx]
        sub_s_mat[start_idx - start_idx: end_idx - start_idx, start_idx - start_idx: end_idx - start_idx] = spatial_matrix[start_idx: end_idx, start_idx: end_idx]
    return sub_t_mat, sub_s_mat