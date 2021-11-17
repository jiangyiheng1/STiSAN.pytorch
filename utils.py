import os
import json
from torch.utils.data import Sampler
import torch
import numpy as np
import random
import math


try:
    import cPickle as _pickle
except ImportError:
    import pickle as _pickle


EarthRadius = 6378137
MinLatitude = -85.05112878
MaxLatitude = 85.05112878
MinLongitude = -180
MaxLongitude = 180


def gengerate_padding_mask(sz, ds, device):
    mask = torch.zeros(len(ds), sz, sz)
    for i in range(len(ds)):
        mask[i][sz - ds[i]:, sz - ds[i]:] = torch.ones(ds[i], ds[i])
    mask = mask.to(device)
    return mask


def generate_attn_mask(sz, device):
    mask = (torch.triu(torch.ones(sz, sz))).transpose(0, 1)
    mask = mask.to(device)
    return mask


def generate_key_padding_mask(sz, num_neg, ds, device):
    mask = torch.zeros(len(ds), sz, sz)
    for i in range(len(ds)):
        mask[i][sz - ds[i]:, sz - ds[i]:] = torch.ones(ds[i], ds[i])
    mask = mask.repeat(1, num_neg + 1, num_neg + 1).to(device)
    return mask


def generate_mem_mask(sz, num_neg, device):
    mask = torch.zeros((1 + num_neg) * sz, (1 + num_neg) * sz)
    attend_items = (torch.triu(torch.ones(sz, sz))).transpose(0, 1)
    for i in range(0, (1 + num_neg) * sz, sz):
        mask[i:i + sz, i:i + sz] = attend_items
    mask = mask.to(device)
    return mask


def serialize(obj, path, in_json=False):
    if in_json:
        with open(path, "w") as file:
            json.dump(obj, file, indent=2)
    else:
        with open(path, 'wb') as file:
            _pickle.dump(obj, file)


def unserialize(path):
    suffix = os.path.basename(path).split(".")[-1]
    if suffix == "json":
        with open(path, "r") as file:
            return json.load(file)
    else:
        with open(path, 'rb') as file:
            return _pickle.load(file)


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


def reset_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clip(n, minValue, maxValue):
    return min(max(n, minValue), maxValue)


def map_size(levelOfDetail):
    return 256 << levelOfDetail


def latlon2pxy(latitude, longitude, levelOfDetail):
    latitude = clip(latitude, MinLatitude, MaxLatitude)
    longitude = clip(longitude, MinLongitude, MaxLongitude)

    x = (longitude + 180) / 360
    sinLatitude = math.sin(latitude * math.pi / 180)
    y = 0.5 - math.log((1 + sinLatitude) / (1 - sinLatitude)) / (4 * math.pi)

    mapSize = map_size(levelOfDetail)
    pixelX = int(clip(x * mapSize + 0.5, 0, mapSize - 1))
    pixelY = int(clip(y * mapSize + 0.5, 0, mapSize - 1))
    return pixelX, pixelY


def txy2quadkey(tileX, tileY, levelOfDetail):
    quadKey = []
    for i in range(levelOfDetail, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (tileX & mask) != 0:
            digit += 1
        if (tileY & mask) != 0:
            digit += 2
        quadKey.append(str(digit))

    return ''.join(quadKey)


def pxy2txy(pixelX, pixelY):
    tileX = pixelX // 256
    tileY = pixelY // 256
    return tileX, tileY


def latlon2quadkey(lat,lon,level):
    pixelX, pixelY = latlon2pxy(lat, lon, level)
    tileX, tileY = pxy2txy(pixelX, pixelY)
    return txy2quadkey(tileX, tileY,level)


def fix_length(sequences, max_len, batch_first=True, padding_value=0.):
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


def pad_sequence(sequences, batch_first=True, padding_value=0):
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