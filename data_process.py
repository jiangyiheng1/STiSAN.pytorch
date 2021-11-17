import torch
import copy
import math
import os
import joblib
import time
import numpy as np
from tqdm import tqdm
from utils import latlon2quadkey, serialize, unserialize
from torch.utils.data import Dataset
from collections import defaultdict
from dateutil import parser as t_parser
from nltk import ngrams
from torchtext.data import Field
from geopy.distance import geodesic

LOD = 17


class Data_Process(Dataset):
    def __init__(self, filename):
        self.poi2idx = {'<pad>': 0}
        self.poi2gps = {'<pad>': (0.0, 0.0)}
        self.idx2poi = {0: '<pad>'}
        self.idx2gps = {0: (0.0, 0.0)}
        self.poi2count = {}
        self.n_poi = 1
        self.build_poi_vocab(filename)
        self.user_seq, self.user2idx, self.region2idx, self.n_user, self.n_region, self.region2poi = self.processing(filename)

    def build_poi_vocab(self, filename, min_freq=10):
        for line in open(filename):
            line = line.strip().split('\t')
            if len(line) < 5:
                continue
            poi = line[4]
            coordinate = line[2], line[3]
            self.add_poi(poi, coordinate)
        if min_freq > 0:
            self.n_poi = 1
            self.poi2idx = {'<pad>': 0}
            self.idx2poi = {0: '<pad>'}
            self.idx2gps = {0: (0.0, 0.0)}
            for poi in self.poi2count:
                if self.poi2count[poi] >= min_freq:
                    self.add_poi(poi, self.poi2gps[poi])
        self.poiidx2freq = np.zeros(self.n_poi - 1, dtype=np.int32)
        for idx, poi in self.idx2poi.items():
            if idx != 0:
                self.poiidx2freq[idx - 1] = self.poi2count[poi]

    def add_poi(self, poi, coordinate):
        if poi not in self.poi2idx:
            self.poi2idx[poi] = self.n_poi
            self.poi2gps[poi] = coordinate
            self.idx2poi[self.n_poi] = poi
            self.idx2gps[self.n_poi] = coordinate
            if poi not in self.poi2count:
                self.poi2count[poi] = 1
            self.n_poi += 1
        else:
            self.poi2count[poi] += 1

    def processing(self, filename, min_freq=20):
        user_seq = {}
        user_seq_array = []
        quadkey2idx = {}
        idx2quadkey = {}
        quadkeyidx2poi = defaultdict(set)
        n_quadkey = 1
        user2idx = {}
        n_user = 1

        for line in open(filename):
            line = line.strip().split('\t')
            if len(line) < 5:
                continue
            user, raw_time, lat, lon, poi = line
            if poi not in self.poi2idx:
                continue
            raw_time = time.strptime(raw_time, '%Y-%m-%dT%H:%M:%SZ')
            timestamp = time.mktime(raw_time)
            poi_idx = self.poi2idx[poi]
            quadkey = latlon2quadkey(float(lat), float(lon), LOD)
            if quadkey not in quadkey2idx:
                quadkey2idx[quadkey] = n_quadkey
                idx2quadkey[n_quadkey] = quadkey
                n_quadkey += 1
            quadkey_idx = quadkey2idx[quadkey]
            quadkeyidx2poi[quadkey_idx].add(poi_idx)
            if user not in user_seq:
                user_seq[user] = []
            user_seq[user].append([poi_idx, timestamp, quadkey, (float(lat), float(lon))])

        for user, seq in user_seq.items():
            if len(seq) >= min_freq:
                user2idx[user] = n_user
                user_idx = n_user
                seq_new = []
                tmp_set = set()
                cnt = 0
                for poi, timestamp, quadkey, gps in sorted(seq, key=lambda e: e[1]):
                    if poi in tmp_set:
                        seq_new.append((user_idx, poi, timestamp, quadkey, gps, True))
                    else:
                        seq_new.append((user_idx, poi, timestamp, quadkey, gps, False))
                        tmp_set.add(poi)
                        cnt += 1
                if cnt > min_freq / 2:
                    n_user += 1
                    user_seq_array.append(seq_new)

        all_quadkeys = []
        for u in range(len(user_seq_array)):
            seq = user_seq_array[u]
            for i in range(len(seq)):
                check_in = seq[i]
                user = check_in[0]
                poi = check_in[1]
                timestamp = check_in[2]
                quadkey = check_in[3]
                gps = check_in[4]
                bool = check_in[5]

                quadkey_n_gram = ' '.join([''.join(x) for x in ngrams(quadkey, 6)])
                quadkey_n_gram = quadkey_n_gram.split()
                all_quadkeys.append(quadkey_n_gram)
                user_seq_array[u][i] = (user, poi, timestamp, quadkey_n_gram, gps, bool)

        self.poi2quadkey = []
        for poi in range(self.n_poi):
            lat, lon = self.idx2gps[poi]
            quadkey = latlon2quadkey(float(lat), float(lon), LOD)
            quadkey_n_gram = ' '.join([''.join(x) for x in ngrams(quadkey, 6)])
            quadkey_n_gram = quadkey_n_gram.split()
            self.poi2quadkey.append(quadkey_n_gram)
            all_quadkeys.append(quadkey_n_gram)

        self.QUADKEY = Field(
                sequential=True,
                use_vocab=True,
                batch_first=True,
                unk_token=None,
                preprocessing=str.split
        )
        self.QUADKEY.build_vocab(all_quadkeys)

        return user_seq_array, user2idx, quadkey2idx, n_user, n_quadkey, quadkeyidx2poi

    def __len__(self):
        return len(self.user_seq)

    def __getitem__(self, idx):
        return self.user_seq[idx]

    def compute_relation(self, max_len=100):
        train_ = copy.copy(self)
        eval_ = copy.copy(self)
        train_seq = []
        eval_seq = []
        for u in tqdm(range(len(self))):
            seq = self[u]
            for i in reversed(range(len(seq))):
                if not seq[i][-1]:
                    break
            n = math.floor((i + max_len - 1) // max_len)
            for b in range(n):
                if (i - b * max_len) > max_len:
                    target = seq[(i - (b + 1) * max_len): (i - b * max_len)]
                    source = seq[(i - (b + 1) * max_len - 1): (i - b * max_len - 1)]
                    t = [cr[2] for cr in source]
                    g = [cr[4] for cr in source]
                    tm = torch.zeros(max_len, max_len)
                    gm = torch.zeros(max_len, max_len)
                    length = len(source)
                    idx = max_len - length
                    for x in range(idx, max_len):
                        for y in range(x, max_len):
                            tm[x][y] = t[y] - t[x]
                            gm[x][y] = math.floor(geodesic(g[y], g[x]).m)
                            tm[y][x] = tm[x][y]
                            gm[y][x] = gm[x][y]
                    train_seq.append((source, target, tm, gm))
                else:
                    if (i - b * max_len - 1) <= 10:
                        break
                    target = seq[1: (i - b * max_len)]
                    source = seq[0: (i - b * max_len - 1)]
                    t = [cr[2] for cr in source]
                    g = [cr[4] for cr in source]
                    tm = torch.zeros(max_len, max_len)
                    gm = torch.zeros(max_len, max_len)
                    length = len(source)
                    idx = max_len - length
                    for x in range(idx, max_len):
                        for y in range(x, max_len):
                            tm[x][y] = t[y - idx] - t[x - idx]
                            gm[x][y] = math.floor(geodesic(g[y - idx], g[x - idx]).m)
                            tm[y][x] = tm[x][y]
                            gm[y][x] = gm[x][y]
                    train_seq.append((source, target, tm, gm))
                    break

            target = seq[i:i+1]
            source = seq[max(0, -max_len+i):i]
            t = [cr[2] for cr in source]
            g = [cr[4] for cr in source]
            tm = torch.zeros(max_len, max_len)
            gm = torch.zeros(max_len, max_len)
            length = len(source)
            idx = max_len - length
            for x in range(idx, max_len):
                for y in range(x, max_len):
                    tm[x][y] = t[y - idx] - t[x - idx]
                    gm[x][y] = math.floor(geodesic(g[y - idx], g[x - idx]).m)
                    tm[y][x] = tm[x][y]
                    gm[y][x] = gm[x][y]
            eval_seq.append((source, target, tm, gm))

        train_.user_seq = train_seq
        eval_.user_seq = sorted(eval_seq, key=lambda e: len(e[0]))

        return train_, eval_


if __name__ == "__main__":
    data_raw = ' '
    data_clean = ' '

    if os.path.exists(data_clean):
        dataset = unserialize(data_clean)
    else:
        dataset = Data_Process(data_raw)
        serialize(dataset, data_clean)

    count = 0
    length = []
    for seq in dataset.user_seq:
        count += len(seq)
        length.append(len(seq))
    print("#check-ins:", count)
    print("#users:", dataset.n_user - 1)
    print("#POIs:", dataset.n_poi - 1)
    print("#median seq len:", np.median(np.array(length)))

    start_time = time.time()
    print("Spatio-Temporal Relation Matrices are constructing... \n"
          "The train dataset and evaluation dataset will be divided at the same time.")
    train, eval = dataset.compute_relation(max_len=100)
    joblib.dump(train, ' ')
    joblib.dump(eval, ' ')
    print("Finished with", math.floor((time.time() - start_time) / 3600), "hours.")