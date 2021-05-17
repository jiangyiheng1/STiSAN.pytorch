import torch
import copy
import math
import argparse
import os
import joblib
import numpy as np
import time as T
from datetime import datetime
from Utils import latlon2quadkey, serialize, unserialize
from torch.utils.data import Dataset
from collections import defaultdict
from dateutil import parser as t_parser
from nltk import ngrams
from torchtext.data import Field
from geopy.distance import geodesic

LOD=17

class LBSNDataset(Dataset):
    def __init__(self, filename):
        self.poi2idx = {'<pad>': 0}
        self.poi2gps = {'<pad>': (0.0, 0.0)}
        self.idx2poi = {0: '<pad>'}
        self.idx2gps = {0: (0.0, 0.0)}
        self.poi2count = {}
        self.n_poi = 1
        self.TimePeriod = self.build_time_period(filename)
        self.build_vocab(filename)
        self.user_seq, self.user2idx, self.region2idx, self.n_user, self.n_region, self.region2poi = self.filtering_cold_user(filename)
        self.region_stats()

    def build_time_period(self, filename):
        self.time_period = {}
        self.timespan = list()
        r_time = list()
        print("Extracting time stamps...")
        for line in open(filename):
            line = line.strip().split('\t')
            time = line[1]
            time = datetime.strptime(time, '%Y-%m-%dT%H:%M:%SZ')
            time = datetime.time(time)
            time = t_parser.parse(str(time))
            r_time.append(time)
        r_time.sort()
        totalcheckins = len(r_time)
        start_pos = 0
        print("Building probability based time periods...")
        for i in range(12):
            self.time_period[i] = list()
            end_pos = start_pos
            if i == 11:
                self.time_period[i].append(r_time[start_pos].time())
                self.time_period[i].append(r_time[totalcheckins - 1].time())
                self.timespan.append((r_time[totalcheckins - 1] - r_time[start_pos]).seconds)
                print('time period', i, ': ', 'start time:', self.time_period[i][0], 'end time:',
                      self.time_period[i][1], 'time span: ', self.timespan[i], 's')
            else:
                while (end_pos - start_pos + 1) / totalcheckins <= 1/12:
                    end_pos += 1
                self.time_period[i].append(r_time[start_pos].time())
                self.time_period[i].append(r_time[end_pos].time())
                self.timespan.append((r_time[end_pos] - r_time[start_pos]).seconds)
                print('time period', i, ': ', 'start time:', self.time_period[i][0], 'end time:',
                      self.time_period[i][1], 'time span: ', self.timespan[i], 's')
                start_pos = end_pos + 1
        timespan = np.array(self.timespan, dtype=np.int32)
        print("min time span: ", np.min(timespan), 's')
        print("max time span: ", np.max(timespan), 's')
        print("avg time span: ", np.mean(timespan), 's')
        return self.time_period

    def region_stats(self):
        num_reg_pois = []
        for reg in self.region2poi:
            num_reg_pois.append(len(self.region2poi[reg]))
        num_reg_pois = np.array(num_reg_pois, dtype=np.int32)
        print("min #poi/region: {:d}, with {:d} regions".format(np.min(num_reg_pois), np.count_nonzero(num_reg_pois == 1)))
        print("max #poi/region:", np.max(num_reg_pois))
        print("avg #poi/region: {: .4f}".format(np.mean(num_reg_pois)))

    def build_vocab(self, filename, min_freq=10):
        print("Filtering cold POIs (be checked less than 10)...")
        for line in open(filename):
            line = line.strip().split('\t')
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

    def filtering_cold_user(self, filename, min_freq=20):
        user_seq = {}
        user_seq_array = list()
        region2idx = {}
        idx2region = {}
        regionidx2poi = defaultdict(set)
        n_region = 1
        user2idx = {}
        n_user = 1
        print("Building QUADKEYs...")
        for line in open(filename):
            user, time, lat, lon, poi = line.strip().split('\t')
            if poi not in self.poi2idx:
                continue
            time = datetime.strptime(time, '%Y-%m-%dT%H:%M:%SZ')
            date = time.year * 1000000 + time.month * 10000 + time.day * 100
            i = 0
            while time.time() > self.TimePeriod[i][1]:
                i += 1
            time_idx = date + i
            poi_idx = self.poi2idx[poi]
            region = latlon2quadkey(float(lat), float(lon), LOD)
            if region not in region2idx:
                region2idx[region] = n_region
                idx2region[n_region] = region
                n_region += 1
            region_idx = region2idx[region]
            regionidx2poi[region_idx].add(poi_idx)
            if user not in user_seq:
                user_seq[user] = list()
            user_seq[user].append([poi_idx, time_idx, region_idx, region, lat, lon, time])
        print("Filtering cold users (checked less than 20)...")
        for user, seq in user_seq.items():
            if len(seq) >= min_freq:
                user2idx[user] = n_user
                user_idx = n_user
                seq_new = list()
                tmp_set = set()
                cnt = 0
                for poi, time_idx, region_idx, region, lat, lon, time in sorted(seq, key=lambda e: e[6]):
                    if poi in tmp_set:
                        seq_new.append((user_idx, poi, time_idx, region, lat, lon, time, True))
                    else:
                        seq_new.append((user_idx, poi, time_idx, region, lat, lon, time, False))
                        tmp_set.add(poi)
                        cnt += 1
                if cnt > min_freq / 2:
                    n_user += 1
                    user_seq_array.append(seq_new)
        print("Spliting QUADKEY to n-grams...")
        all_quadkeys = []
        for u in range(len(user_seq_array)):
            checkin_seq = user_seq_array[u]
            for i in range(len(checkin_seq)):
                check_in = checkin_seq[i]
                user_idx = check_in[0]
                poi = check_in[1]
                time_idx = check_in[2]
                quadkey = check_in[3]
                lat = check_in[4]
                lon = check_in[5]
                time = check_in[6]
                bool = check_in[7]

                quadkey_n_gram = ' '.join([''.join(x) for x in ngrams(quadkey, 6)])
                quadkey_n_gram = quadkey_n_gram.split()
                all_quadkeys.append(quadkey_n_gram)
                user_seq_array[u][i] = (user_idx, poi, time_idx, quadkey_n_gram, lat, lon, time, bool)
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

        return user_seq_array, user2idx, region2idx, n_user, n_region, regionidx2poi

    def __len__(self):
        return len(self.user_seq)

    def __getitem__(self, idx):
        return self.user_seq[idx]

    def split_build_matrix(self, max_len=100):
        train_ = copy.copy(self)
        test_ = copy.copy(self)
        train_seq = list()
        test_seq = list()
        k_t = 864000
        k_g = 10000
        print("Spliting train / test dataset and Building time / geography distance interval matrix...")
        start_time_total = T.time()
        for u in range(len(self)):
            start_time_u = T.time()
            print("\n", u+1, "/", len(self) + 1, "Sequence Length =", len(self[u]))
            checkin_seq = self[u]
            for i in reversed(range(len(checkin_seq))):
                if not checkin_seq[i][-1]:
                    break
            print("Last Novel POI Index =", i)
            num_subseq = math.floor((i + max_len - 1 - 1) // max_len)
            print("Train Num Subsequences =", num_subseq)
            for b in range(num_subseq):
                if (i - b * max_len) > max_len:
                    tgt = checkin_seq[(i - (b + 1) * max_len): (i - b * max_len)]
                    src = checkin_seq[(i - (b + 1) * max_len - 1): (i - b * max_len - 1)]
                    start_pos = i - (b + 1) * max_len - 1
                    end_pos = i - b * max_len - 1
                    print("Train ", b+1, "/", num_subseq, "Sub src Length = ", len(src), "[", start_pos, ":", end_pos, "]")
                    print("Train ", b+1, "/", num_subseq, "Sub tgt Length = ", len(tgt), "[", start_pos + 1, ":", end_pos + 1, "]")
                    time = list()
                    lat = list()
                    lon = list()
                    for counter in range(max_len):
                        time.append(checkin_seq[start_pos + counter][6])
                        lat.append(checkin_seq[start_pos + counter][4])
                        lon.append(checkin_seq[start_pos + counter][5])
                    time_interval_matrix = torch.zeros((max_len, max_len))
                    geo_interval_matrix = torch.zeros((max_len, max_len))
                    print("Train Building matrix", b+1, "/", num_subseq)
                    for x in range(max_len):
                        for y in range(x, max_len):
                            time_interval_matrix[x][y] = abs(time[x] - time[y]).total_seconds()
                            time_interval_matrix[x][y] = min(time_interval_matrix[x][y], k_t)
                            time_interval_matrix[y][x] = time_interval_matrix[x][y]
                            x_point = (lat[x], lon[x])
                            y_point = (lat[y], lon[y])
                            geo_interval_matrix[x][y] = geodesic(x_point, y_point).meters
                            geo_interval_matrix[x][y] = min(geo_interval_matrix[x][y], k_g)
                            geo_interval_matrix[y][x] = geo_interval_matrix[x][y]

                    time_interval_matrix = time_interval_matrix.max() - time_interval_matrix
                    geo_interval_matrix = geo_interval_matrix.max() - geo_interval_matrix
                    train_seq.append((src, tgt, time_interval_matrix, geo_interval_matrix))
                else:
                    if (i - b * max_len - 1) == 0:
                        break
                    tgt = checkin_seq[1: (i - b * max_len)]
                    src = checkin_seq[0: (i - b * max_len - 1)]
                    time = list()
                    lat = list()
                    lon = list()
                    start_pos = 0
                    end_pos = i - b * max_len - 1
                    print("Train ", b+1, "/", num_subseq, "Sub src Length = ", len(src), "[", start_pos, ":", end_pos, "]")
                    print("Train ", b+1, "/", num_subseq, "Sub tgt Length = ", len(tgt), "[", start_pos + 1, ":", end_pos + 1, "]")
                    l = len(src)
                    time_interval_matrix = torch.zeros((max_len, max_len))
                    geo_interval_matrix = torch.zeros((max_len, max_len))
                    for counter in range(l):
                        time.append(checkin_seq[start_pos + counter][6])
                        lat.append(checkin_seq[start_pos + counter][4])
                        lon.append(checkin_seq[start_pos + counter][5])
                    print("Train Building matrix", b+1, "/", num_subseq)
                    for x in range(max_len - l, max_len):
                        for y in range(x, max_len):
                            time_interval_matrix[x][y] = abs(
                                time[x - (max_len - l)] - time[y - (max_len - l)]).total_seconds()
                            time_interval_matrix[x][y] = min(time_interval_matrix[x][y], k_t)
                            time_interval_matrix[y][x] = time_interval_matrix[x][y]
                            x_point = (lat[x - (max_len - l)], lon[x - (max_len - l)])
                            y_point = (lat[y - (max_len - l)], lon[y - (max_len - l)])
                            geo_interval_matrix[x][y] = geodesic(x_point, y_point).meters
                            geo_interval_matrix[x][y] = min(geo_interval_matrix[x][y], k_g)
                            geo_interval_matrix[y][x] = geo_interval_matrix[x][y]

                    time_interval_matrix = time_interval_matrix.max() - time_interval_matrix
                    geo_interval_matrix = geo_interval_matrix.max() - geo_interval_matrix
                    train_seq.append((src, tgt, time_interval_matrix, geo_interval_matrix))
                    break

            time_interval_matrix = torch.zeros((max_len, max_len))
            geo_interval_matrix = torch.zeros((max_len, max_len))
            if i - max_len > 0:
                test_tgt = checkin_seq[i: i + 1]
                test_src = checkin_seq[(i - max_len): i]
                start_pos = i - max_len
                end_pos = i
                print("Test src Length = ", len(test_src), "[", start_pos, ":", end_pos, "]")
                print("Test tgt Length = ", len(test_tgt), "[", i, ":", i + 1, "]")
                start_pos = i - max_len
                time = list()
                lat = list()
                lon = list()
                for counter in range(max_len):
                    time.append(checkin_seq[start_pos + counter][6])
                    lat.append(checkin_seq[start_pos + counter][4])
                    lon.append(checkin_seq[start_pos + counter][5])
                print("Test Building Matrix")
                for x in range(max_len):
                    for y in range(x, max_len):
                        time_interval_matrix[x][y] = abs(time[x] - time[y]).total_seconds()
                        time_interval_matrix[x][y] = min(time_interval_matrix[x][y], k_t)
                        time_interval_matrix[y][x] = time_interval_matrix[x][y]
                        x_point = (lat[x], lon[x])
                        y_point = (lat[y], lon[y])
                        geo_interval_matrix[x][y] = geodesic(x_point, y_point).meters
                        geo_interval_matrix[x][y] = min(geo_interval_matrix[x][y], k_g)
                        geo_interval_matrix[y][x] = geo_interval_matrix[x][y]

            else:
                test_tgt = checkin_seq[i: i + 1]
                test_src = checkin_seq[0: i]
                start_pos = 0
                end_pos = i
                print("Test src Length = ", len(test_src), "[", start_pos, ":", end_pos, "]")
                print("Test tgt Length = ", len(test_tgt), "[", i, ":", i + 1, "]")
                start_pos = 0
                time = list()
                lat = list()
                lon = list()
                l = len(test_src)
                for counter in range(l):
                    time.append(checkin_seq[start_pos + counter][6])
                    lat.append(checkin_seq[start_pos + counter][4])
                    lon.append(checkin_seq[start_pos + counter][5])
                print("Test Building Matrix")
                for x in range(max_len - l, max_len):
                    for y in range(x, max_len):
                        time_interval_matrix[x][y] = abs(
                            time[x - (max_len - l)] - time[y - (max_len - l)]).total_seconds()
                        time_interval_matrix[x][y] = min(time_interval_matrix[x][y], k_t)
                        time_interval_matrix[y][x] = time_interval_matrix[x][y]
                        x_point = (lat[x - (max_len - l)], lon[x - (max_len - l)])
                        y_point = (lat[y - (max_len - l)], lon[y - (max_len - l)])
                        geo_interval_matrix[x][y] = geodesic(x_point, y_point).meters
                        geo_interval_matrix[x][y] = min(geo_interval_matrix[x][y], k_g)
                        geo_interval_matrix[y][x] = geo_interval_matrix[x][y]

            time_interval_matrix = time_interval_matrix.max() - time_interval_matrix
            geo_interval_matrix = geo_interval_matrix.max() - geo_interval_matrix
            test_seq.append((test_src, test_tgt, time_interval_matrix, geo_interval_matrix))
            print(u+1, "/", len(self) + 1, "Takes", T.time()-start_time_u / 1, "min")

        print("Finished! Takes:", (T.time()-start_time_total) / 3600, "h")
        train_.user_seq = train_seq
        test_.user_seq = sorted(test_seq, key=lambda e: len(e[0]))

        joblib.dump(train_, 'data/demo_train.data')
        joblib.dump(test_, 'data/demo_test.data')
        return train_, test_


if __name__ == "__main__":
    data_raw = 'data/demo.txt'
    data_clean = 'data/demo.data'

    if os.path.exists(data_clean):
        dataset = unserialize(data_clean)
    else:
        dataset = LBSNDataset(data_raw)
        serialize(dataset)

    count = 0
    length = []
    for seq in dataset.user_seq:
        count += len(seq)-2
        length.append(len(seq))
    print("#check-ins:", count)
    print("#users:", dataset.n_user - 1)
    print("#POIs:", dataset.n_poi - 1)
    print("#median seq len:", np.median(np.array(length)))


