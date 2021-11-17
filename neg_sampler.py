import torch
import torch.nn as nn
import numpy as np


class KNNSampler(nn.Module):
    def __init__(self, query_sys, user_visited_locs, num_nearest=2000, exclude_visited=False):
        nn.Module.__init__(self)
        self.query_sys = query_sys
        self.num_nearest = num_nearest
        self.user_visited_locs = user_visited_locs
        self.exclude_visited = exclude_visited

    def forward(self, trg_seq, num_neg, user, **kwargs):
        neg_samples = []
        for check_in in trg_seq:
            trg_loc = check_in[1]
            nearby_locs = self.query_sys.get_knn(trg_loc, self.num_nearest)
            if not self.exclude_visited:
                samples = np.random.choice(nearby_locs, size=num_neg, replace=True)
            else:
                samples = []
                for _ in range(num_neg):
                    sample = np.random.choice(nearby_locs)
                    while sample in self.user_visited_locs[user]:
                        sample = np.random.choice(nearby_locs)
                    samples.append(sample)
            neg_samples.append(samples)
        neg_samples = torch.tensor(neg_samples, dtype=torch.long)
        probs = torch.ones_like(neg_samples, dtype=torch.float32)
        return neg_samples, probs