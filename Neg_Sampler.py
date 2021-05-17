import torch
import torch.nn as nn


class UniformNegativeSampler(nn.Module):
    def __init__(self, nloc):
        nn.Module.__init__(self)
        self.n_loc = nloc

    def forward(self, trg_seq, k, **kwargs):
        return torch.randint(1, self.n_loc, [len(trg_seq), k]), torch.ones(len(trg_seq), k, dtype=torch.float32)