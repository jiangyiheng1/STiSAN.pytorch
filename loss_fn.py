import torch.nn as nn
import torch.nn.functional as F
from einops import reduce, rearrange
import torch

class WeightedBCELoss(nn.Module):
    def __init__(self, temperature):
        super(WeightedBCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, pos_scores, neg_scores):
        # (b, n, 1) -> (b, n)
        pos_scores = rearrange(pos_scores, 'b n l -> b (n l)')
        # log(sigmoid(x)) (b, n)
        pos_part = F.logsigmoid(pos_scores)
        # (b, n, num_negs)
        weight = F.softmax(neg_scores / self.temperature, dim=-1)
        # negative scores: (b, n, num_negs) -> (b, n)
        neg_part = reduce(F.softplus(neg_scores) * weight, 'b n num_negs -> b n', 'mean')
        loss = -pos_part + neg_part

        return loss