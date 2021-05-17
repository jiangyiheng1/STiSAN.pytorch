import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryCELoss(nn.Module):
    def __init__(self):
        super(BinaryCELoss, self).__init__()

    def forward(self, pos_score, neg_score, probs):
        loss = -F.logsigmoid(pos_score.squeeze()) + torch.sum(F.softplus(neg_score) / neg_score.size(2), dim=-1)
        return loss