import torch

import torch.nn as nn

class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()

    # Adapted from Steven Tsan https://github.com/stsan9/AnomalyDetection4Jets/blob/emd/code/loss_util.py#L3
    def forward(self, x, y):
        num_pts = x.shape[0]
        dist = torch.pow(torch.cdist(x,y),2)
        in_dist_out = torch.min(dist, dim=1)
        out_dist_in = torch.min(dist, dim=2)
        loss = torch.sum(in_dist_out.values + out_dist_in.values) / num_pts
        return loss
