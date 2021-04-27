import torch

import torch.nn as nn

class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__(device)

        self.device = device

    # # Adapted from Steven Tsan https://github.com/stsan9/AnomalyDetection4Jets/blob/emd/code/loss_util.py#L3
    # def forward(self, x, y):
    #     num_pts = x.shape[0]
    #     dist = torch.pow(torch.cdist(x,y),2) + 1e-6
    #     in_dist_out = torch.min(dist, dim=1)
    #     out_dist_in = torch.min(dist, dim=2)
    #     loss = torch.sum(in_dist_out.values + out_dist_in.values) / num_pts
    #     return loss

    def forward(self, x, y):
        dist = pairwise_distance(x, y)

        min_xy = torch.min(dist, dim=-1)
        min_yx = torch.min(dist, dim=-2)

        loss = torch.sum(in_dist_out.values + out_dist_in.values)

        return loss

    def pairwise_distance(self, x, y):
        assert (x.shape[0] == y.shape[0]), f"The batch size of x and y are not equals! x.shape[0] is {x.shape[0]}, whereas y.shape[0] is {y.shape[0]}!"
        assert (x.shape[-1] == y.shape[-1]), f"Feature dimesion of x and y are not equals! x.shape[-1] is {x.shape[-1]}, whereas y.shape[-1] is {y.shape{-1}}!"

        batch_size = x.shape[0]
        num_row = x.shape[1]
        num_col = y.shape[1]
        vec_dim = x.shape[-1]

        x1 = x.repeat(1, 1, num_col).view(batch_size, -1, num_col, vec_dim).to(self.device)
        y1 = y.repeat(1, num_row, 1).view(batch_size, num_row, -1, vec_dim).to(self.device)

        dist = torch.norm(x1 - y1 + 1e-12, dim=-1)

        return dist
