import torch

import torch.nn as nn

class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()

    def forward(self, x, y):
        num_pts = x.shape[0]
        dist[:] = torch.pow(torch.cdist(x,y),2)
        in_dist_out = torch.min(dist, dim=0)
        out_dist_in = torch.min(dist, dim=1)
        loss = torch.sum(in_dist_out.values + out_dist_in.values) / num_pts
        loss.requres_grad = True

'''Unused in this project'''
# from Steven Tsan https://github.com/stsan9/AnomalyDetection4Jets/blob/emd/code/loss_util.py#L3
def chamfer_loss(x, y):
    nparts = x.shape[0]
    dist = torch.pow(torch.cdist(x,y),2)
    in_dist_out = torch.min(dist,dim=0)
    out_dist_in = torch.min(dist,dim=1)
    loss = torch.sum(in_dist_out.values + out_dist_in.values) / nparts
    loss.requres_grad = True
    return loss

# Reconstruction + KL divergence losses
def vae_loss(x, y, mu, logvar):
    BCE = chamfer_loss(x,y)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD
