import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from nflib.made import ARMLP


class MAF(nn.Module):
    """ Masked Autoregressive Flow that uses a MADE-style network for fast forward """
    def __init__(self, dim, parity, net_class=ARMLP, nh=24):
        super().__init__()
        self.dim = dim
        self.net = net_class(dim, dim*2, nh)
        self.parity = parity

    def forward(self, x):
        s, t = self.net(x)
        z = x * torch.exp(s) + t
        # permuted elements to improve learning efficiency 
        # pg5 https://arxiv.org/pdf/1705.07057.pdf
        z = z.flip(dims=(1,)) if self.parity else z
        log_det = torch.sum(s, dim=1)
        return z, log_det
    
    def backward(self, z):
        x = torch.zeros_like(z)
        log_det = 0 
        z = z.flip(dims=(1,)) if self.parity else z
        for i in range(self.dim): # iterative sampling from d=1 to D
            s, t = self.net(x.clone()) # clone to avoid in-place op errors if using IAF
            x[:, i] = (z[:, i] - t[:, i]) * torch.exp(-s[:, i])
            log_det += -s[:, i]
        return x, log_det

class NormalizingFlowModel(nn.Module):
    """ A Normalizing Flow Model is a (prior, flow) pair """
    def __init__(self, prior, af_flows):
        super().__init__()
        self.prior = prior
        self.af_flow = nn.ModuleList(af_flows)
        
    def iterate(self, data, direction=1):
        log_det = torch.zeros(data.shape[0])
        for af_flow in self.af_flow[::direction]:
            cur_flow = af_flow.forward if direction == 1 else af_flow.backward
            data, ld = cur_flow(data)
            log_det += ld
        return data, log_det
    
    def forward(self, x):
        z, log_det = self.iterate(x, direction=+1)
        prior_logprob = self.prior.log_prob(z).view(x.size(0), -1).sum(1)
        return z, prior_logprob, log_det

    def backward(self, z):
        return self.iterate(z, direction=-1)
    
    def sample(self, num_samples):
        z = self.prior.sample((num_samples,))
        x, _ = self.iterate(z, direction=-1)
        return x