import torch
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from scipy.stats import truncnorm

def truncated_normal_initializer(shape, mean, stddev):
    # compute threshold at 2 std devs
    values = truncnorm.rvs(mean - 2 * stddev, mean + 2 * stddev, size=shape)
    return torch.from_numpy(values).float()

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Modified from: https://github.com/baudm/MONet-pytorch/blob/master/models/networks.py

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == 'truncated_normal':
                m.weight.data = truncated_normal_initializer(m.weight.shape, 0.0, stddev=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)

class GECO(torch.nn.Module):
    def __init__(self, recon_target, err_recon, alpha=0.99):
        super(GECO, self).__init__()
        self.C = recon_target
        self.C_ema = torch.FloatTensor([self.C - err_recon]).to('cuda')
        self.beta = torch.nn.Parameter(torch.FloatTensor([0.55]))
        self.alpha = alpha

    def step_beta(self, step_size):
        with torch.no_grad():
            self.beta += step_size * self.C_ema
    
    def forward(self, err_recon):
        # update the EMA
        self.C_ema = (self.C_ema * self.alpha).detach() + ((self.C - err_recon) * (1. - self.alpha))
        # return the constraint term
        C_t = (self.C - err_recon)
        return torch.nn.functional.softplus(self.beta).detach() * (C_t + (self.C_ema - C_t).detach())
