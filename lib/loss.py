import torch
import math

def image_batch_gmm_log_prob(imgs, log_pi, loc, x_log_scale):
    """
    K-component GMM likelihood for image batch

    imgs: [N, C, H, W]
    log_pi: [K, N, 1, H, W]
    loc: [K, N, C, H, W]
    log_scale: [K, 1, 1, 1, 1]

    returns [batch_size] FloatTensor
    """
    x_log_var = 2 * x_log_scale
    # compute log prob of imgs
    # sq is [K, batch_size, C, H, W]
    sq = (loc - imgs).pow(2)
    log_p = log_pi - 0.5 * x_log_var - 0.5 * (sq / torch.exp(x_log_var))
    # logsumexp over slots - result is [batch_size, C, H, W]
    log_p = torch.logsumexp(log_p, dim=0)
    # sum over pixels - result is [batch_size]
    gmm_log_prob = torch.sum(log_p, dim=[1,2,3])

    return gmm_log_prob

