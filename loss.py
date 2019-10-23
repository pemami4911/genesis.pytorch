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

def genesis_loss(data_batch, vae, geco_ema, reconstruction_target):
    """
    GECO constrained opt ELBO

    data_batch: images [N, C, H, W]
    vae: computes variational posterior
    returns loss, to be optimized and the elbo
    """

    batch_size = data_batch.shape[0]
    results_dict = vae(data_batch)

    log_pi = results_dict['pis']
    x_loc = results_dict['x_loc']
    x_log_scale = results_dict['x_log_scale']

    ## Likelihood term
    # output is [batch_size]
    log_prob = image_batch_gmm_log_prob(data_batch, log_pi, x_loc, x_log_scale)

    # KL terms
    # MVNs same shapes as priors
    q_zm = results_dict['q_zm']
    q_zc = results_dict['q_zc']
    
    prior_zm, prior_zc = vae.sample_prior()

    kl_q_zm = torch.distributions.kl.kl_divergence(q_zm, prior_zm)
    kl_q_zc = torch.distributions.kl.kl_divergence(q_zc, prior_zc)
 
    # sum over K for the KL divergences, result is [batch_size]
    kl_q_zm = kl_q_zm.view(-1, batch_size).sum(0)
    kl_q_zc = kl_q_zc.view(-1, batch_size).sum(0)

    # GECO doesn't optimize this
    # [batch_size]
    train_elbo = log_prob - (kl_q_zm + kl_q_zc)
    
    # GECO optimizes this one
    # [batch_size]
    if not geco_ema:
        constraint_term = reconstruction_target - torch.mean(log_prob)
    else:
        # update EMA
        constraint_term = geco_ema(torch.mean(log_prob))
    
    loss = torch.mean(kl_q_zm + kl_q_zc) + constraint_term
    
    return {
        'loss': loss,
        'elbo': torch.mean(train_elbo),
        'KL': torch.mean(kl_q_zm + kl_q_zc),
        'reconstruction': torch.mean(log_prob),
        'model_outs': results_dict
    }
