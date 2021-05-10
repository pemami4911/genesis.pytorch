import torch
import torch.nn as nn
import math
from lib.layers import GatedConv2dBN, GatedConvTranspose2dBN
from lib.layers import Flatten, Reshape
from sacred import Ingredient
from lib.utils import init_weights, mvn
from lib.geco import GECO
from lib.loss import image_batch_gmm_log_prob

import numpy as np

net = Ingredient('Net')

@net.config
def cfg():
    input_size = [3, 64, 64] # [C, H, W]
    zm_size = 64
    zc_size = 16
    K = 7
    background_log_scale = math.log(0.09)
    foreground_log_scale = math.log(0.11)
    geco_warm_start = 1000

class AutoregressivePrior(nn.Module):
    """
    p(z^m_1:K).
    """

    @net.capture
    def __init__(self, zm_size, K, batch_size):
        super(AutoregressivePrior, self).__init__()

        self.zm_size = zm_size
        self.K = K
        self.batch_size = batch_size

        self.zm_1 = nn.Parameter(torch.zeros(1, 256))
        self.lstm = nn.LSTM(256, 256)
        self.h, self.c = (torch.zeros(1, 1, 256),
                torch.zeros(1, 1, 256))
        self.p_zm_loc = nn.Linear(256, self.zm_size)
        self.p_zm_scale = nn.Linear(256, self.zm_size)

    def forward(self):
        """
        MVN with diagonal covariance of shape [K * batch_size, zm_size]
        """
        z_mK = [self.zm_1]
        h = self.h.clone()
        c = self.c.clone()
        for i in range(self.K-1):
            z, (h, c) = self.lstm(z_mK[-1].unsqueeze(1), (h, c))
            z_mK += [z.squeeze(1)]
        p_z = torch.stack(z_mK).view(-1, 256) # [K, 256]
        p_zm_loc = self.p_zm_loc(p_z) # [K, zm_size]
        p_zm_scale = self.p_zm_scale(p_z) # [K, zm_size]

        # copy across batch size elements
        p_zm_loc = p_zm_loc.unsqueeze(1).repeat(1, self.batch_size, 1).view(-1, self.zm_size)
        p_zm_scale = p_zm_scale.unsqueeze(1).repeat(1, self.batch_size, 1).view(-1, self.zm_size)
        return mvn(p_zm_loc, p_zm_scale)

class ContentPrior(nn.Module):
    """
    p(z^c | z^m)
    """
    @net.capture
    def __init__(self, zm_size, zc_size, batch_size):
        super(ContentPrior, self).__init__()
        self.batch_size = batch_size
        self.zc_size = zc_size

        self.p_zc = nn.Sequential(
            nn.Linear(zm_size, 256),
            nn.ELU(True),
            nn.Linear(256, 256),
            nn.ELU(True)
        )
        self.p_zc_loc = nn.Linear(256, zc_size)
        self.p_zc_scale = nn.Linear(256, zc_size)

    def forward(self, zm):
        zc = self.p_zc(zm)
        p_zc_loc = self.p_zc_loc(zc)
        p_zc_scale = self.p_zc_scale(zc)

        return mvn(p_zc_loc, p_zc_scale)


class AutoregressiveMaskEncoder(nn.Module):
    """
    q(z_m | x)

    Creates a gated convnet encoder.
    the encoder expects data as input of shape (batch_size, num_channels, height, width).

    channels [32, 32, 64, 64, 64]
    strides [1, 2, 1, 2, 1] for 4x downsample
    kernel size 5x5
    """
    @net.capture
    def __init__(self, input_size, zm_size, K):
        super(AutoregressiveMaskEncoder, self).__init__()
        self.input_size = input_size
        self.zm_size = zm_size
        self.K = K

        self.q_zm = nn.Sequential(
            GatedConv2dBN(self.input_size[0], 32, 1, 2),
            GatedConv2dBN(32, 32, 2, 2),
            GatedConv2dBN(32, 64, 1, 2),
            GatedConv2dBN(64, 64, 2, 2),
            GatedConv2dBN(64, 64, 1, 2),
            Flatten(),
            nn.Linear( (self.input_size[1]//4)*(self.input_size[1]//4)*64, self.zm_size*2)
        )
        self.lstm = nn.LSTM(self.zm_size*2, self.zm_size*2, batch_first=True)
        self.h, self.c = (torch.zeros(1, self.zm_size*2),
                torch.zeros(1, self.zm_size*2))
        self.q_zm_loc = nn.Linear(self.zm_size*2, self.zm_size)
        self.q_zm_scale = nn.Linear(self.zm_size*2, self.zm_size)

    def forward(self, x):
        """
        expects following data shapes as input:
        x.shape = (batch_size, num_channels, height, width)

        returns MVN with shape = (K * batch_size, zm_size)
        """

        # image embedding
        phi_x = self.q_zm(x).unsqueeze(1)
        z_mK = []
        # autoregressive embedding
        h = self.h.unsqueeze(0).repeat(1, x.shape[0], 1)
        c = self.c.unsqueeze(0).repeat(1, x.shape[0], 1)
        for i in range(self.K):
            z, (h, c) = self.lstm(phi_x, (h, c))
            z_mK += [z.squeeze(1)]
        q_z = torch.stack(z_mK)  # [K, batch_size, self.zm_size*2]
        q_z = q_z.view(-1, self.zm_size*2) # [K * batch_size, self.zm_size*2]
        q_zm_loc = self.q_zm_loc(q_z)
        q_zm_scale = self.q_zm_scale(q_z)

        return mvn(q_zm_loc, q_zm_scale)

class MONetComponentEncoder(nn.Module):
    """
    q(z_c | z_m, x)
    """
    @net.capture
    def __init__(self, input_size, zc_size, K):
        super(MONetComponentEncoder, self).__init__()
        self.input_size = input_size
        self.zc_size = zc_size
        self.K = K

        self.encode = nn.Sequential(
            nn.Conv2d(input_size[0]+1, 32, 3, stride=2, padding=1),
            nn.ELU(True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ELU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ELU(True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ELU(True),
            Flatten(),
            nn.Linear((self.input_size[1]//16)*(self.input_size[1]//16)*64, 256),
            nn.ELU(True),
        )

        self.q_zc_loc = nn.Linear(256, self.zc_size)
        self.q_zc_scale = nn.Linear(256, self.zc_size)

    def forward(self, x, log_mask):
        """
        x is (batch_size, num_channels, height, width)
        log_mask is (K * batch_size, 1, height, width)
        """
        x = x.repeat(self.K, 1, 1, 1)
        x = torch.cat((x, log_mask), axis=1)
        zc = self.encode(x) # [K * batch_size, zc_size*2]
        loc = self.q_zc_loc(zc)
        scale = self.q_zc_scale(zc)
        return mvn(loc, scale)

class MaskDecoder(nn.Module):
    """
    \pi(z_m)
    """
    @net.capture
    def __init__(self, input_size, zm_size, K):
        super(MaskDecoder, self).__init__()
        self.input_size = input_size
        self.zm_size = zm_size
        self.K = K

        self.pi_zm_logits = nn.Sequential(
            nn.Linear(self.zm_size, 64*(input_size[1]//4)*(input_size[1]//4)),
            Reshape((-1, 64, self.input_size[1] // 4, self.input_size[2] // 4)),
            GatedConvTranspose2dBN(64, 64, 1, 2),
            GatedConvTranspose2dBN(64, 32, 2, 2, output_padding=1),
            GatedConvTranspose2dBN(32, 32, 1, 2),
            GatedConvTranspose2dBN(32, 32, 2, 2, output_padding=1),
            GatedConvTranspose2dBN(32, 32, 1, 2),
        )
        self.out = nn.Conv2d(32, 1, 1)

    def init_bias(self):
        self.out.bias = nn.Parameter(
                -math.log(self.K-1) * torch.ones(self.out.bias.shape))

    def forward(self, z):
        """
        Decoder outputs masks in the following
        shape = [K * batch_size, 1, height, width]
        """
        mask_logits = self.pi_zm_logits(z)
        mask_logits = self.out(mask_logits)
        probs = nn.functional.logsigmoid(mask_logits)
        return mask_logits, probs

class SpatialBroadcastDecoder(nn.Module):
    """
    Decodes the individual Gaussian image componenets of
    the mixture p(x | z_m, z_c)
    """
    @net.capture
    def __init__(self, input_size, zc_size):
        super(SpatialBroadcastDecoder, self).__init__()

        self.h, self.w = input_size[1], input_size[2]

        self.decode = nn.Sequential(
            nn.Conv2d(zc_size+2, 32, 3, 1, 0),
            nn.ELU(True),
            nn.Conv2d(32, 32, 3, 1, 0),
            nn.ELU(True),
            nn.Conv2d(32, 32, 3, 1, 0),
            nn.ELU(True),
            nn.Conv2d(32, 32, 3, 1, 0),
            nn.ELU(True),
            nn.Conv2d(32, input_size[0], 1, 1)
        )

    @staticmethod
    def spatial_broadcast(z, h, w):
        """
        source: https://github.com/baudm/MONet-pytorch/blob/master/models/networks.py
        """
        # Batch size
        n = z.shape[0]
        # Expand spatially: (n, z_dim) -> (n, z_dim, h, w)
        z_b = z.view((n, -1, 1, 1)).expand(-1, -1, h, w)
        # Coordinate axes:
        x = torch.linspace(-1, 1, w, device=z.device)
        y = torch.linspace(-1, 1, h, device=z.device)
        y_b, x_b = torch.meshgrid(y, x)
        # Expand from (h, w) -> (n, 1, h, w)
        x_b = x_b.expand(n, 1, -1, -1)
        y_b = y_b.expand(n, 1, -1, -1)
        # Concatenate along the channel dimension: final shape = (n, z_dim + 2, h, w)
        z_sb = torch.cat((z_b, x_b, y_b), dim=1)
        return z_sb

    def forward(self, zc):
        z_sb = SpatialBroadcastDecoder.spatial_broadcast(zc, self.h + 8, self.w + 8)
        x_loc = self.decode(z_sb) # [batch_size * K, 3, h, w]
        return x_loc

class GENESIS(nn.Module):
    """
    GENESIS wrapper class for various encoders/decoders

    Based on https://github.com/riannevdberg/sylvester-flows/blob/master/models/VAE.py
    """
    @net.capture
    def __init__(self, zm_size, zc_size, input_size, K, batch_size, geco_warm_start, background_log_scale, foreground_log_scale):
        super(GENESIS, self).__init__()

        self.zm_size = zm_size
        self.zc_size = zc_size
        self.input_size = input_size
        self.K = K
        self.gmm_log_scale = torch.cat([torch.FloatTensor([background_log_scale]), foreground_log_scale * torch.ones(K-1)], 0)
        self.gmm_log_scale = self.gmm_log_scale.view(K, 1, 1, 1, 1)

        self.prior_zm = AutoregressivePrior(batch_size=batch_size)
        self.prior_zc = ContentPrior(batch_size=batch_size)
        self.autoregressive_mask_encoder = AutoregressiveMaskEncoder()
        self.component_encoder = MONetComponentEncoder()
        self.mask_decoder = MaskDecoder()
        self.image_decoder = SpatialBroadcastDecoder()

        init_weights(self.prior_zm, 'truncated_normal')
        init_weights(self.prior_zc, 'truncated_normal')
        init_weights(self.autoregressive_mask_encoder, 'truncated_normal')
        init_weights(self.component_encoder, 'truncated_normal')
        init_weights(self.mask_decoder, 'truncated_normal')
        init_weights(self.image_decoder, 'truncated_normal')

        self.init_scope = torch.zeros(batch_size, 1, input_size[1], input_size[2])
        self.eps = torch.finfo(torch.float).eps

        self.geco_warm_start = geco_warm_start
        self.geco_C_ema = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.geco_beta = nn.Parameter(torch.tensor(0.55), requires_grad=False)

    def sample_prior(self):
        """
        Returns two MVNs for zm and zc
        """

        p_zm = self.prior_zm()
        # Sample z_m from the prior
        zm = p_zm.rsample()
        p_zc = self.prior_zc(zm)

        return p_zm, p_zc

    def stick_breaking(self, log_probs):
        """
        Implements a stick breaking construction
        of GMM assignment probabilities pi_k out
        of the pis
        """

        log_probs = log_probs.view(self.K, -1, 1, self.input_size[1], self.input_size[2])
        sum_p_k = self.init_scope.clone()

        log_p_k = [log_probs[0]]
        sum_p_k += log_probs[0].exp()

        for k in range(1,self.K-1):
            one_minus_sum_p_k = (1. - sum_p_k).clamp(min=self.eps)
            log_p_k += [one_minus_sum_p_k.log() + log_probs[k]]
            sum_p_k += log_p_k[-1].exp()

        log_p_k += [(1. - sum_p_k).clamp(min=self.eps).log()]
        return torch.stack(log_p_k) # [K, batch_size, 1, H, W]


    def genesis_loss(self, imgs, preds, geco, step, kl_beta):
        """
        GECO constrained opt ELBO

        data_batch: images [N, C, H, W]
        vae: computes variational posterior
        returns loss, to be optimized and the elbo
        """

        batch_size = imgs.shape[0]

        log_pi = preds['pis']
        x_loc = preds['x_loc']
        x_log_scale = preds['x_log_scale']

        ## Likelihood term
        # output is [batch_size]
        log_prob = image_batch_gmm_log_prob(imgs, log_pi, x_loc, x_log_scale)

        # KL terms
        # MVNs same shapes as priors
        q_zm = preds['q_zm']
        q_zc = preds['q_zc']
        
        prior_zm, prior_zc = self.sample_prior()

        kl_q_zm = torch.distributions.kl.kl_divergence(q_zm, prior_zm)
        kl_q_zc = torch.distributions.kl.kl_divergence(q_zc, prior_zc)
    
        # sum over K for the KL divergences, result is [batch_size]
        kl_q_zm = kl_q_zm.view(-1, batch_size).sum(0)
        kl_q_zc = kl_q_zc.view(-1, batch_size).sum(0)

        # GECO doesn't optimize this
        # [batch_size]
        train_elbo = log_prob - kl_beta * (kl_q_zm + kl_q_zc)

        nll = -log_prob
        if self.geco_warm_start > step or geco is None:
            loss = torch.mean(nll + kl_beta * (kl_q_zm + kl_q_zc))
        else:
            loss = kl_beta * torch.mean((kl_q_zm + kl_q_zc)) - geco.constraint(self.geco_C_ema, self.geco_beta, torch.mean(nll))
                

        return {
            'loss': loss,
            'elbo': torch.mean(train_elbo),
            'KL': torch.mean(kl_q_zm + kl_q_zc),
            'reconstruction': torch.mean(log_prob),
            'model_outs': preds
        }


    def forward(self, x, geco, global_step, kl_beta):
        """
        Evaluates the model as a whole, encodes and decodes.
        """
        self.gmm_log_scale=self.gmm_log_scale.to(x.device)
        self.init_scope=self.init_scope.to(x.device)
        self.prior_zm.zm_1=self.prior_zm.zm_1.to(x.device)
        self.prior_zm.h=self.prior_zm.h.to(x.device)
        self.prior_zm.c=self.prior_zm.c.to(x.device)
        self.autoregressive_mask_encoder.c=self.autoregressive_mask_encoder.c.to(x.device)
        self.autoregressive_mask_encoder.h=self.autoregressive_mask_encoder.h.to(x.device)

        # mean and variance of zm
        q_zm = self.autoregressive_mask_encoder(x)
        # sample zm
        zm = q_zm.rsample()

        mask_logits, mask_logprobs = self.mask_decoder(zm)

        # mean and variance of zc
        q_zc = self.component_encoder(x, mask_logprobs)
        # sample zc
        zc = q_zc.rsample()

        # [batch_size * K, C, H, W]
        x_loc = self.image_decoder(zc)
        x_loc = torch.sigmoid(x_loc)  # map values to (0,1)
        x_loc = x_loc.view(self.K, -1, self.input_size[0], self.input_size[1], self.input_size[2])

        # [K, batch_size, 1, H, W]
        pis = self.stick_breaking(mask_logprobs)

        if torch.isinf(pis).any() or torch.isnan(pis).any():
            print('Invalid value encountered during training')
            import pdb; pdb.set_trace()

        preds = {
            'q_zm': q_zm,
            'q_zc': q_zc,
            'mask_logprobs': mask_logprobs,
            'x_loc': x_loc,
            'pis': pis,
            'x_log_scale': self.gmm_log_scale,
            'zm': zm,
            'zc': zc
        }

        out_dict = self.genesis_loss(x, preds, geco, global_step, kl_beta)
        return out_dict
