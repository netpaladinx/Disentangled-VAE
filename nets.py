import math

import torch
import torch.nn.functional as F


def bernoulli_reconstruction_loss(recon_x, x):
    bs = x.size(0)
    x = x.reshape(bs, -1)
    recon_x = recon_x.reshape(bs, -1)
    loss_per_sample = F.binary_cross_entropy_with_logits(recon_x, x, reduction='none').sum(1)
    clamped_x = x.clamp(1e-6, 1-1e-6)
    loss_lower_bound = F.binary_cross_entropy(clamped_x, x, reduction='none').sum(1)
    loss = (loss_per_sample - loss_lower_bound).mean()
    return loss


def l2_reconstruction_loss(recon_x, x):
    bs = x.size(0)
    x = x.reshape(bs, -1)
    recon_x = recon_x.reshape(bs, -1)
    loss_per_sample = (torch.sigmoid(recon_x) - x).pow(2).sum(1)
    loss = loss_per_sample.mean()
    return loss


def gaussian_kl_divergence(z_mean, z_logvar):
    # 1/B sum_i{ 1/2 * sum_d{ mu_d^2 + sigma_d^2 - log(sigma_d^2) - 1 } }
    kl_i = 0.5 * torch.sum(z_mean * z_mean + z_logvar.exp() - z_logvar - 1, 1)  # B
    return torch.mean(kl_i)


def gaussian_total_correlation(z_mean, z_logvar, N=None):
    # samples: (x_i, z_i), i = 1, ..., B
    # E_z[log q(z)] ~= 1/B sum_i{ log(sum_j q(z_i|x_j)) - log(N*B) }
    #   q(z_i|x_j) = prod_d q(z_i^d|x_j)
    # E_z[log prod_d q(z^d)] ~= 1/B sum_i{ sum_d log(sum_j q(z_i^d|x_j)) - D * log(N*B) }

    z = sample_from_gaussian(z_mean, z_logvar)
    B = z.size(0)
    # compute log(q(z_i^d|x_j))
    diff = z.unsqueeze(1) - z_mean.unsqueeze(0)  # B (for i) x B (for j) x n_latents
    inv_sigma = torch.exp(-z_logvar.unsqueeze(0))  # 1 (for i) x B (for j) x n_latents
    normalization = math.log(2 * math.pi)  # 1
    log_q_zi_d_xj = -0.5 * (diff * diff * inv_sigma + z_logvar + normalization)  # B (for i) x B (for j) x n_latents
    # compute log q(z_i^d) = log(sum_j q(z_i^d|x_j) - log(N*B)
    log_q_zi_d = log_q_zi_d_xj.logsumexp(1) if N is None else log_q_zi_d_xj.logsumexp(1) - math.log(N * B)  # B (for i) x n_latent
    # compute i.i.d. log q(z_i) = log prod_d q(z_i^d) = sum log q(z_i^d)
    log_q_zi_iid = log_q_zi_d.sum(1)  # B (for i)
    # compute log q(z_i|x_j) = log(prod_d q(z_i^d|x_j))
    log_q_zi_xj = log_q_zi_d_xj.sum(2)  # B (for i) x B (for j)
    # compute log q(z_i) = log(sum_j q(z_i|x_j)) - log(N*B) (the constant term ignored)
    log_q_zi = log_q_zi_xj.logsumexp(1) if N is None else log_q_zi_xj.logsumexp(1) - math.log(N * B)  # B (for i)
    # compute 1/B sum_i{log q(z_i) - log q(z_i)_iid}
    return torch.mean(log_q_zi - log_q_zi_iid)  # 1


def covariance_matrix(z_mean):
    # cov(z_mean) = E[z_mean*z_mean^T] - E[z_mean]E[z_mean]^T
    mat = z_mean.unsqueeze(2) * z_mean.unsqueeze(1)  # B x n_latents x n_latents
    E_z_mean_z_mean_T = mat.mean(0)  # n_latents x n_latents
    E_z_mean = z_mean.mean(0)  # n_latents
    cov_z_mean = E_z_mean_z_mean_T - E_z_mean.unsqueeze(1) * E_z_mean.unsqueeze(0)  # n_latents x n_latents
    return cov_z_mean


def sample_from_gaussian(z_mean, z_logvar):
    noise = torch.randn_like(z_mean)
    return noise * (z_logvar / 2).exp() + z_mean


def shuffle_code(z):
    z_clone = z.clone()
    bs, n_latents = z.size()
    for i in range(bs):
        z_clone[i] = z_clone[i][torch.randperm(n_latents, device=z_clone.device)]
    return z_clone
