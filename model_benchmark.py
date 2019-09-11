import torch
import torch.nn as nn
import torch.nn.functional as F

import nets as N


class ConvEncoder(nn.Module):
    def __init__(self, in_channels, n_latent):
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 4, 2, padding=1)  # 3 x 64 x 64 --4x4+2--> 32 x 32 x 32
        self.conv2 = nn.Conv2d(32, 32, 4, 2, padding=1)  # 32 x 32 x 32 --4x4+2--> 32 x 16 x 16
        self.conv3 = nn.Conv2d(32, 64, 4, 2, padding=1)  # 32 x 16 x 16 --4x4+2--> 64 x 8 x 8
        self.conv4 = nn.Conv2d(64, 64, 4, 2, padding=1)  # 64 x 8 x 8 --4x4+2--> 64 x 4 x 4
        self.fc = nn.Linear(1024, 256)
        self.fc_mean = nn.Linear(256, n_latent)
        self.fc_logvar = nn.Linear(256, n_latent)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = out.reshape(-1, 1024)
        out = F.relu(self.fc(out))
        mean = self.fc_mean(out)
        logvar = self.fc_logvar(out)
        return mean, logvar


class DeconvDecoder(nn.Module):
    def __init__(self, out_channels, n_latent):
        super(DeconvDecoder, self).__init__()
        self.fc_latent = nn.Linear(n_latent, 256)
        self.fc = nn.Linear(256, 1024)
        self.deconv4 = nn.ConvTranspose2d(64, 64, 4, 2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, 2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 32, 4, 2, padding=1)
        self.deconv1 = nn.ConvTranspose2d(32, out_channels, 4, 2, padding=1)

    def forward(self, z):
        out = F.relu(self.fc_latent(z))
        out = F.relu(self.fc(out))
        out = out.reshape(-1, 64, 4, 4)
        out = F.relu(self.deconv4(out))
        out = F.relu(self.deconv3(out))
        out = F.relu(self.deconv2(out))
        out = self.deconv1(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, n_latent):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(n_latent, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 1000)
        self.fc4 = nn.Linear(1000, 1000)
        self.fc5 = nn.Linear(1000, 1000)
        self.fc6 = nn.Linear(1000, 1000)
        self.fc_logits = nn.Linear(1000, 2)

    def forward(self, z):
        out = F.leaky_relu(self.fc1(z))
        out = F.leaky_relu(self.fc2(out))
        out = F.leaky_relu(self.fc3(out))
        out = F.leaky_relu(self.fc4(out))
        out = F.leaky_relu(self.fc5(out))
        out = F.leaky_relu(self.fc6(out))
        logits = self.fc_logits(out)
        return logits


def vae_loss(x, recon_x, z_mean, z_logvar, recon_type='bernoulli'):
    recon_loss = N.bernoulli_reconstruction_loss(recon_x, x) if recon_type == 'bernoulli' \
        else N.l2_reconstruction_loss(recon_x, x)
    kl_loss = N.gaussian_kl_divergence(z_mean, z_logvar)
    loss = recon_loss + kl_loss
    return loss, recon_loss, kl_loss


class VAE(nn.Module):
    def __init__(self, n_channels, n_latent):
        super(VAE, self).__init__()
        self.encoder = ConvEncoder(n_channels, n_latent)
        self.decoder = DeconvDecoder(n_channels, n_latent)

    def forward(self, x, encoder_only=False):
        z_mean, z_logvar = self.encoder(x)
        z = N.sample_from_gaussian(z_mean, z_logvar)
        if encoder_only:
            return z_mean, z_logvar, z
        recon_x = self.decoder(z)
        return recon_x, z_mean, z_logvar, z

    def loss(self, x, recon_x, z_mean, z_logvar, recon_type='bernoulli'):
        return vae_loss(x, recon_x, z_mean, z_logvar, recon_type=recon_type)


def beta_vae_loss(x, recon_x, z_mean, z_logvar, beta, recon_type='bernoulli'):
    neg_elbo, recon_loss, kl_loss = vae_loss(x, recon_x, z_mean, z_logvar, recon_type=recon_type)
    loss = recon_loss + beta * kl_loss
    return loss, neg_elbo, recon_loss, kl_loss


class BetaVAE(VAE):
    ''' beta: 1, 2, 4 (recommended), 6, 8, 16
    '''
    def __init__(self, n_channels, n_latent, beta):
        super(BetaVAE, self).__init__(n_channels, n_latent)
        self.beta = beta

    def loss(self, x, recon_x, z_mean, z_logvar, recon_type='bernoulli'):
        return beta_vae_loss(x, recon_x, z_mean, z_logvar, self.beta, recon_type=recon_type)


def annealed_vae_loss(x, recon_x, z_mean, z_logvar, step, gamma, c_max, iter_threshold, recon_type='bernoulli'):
    neg_elbo, recon_loss, kl_loss = vae_loss(x, recon_x, z_mean, z_logvar, recon_type=recon_type)
    c = c_max * min(1.0, step / iter_threshold)
    loss = recon_loss + gamma * (kl_loss - c).abs()
    return loss, neg_elbo, recon_loss, kl_loss


class AnnealedVAE(VAE):
    ''' gamma: 1000
        c_max: 5, 10, 25, 50, 75, 100
        iter_threshold: 100000
    '''
    def __init__(self, n_channels, n_latent, gamma, c_max, iter_threshold):
        super(AnnealedVAE, self).__init__(n_channels, n_latent)
        self.gamma = gamma
        self.c_max = c_max
        self.iter_threshold = iter_threshold

    def loss(self, x, recon_x, z_mean, z_logvar, step=0, recon_type='bernoulli'):
        return annealed_vae_loss(x, recon_x, z_mean, z_logvar, step, self.gamma, self.c_max, self.iter_threshold,
                            recon_type=recon_type)


def factor_vae_discriminate(discriminator, z_mean, z_logvar, mode):
    z = N.sample_from_gaussian(z_mean, z_logvar)
    if mode == 'train':
        z = z.detach()
        prob_gen = discriminator(z).softmax(1).clamp(1e-6, 1 - 1e-6)  # B x 2
        prob_tar = discriminator(N.shuffle_code(z)).softmax(1).clamp(1e-6, 1 - 1e-6)  # B x 2
        return - 0.5 * (prob_gen[:, 0].log().mean() +
                        prob_tar[:, 1].log().mean())  # 0: fake, # 1: real
    elif mode == 'loss':
        logits_gen = discriminator(z)  # B x 2
        return (logits_gen[:, 0] - logits_gen[:, 1]).mean()
    else:
        raise ValueError('Invalid `mode`')


def factor_vae_loss(x, recon_x, z_mean, z_logvar, discriminator, gamma, recon_type='bernoulli', clamp_min=None):
    neg_elbo, recon_loss, kl_loss = vae_loss(x, recon_x, z_mean, z_logvar, recon_type=recon_type)
    tc_loss = factor_vae_discriminate(discriminator, z_mean, z_logvar, 'loss')
    if clamp_min is not None:
        tc_loss = tc_loss.clamp_min(clamp_min)
    loss = recon_loss + kl_loss + gamma * tc_loss
    return loss, neg_elbo, recon_loss, kl_loss, tc_loss


class FactorVAE(VAE):
    ''' gamma: 10, 20, 30, 40, 50, 100
    '''
    def __init__(self, n_channels, n_latent, gamma):
        super(FactorVAE, self).__init__(n_channels, n_latent)
        self.gamma = gamma
        self.discriminator = Discriminator(n_latent)

    def discriminate(self, z_mean, z_logvar):
        return factor_vae_discriminate(self.discriminator, z_mean, z_logvar, 'train')

    def loss(self, x, recon_x, z_mean, z_logvar, recon_type='bernoulli'):
        return factor_vae_loss(x, recon_x, z_mean, z_logvar, self.discriminator, self.gamma, recon_type=recon_type)

    def parameters(self, for_discriminator=False):
        if for_discriminator:
            return self.discriminator.parameters()
        else:
            return super(FactorVAE, self).parameters()


def dip_vae_loss(x, recon_x, z_mean, z_logvar, dip_type, lambda_od, lambda_d, recon_type='bernoulli'):
    neg_elbo, recon_loss, kl_loss = vae_loss(x, recon_x, z_mean, z_logvar, recon_type=recon_type)
    cov_z_mean = N.covariance_matrix(z_mean)
    if dip_type == 'i':
        cov_mat = cov_z_mean
    elif dip_type == 'ii':
        cov_mat = cov_z_mean + z_logvar.exp().mean(0).diag()
    else:
        raise ValueError('Invalid `dip_type`')
    cov_d = cov_mat.diag()
    cov_od = cov_mat - cov_d.diag()
    cov_d_loss = (cov_d - 1).pow(2).sum()
    cov_od_loss = cov_od.pow(2).sum()
    loss = recon_loss + kl_loss + lambda_d * cov_d_loss + lambda_od * cov_od_loss
    return loss, neg_elbo, recon_loss, kl_loss, cov_d_loss, cov_od_loss


class DIPVAE(VAE):
    ''' lambda_od: 1, 2, 5, 10, 20, 50
        lambda_d: 10 * lambda_od (for DIP-VAE-I) or lambda_od (for DIP-VAE-II)
    '''
    def __init__(self, n_channels, n_latent, lambda_od, lambda_d, dip_type='i'):
        super(DIPVAE, self).__init__(n_channels, n_latent)
        self.lambda_od = lambda_od
        self.lambda_d = lambda_d
        self.dip_type = dip_type

    def loss(self, x, recon_x, z_mean, z_logvar, recon_type='bernoulli'):
        return dip_vae_loss(x, recon_x, z_mean, z_logvar, self.dip_type, self.lambda_od, self.lambda_d,
                            recon_type=recon_type)


def beta_tc_vae_loss(x, recon_x, z_mean, z_logvar, beta, total_samples, recon_type='bernoulli'):
    neg_elbo, recon_loss, kl_loss = vae_loss(x, recon_x, z_mean, z_logvar, recon_type=recon_type)
    tc_loss = N.gaussian_total_correlation(z_mean, z_logvar, N=total_samples)
    loss = recon_loss + kl_loss + (beta - 1) * tc_loss
    return loss, neg_elbo, recon_loss, kl_loss, tc_loss


class BetaTCVAE(VAE):
    ''' beta: 1, 2, 4, 6, 8, 10
    '''
    def __init__(self, n_channels, n_latent, beta, total_samples=None):
        super(BetaTCVAE, self).__init__(n_channels, n_latent)
        self.beta = beta
        self.total_samples = total_samples

    def loss(self, x, recon_x, z_mean, z_logvar, recon_type='bernoulli'):
        return beta_tc_vae_loss(x, recon_x, z_mean, z_logvar, self.beta, self.total_samples, recon_type=recon_type)


def get_model_obj(name, args):
    if name == 'vae':
        return VAE(args.n_channels, args.n_latent)
    elif name == 'beta_vae':
        return BetaVAE(args.n_channels, args.n_latent, args.beta)
    elif name == 'annealed_vae':
        return AnnealedVAE(args.n_channels, args.n_latent, args.gamma, args.c_max, args.iter_threshold)
    elif name == 'factor_vae':
        return FactorVAE(args.n_channels, args.n_latent, args.gamma)
    elif name == 'dip_vae_i':
        return DIPVAE(args.n_channels, args.n_latent, args.lambda_od, args.lambda_d, dip_type='i')
    elif name == 'dip_vae_ii':
        return DIPVAE(args.n_channels, args.n_latent, args.lambda_od, args.lambda_d, dip_type='ii')
    elif name == 'beta_tc_vae':
        return BetaTCVAE(args.n_channels, args.n_latent, args.beta)
    else:
        raise ValueError('Invalid `name`')
