from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F

import nets as N
import model_benchmark as M


def get_cross_indices(max_id, order):
    indices = list(product(range(max_id + 1), repeat=order))[1:]
    indices = list(filter(lambda x: all([e1 <= e2 for e1, e2 in zip(x[:-1], x[1:])]) and (max_id in x), indices))
    return torch.tensor(indices)


class PDUnit(nn.Module):
    def __init__(self, z_id, n_dims, order=None, detach_z=False, scale_z=False):
        ''' z_id: starts from 1
        '''
        super(PDUnit, self).__init__()
        self.detach_z = detach_z
        self.scale_z = scale_z
        if scale_z:
            self.scale = nn.Parameter(torch.ones(z_id))  # i
        self.cross_indices = get_cross_indices(z_id, order)  # n_cross x order
        self.fc1 = nn.Linear(n_dims, 2)
        self.fc2 = nn.Sequential(nn.Linear(self.cross_indices.size(0), n_dims), nn.ReLU())

    def forward(self, x, prev_zs):
        ''' prez_zs: (contains z_1, ..., z_{i-1}) B x (i-1)
        '''
        out = self.fc1(x)  # B x 2
        z_mean = out[:, :1]  # B x 1
        z_logvar = out[:, 1:]  # B x 1
        z = N.sample_from_gaussian(z_mean, z_logvar)  # B x 1
        zs = z if prev_zs is None else torch.cat([prev_zs, z], 1)  # B x i

        ones = torch.ones(zs.size(0), 1).to(zs)  # B x 1
        zs_in = zs.detach() if self.detach_z else zs
        zs_in = zs_in * self.scale if self.scale_z else zs_in
        zs_tanh = torch.cat([ones, torch.tanh(zs_in)], 1)  # B x (i+1)
        zs_cross = zs_tanh[:, self.cross_indices].prod(2) # B x n_cross
        out = self.fc2(zs_cross)  # B x n_dims
        return out, z_mean, z_logvar, zs


class ConvEncoder(nn.Module):
    def __init__(self, in_channels, n_latent, pd_params):
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 4, 2, padding=1)  # 3 x 64 x 64 --4x4+2--> 32 x 32 x 32
        self.conv2 = nn.Conv2d(32, 32, 4, 2, padding=1)  # 32 x 32 x 32 --4x4+2--> 32 x 16 x 16
        self.conv3 = nn.Conv2d(32, 64, 4, 2, padding=1)  # 32 x 16 x 16 --4x4+2--> 64 x 8 x 8
        self.conv4 = nn.Conv2d(64, 64, 4, 2, padding=1)  # 64 x 8 x 8 --4x4+2--> 64 x 4 x 4
        self.fc = nn.Linear(1024, 256)
        self.pd_units = nn.ModuleList([PDUnit(i+1, 256, **pd_params) for i in range(n_latent)])

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = out.reshape(-1, 1024)
        out = F.relu(self.fc(out))

        zs_mean, zs_logvar = [], []
        zs = None
        for pd_unit in self.pd_units:
            pd_out, z_mean, z_logvar, zs = pd_unit(out, zs)
            out = out - pd_out
            zs_mean.append(z_mean)
            zs_logvar.append(z_logvar)
        zs_mean = torch.cat(zs_mean, 1)  # B x n_latent
        zs_logvar = torch.cat(zs_logvar, 1)  # B x n_latent
        return zs_mean, zs_logvar, zs


class DeconvDecoder(nn.Module):
    def __init__(self, out_channels, n_latent):
        super(DeconvDecoder, self).__init__()
        self.fc_latent = nn.Linear(n_latent, 256)
        self.fc = nn.Linear(256, 1024)
        self.deconv4 = nn.ConvTranspose2d(64, 64, 4, 2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, 2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 32, 4, 2, padding=1)
        self.deconv1 = nn.ConvTranspose2d(32, out_channels, 4, 2, padding=1)

    def forward(self, zs):
        out = F.relu(self.fc_latent(zs))
        out = F.relu(self.fc(out))
        out = out.reshape(-1, 64, 4, 4)
        out = F.relu(self.deconv4(out))
        out = F.relu(self.deconv3(out))
        out = F.relu(self.deconv2(out))
        out = self.deconv1(out)
        return out


class PDVAE(nn.Module):
    ''' Progressively Disentangled VAE '''
    def __init__(self, n_channels, n_latent, pd_params, reg_params, objective_type='factor_vae'):
        super(PDVAE, self).__init__()
        self.reg_params = reg_params
        self.objective_type = objective_type
        self.encoder = ConvEncoder(n_channels, n_latent, pd_params)
        self.decoder = DeconvDecoder(n_channels, n_latent)
        if objective_type == 'factor_vae':
            self.discriminator = M.Discriminator(n_latent)

    def dicriminate(self, zs_mean, zs_logvar):
        if self.objective_type == 'factor_vae':
            return M.factor_vae_discriminate(self.discriminator, zs_mean, zs_logvar, 'train')
        else:
            raise ValueError('`objective_type` should be `factor_vae`')

    def parameters(self, for_discriminator=False):
        if for_discriminator:
            return self.discriminator.parameters()
        else:
            return [{'params': self.encoder.parameters()},
                    {'params': self.decoder.parameters()}]

    def forward(self, x, encoder_only=False):
        zs_mean, zs_logvar, zs = self.encoder(x)
        if encoder_only:
            return zs_mean, zs_logvar, zs
        recon_x = self.decoder(zs)
        return recon_x, zs_mean, zs_logvar, zs

    def loss(self, x, recon_x, zs_mean, zs_logvar, step=None, total_samples=None, recon_type='bernoulli'):
        if self.objective_type == 'vae':
            return M.vae_loss(x, recon_x, zs_mean, zs_logvar, recon_type=recon_type)
        elif self.objective_type == 'beta_vae':
            return M.beta_vae_loss(x, recon_x, zs_mean, zs_logvar, self.reg_params['beta'], recon_type=recon_type)
        elif self.objective_type == 'annealed_vae':
            assert step is not None, 'Invalid `step`'
            return M.annealed_vae_loss(x, recon_x, zs_mean, zs_logvar, step, self.reg_params['gamma'],
                                       self.reg_params['c_max'], self.reg_params['iter_threshold'], recon_type=recon_type)
        elif self.objective_type == 'factor_vae':
            return M.factor_vae_loss(x, recon_x, zs_mean, zs_logvar, self.discriminator, self.reg_params['gamma'],
                                     recon_type=recon_type, clamp_min=0.0)
        elif self.objective_type == 'dip_vae_i':
            return M.dip_vae_loss(x, recon_x, zs_mean, zs_logvar, 'i', self.reg_params['lambda_od'],
                                  self.reg_params['lambda_d'], recon_type=recon_type)
        elif self.objective_type == 'dip_vae_ii':
            return M.dip_vae_loss(x, recon_x, zs_mean, zs_logvar, 'ii', self.reg_params['lambda_od'],
                                  self.reg_params['lambda_d'], recon_type=recon_type)
        elif self.objective_type == 'beta_tc_vae':
            return M.beta_tc_vae_loss(x, recon_x, zs_mean, zs_logvar, self.reg_params['beta'], total_samples,
                                      recon_type=recon_type)
