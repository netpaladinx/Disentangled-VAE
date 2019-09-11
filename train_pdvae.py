import os
import argparse

import numpy as np
import torch
import torch.optim as optim

from data import get_data_loader
import utils as U
from model_pdvae import PDVAE


parser = argparse.ArgumentParser()
parser.add_argument('-p', '--long_train_params', type=str, default='')

parser.add_argument('--name', type=str, default='benchmark')
parser.add_argument('--version', type=str, default='v1')
parser.add_argument('-d', '--dataset', type=str, default='dsprites_full')
parser.add_argument('-e', '--experiment', type=int, default=0)
parser.add_argument('-s', '--seed', type=int, default=0)
parser.add_argument('-c', '--cuda_id', type=int, default=0)
parser.add_argument('-o', '--objective_type', type=str, default='beta_vae')
parser.add_argument('-r', '--recon_type', type=str, default='bernoulli')

parser.add_argument('--n_channels', type=int, default=1)
parser.add_argument('--n_latent', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--max_steps', type=int, default=300000)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--adam_beta1', type=float, default=0.9)
parser.add_argument('--adam_beta2', type=float, default=0.999)
parser.add_argument('--d_learning_rate', type=float, default=0.0001)
parser.add_argument('--d_adam_beta1', type=float, default=0.5)
parser.add_argument('--d_adam_beta2', type=float, default=0.9)

parser.add_argument('--beta', type=float, default=4)  # for Beta-VAE (1,2,4,6,8,16) and Beta-TCVAE (1,2,4,6,8,10)
parser.add_argument('--gamma', type=float, default=10)  #  for FactorVAE (10,20,30,40,50,100) and AnnealedVAE (1000)
parser.add_argument('--c_max', type=float, default=25)  # for AnnealedVAE (5,10,25,50,75,100)
parser.add_argument('--iter_threshold', type=float, default=100000)  # for AnnealedVAE (100000)
parser.add_argument('--lambda_od', type=float, default=10)  # for DIP-VAE-I (1,2,5,10,20,50) and DIP-VAE-II (1,2,5,10,20,50)
parser.add_argument('--lambda_d_factor', type=float, default=10)  # for DIP-VAE-I (10) and DIP-VAE-II (1)

parser.add_argument('--order', type=int, default=3)
parser.add_argument('--detach_z', action='store_true', default=False)
parser.add_argument('--scale_z', action='store_true', default=False)

parser.add_argument('--save_base', type=str, default='./checkpoints')
parser.add_argument('--output_base', type=str, default='./output')
parser.add_argument('--print_freq', type=int, default=300)
parser.add_argument('--save_freq', type=int, default=30000)

args = parser.parse_args()


def train(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.lambda_d = args.lambda_d_factor * args.lambda_od
    hp_tags = {'beta_vae': 'beta({})'.format(args.beta),
               'beta_tc_vae': 'beta({})'.format(args.beta),
               'annealed_vae': 'c_max({})'.format(args.c_max),
               'factor_vae': 'gamma({})'.format(args.gamma),
               'dip_vae_i': 'lambda_od({})'.format(args.lambda_od),
               'dip_vae_ii': 'lambda_od({})'.format(args.lambda_od)}
    args.tag = '{},{},{},{},{},order{},exp{}'.format(args.name, args.version, args.dataset, args.objective_type,
                                                     hp_tags[args.objective_type], args.order, args.experiment)
    args.cuda = 'cuda:{}'.format(args.cuda_id)
    args.save_dir = os.path.join(args.save_base, args.tag)
    args.output_dir = os.path.join(args.output_base, args.tag)

    U.mkdir(args.save_dir)
    device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')

    train_loader, ds = get_data_loader(args.dataset, args.batch_size, args.max_steps)
    args.n_channels = ds.channels
    pd_params = {'order': args.order,
                 'detach_z': args.detach_z,
                 'scale_z': args.scale_z}
    reg_params = {'beta': args.beta,
                  'gamma': args.gamma,
                  'c_max': args.c_max,
                  'iter_threshold': args.iter_threshold,
                  'lambda_od': args.lambda_od,
                  'lambda_d': args.lambda_d}
    model = PDVAE(args.n_channels, args.n_latent, pd_params, reg_params, objective_type=args.objective_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2))

    if args.objective_type == 'factor_vae':
        d_optimizer = optim.Adam(model.parameters(for_discriminator=True), lr=args.d_learning_rate,
                                 betas=(args.d_adam_beta1, args.d_adam_beta2))

    model.train()
    step = 0
    moving = 0.01
    loss_avg, neg_elbo_avg, recon_loss_avg, kl_loss_avg, tc_loss_avg, cov_d_loss_avg, cov_od_loss_avg = \
        None, None, None, None, None, None, None
    for batch in train_loader:
        step = step + 1
        _, inputs = batch
        x = inputs.to(device).float()
        recon_x, zs_mean, zs_logvar, _ = model(x)
        losses = model.loss(x, recon_x, zs_mean, zs_logvar, step=step, recon_type=args.recon_type)

        if args.objective_type == 'factor_vae':
            d_loss = model.dicriminate(zs_mean, zs_logvar)

        if args.objective_type == 'vae':
            loss, recon_loss, kl_loss = losses
            loss_avg, recon_loss_avg, kl_loss_avg = \
                U.moving_average(moving, (loss.item(), recon_loss.item(), kl_loss.item()),
                                 (loss_avg, recon_loss_avg, kl_loss_avg))
            neg_elbo_avg = loss_avg
            loss_info = 'loss: {:.4f}, neg_elbo: {:.4f}, recon_loss: {:.4f}, kl_loss: {:.4f}'.format(
                loss_avg, recon_loss_avg, kl_loss_avg, neg_elbo_avg)

        elif args.objective_type == 'beta_vae':
            loss, neg_elbo, recon_loss, kl_loss = losses
            loss_avg, neg_elbo_avg, recon_loss_avg, kl_loss_avg = \
                U.moving_average(moving, (loss.item(), neg_elbo.item(), recon_loss.item(), kl_loss.item()),
                                 (loss_avg, neg_elbo_avg, recon_loss_avg, kl_loss_avg))
            loss_info = 'loss: {:.4f}, neg_elbo: {:.4f}, recon_loss: {:.4f}, kl_loss: {:.4f}'.format(
                loss_avg, neg_elbo_avg, recon_loss_avg, kl_loss_avg)

        elif args.objective_type == 'annealed_vae':
            loss, neg_elbo, recon_loss, kl_loss = losses
            loss_avg, neg_elbo_avg, recon_loss_avg, kl_loss_avg = \
                U.moving_average(moving, (loss.item(), neg_elbo.item(), recon_loss.item(), kl_loss.item()),
                                 (loss_avg, neg_elbo_avg, recon_loss_avg, kl_loss_avg))
            loss_info = 'loss: {:.4f}, neg_elbo: {:.4f}, recon_loss: {:.4f}, kl_loss: {:.4f}'.format(
                loss_avg, neg_elbo_avg, recon_loss_avg, kl_loss_avg)

        elif args.objective_type == 'factor_vae':
            loss, neg_elbo, recon_loss, kl_loss, tc_loss = losses
            loss_avg, neg_elbo_avg, recon_loss_avg, kl_loss_avg, tc_loss_avg = \
                U.moving_average(moving, (loss.item(), neg_elbo.item(), recon_loss.item(), kl_loss.item(), tc_loss.item()),
                                 (loss_avg, neg_elbo_avg, recon_loss_avg, kl_loss_avg, tc_loss_avg))
            loss_info = 'loss: {:.4f}, neg_elbo: {:.4f}, recon_loss: {:.4f}, kl_loss: {:.4f}, tc_loss: {:.4f}'.format(
                loss_avg, neg_elbo_avg, recon_loss_avg, kl_loss_avg, tc_loss_avg)

        elif args.objective_type in ['dip_vae_i', 'dip_vae_ii']:
            loss, neg_elbo, recon_loss, kl_loss, cov_d_loss, cov_od_loss = losses
            loss_avg, neg_elbo_avg, recon_loss_avg, kl_loss_avg, cov_d_loss_avg, cov_od_loss_avg = \
                U.moving_average(moving, (loss.item(), neg_elbo.item(), recon_loss.item(), kl_loss.item(), cov_d_loss.item(), cov_od_loss.item()),
                                 (loss_avg, neg_elbo_avg, recon_loss_avg, kl_loss_avg, cov_d_loss_avg, cov_od_loss_avg))
            loss_info = 'loss: {:.4f}, neg_elbo: {:.4f}, recon_loss: {:.4f}, kl_loss: {:.4f}, cov_d_loss: {:.4f}, ' \
                        'cov_od_loss: {:.4f}'.format(
                loss_avg, neg_elbo_avg, recon_loss_avg, kl_loss_avg, cov_d_loss_avg, cov_od_loss_avg)

        elif args.objective_type == 'beta_tc_vae':
            loss, neg_elbo, recon_loss, kl_loss, tc_loss = losses
            loss_avg, neg_elbo_avg, recon_loss_avg, kl_loss_avg, tc_loss_avg = \
                U.moving_average(moving, (loss.item(), neg_elbo.item(), recon_loss.item(), kl_loss.item(), tc_loss.item()),
                                 (loss_avg, neg_elbo_avg, recon_loss_avg, kl_loss_avg, tc_loss_avg))
            loss_info = 'loss: {:.4f}, neg_elbo: {:.4f}, recon_loss: {:.4f}, kl_loss: {:.4f}, tc_loss: {:.4f}'.format(
                loss_avg, neg_elbo_avg, recon_loss_avg, kl_loss_avg, tc_loss_avg)

        else:
            raise ValueError('Invalid `objective_type`')

        optimizer.zero_grad()
        loss.backward()

        if args.objective_type == 'factor_vae':
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

        optimizer.step()

        if step % args.print_freq == 0:
            print('[{} {} {}][Step {:d}] {}'.format(
                args.objective_type, args.dataset, hp_tags[args.objective_type], step, loss_info))
        if step % args.save_freq == 0:
            path = os.path.join(args.save_dir, 'step-%d.ckpt' % step)
            print('Save {}'.format(path))
            torch.save({'step': step, 'loss_info': loss_info, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, path)


if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    train(args)
