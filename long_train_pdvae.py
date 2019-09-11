import numpy as np
import torch

import train_pdvae as T

N_SEEDS = 10


def sweep_order_for_pdvae(args, train):
    order_choices = [1, 2, 3, 4]
    for order in order_choices:
        print('===== order: %d =====' % order)
        args.order = order
        train(args)


def sweep_hparams_for_beta_vae(args, train):
    beta_choices = [1, 2, 4, 6, 8, 16]
    for beta in beta_choices:
        print('===== beta: %d =====' % beta)
        args.beta = beta
        sweep_order_for_pdvae(args, train)


def sweep_hparams_for_annealed_vae(args, train):
    c_max_choices = [5, 10, 25, 50, 75, 100]
    args.iter_threshold = 100000
    args.gamma = 1000
    for c_max in c_max_choices:
        print('===== c_max: %d =====' % c_max)
        args.c_max = c_max
        sweep_order_for_pdvae(args, train)


def sweep_hparams_for_beta_tc_vae(args, train):
    beta_choices = [1, 2, 4, 6, 8, 10]
    for beta in beta_choices:
        print('===== beta: %d =====' % beta)
        args.beta = beta
        sweep_order_for_pdvae(args, train)


def sweep_hparams_for_factor_vae(args, train):
    gamma_choices = [10, 20, 30, 40, 50, 100]
    for gamma in gamma_choices:
        print('===== gamma: %d =====' % gamma)
        args.gamma = gamma
        sweep_order_for_pdvae(args, train)


def sweep_hparams_for_dip_vae_i(args, train):
    lambda_od_choices = [1, 2, 5, 10, 20, 50]
    for lambda_od in lambda_od_choices:
        print('===== lambda_od: %d =====' % lambda_od)
        args.lambda_od = lambda_od
        args.lambda_d_factor = 10
        sweep_order_for_pdvae(args, train)


def sweep_hparams_for_dip_vae_ii(args, train):
    lambda_od_choices = [1, 2, 5, 10, 20, 50]
    for lambda_od in lambda_od_choices:
        print('===== lambda_od: %d =====' % lambda_od)
        args.lambda_od = lambda_od
        args.lambda_d_factor = 1
        sweep_order_for_pdvae(args, train)


sweep_train_fns = {'beta_vae': sweep_hparams_for_beta_vae,
                   'annealed_vae': sweep_hparams_for_annealed_vae,
                   'beta_tc_vae': sweep_hparams_for_beta_tc_vae,
                   'factor_vae': sweep_hparams_for_factor_vae,
                   'dip_vae_i': sweep_hparams_for_dip_vae_i,
                   'dip_vae_ii': sweep_hparams_for_dip_vae_ii}


def sweep_train(objective_type, args, train):
    sweep_train_fns[objective_type](args, train)


if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    T.args.version = 'v2'
    T.args.detach_z = False
    T.args.scale_z = True

    long_train_params = T.args.long_train_params
    sp = long_train_params.split(',')
    dataset= sp[0]
    objective_type_list = sp[1:]
    seeds = range(N_SEEDS)
    for seed in seeds:
        for objective_type in objective_type_list:

            T.args.dataset = dataset
            T.args.objective_type = objective_type
            print('')
            print('===== name: %s =====' % T.args.name)
            print('===== objective: %s =====' % objective_type)
            print('===== dataset: %s =====' % dataset)
            print('===== seed: %d =====' % seed)
            T.args.seed = seed
            T.args.experiment = seed

            sweep_train(objective_type, T.args, T.train)
