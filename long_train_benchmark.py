import numpy as np
import torch

import train_benchmark as T
import train_pdvae as T_pd
import long_train_pdvae as LT_pd

N_SEEDS = 10


def sweep_hparams_for_beta_vae(args, train):
    beta_choices = [1, 2, 4, 6, 8, 16]
    for beta in beta_choices:
        print('===== beta: %d =====' % beta)
        args.beta = beta
        train(args)


def sweep_hparams_for_annealed_vae(args, train):
    c_max_choices = [5, 10, 25, 50, 75, 100]
    args.iter_threshold = 100000
    args.gamma = 1000
    for c_max in c_max_choices:
        print('===== c_max: %d =====' % c_max)
        args.c_max = c_max
        train(args)


def sweep_hparams_for_beta_tc_vae(args, train):
    beta_choices = [1, 2, 4, 6, 8, 10]
    for beta in beta_choices:
        print('===== beta: %d =====' % beta)
        args.beta = beta
        train(args)


def sweep_hparams_for_factor_vae(args, train):
    gamma_choices = [10, 20, 30, 40, 50, 100]
    for gamma in gamma_choices:
        print('===== gamma: %d =====' % gamma)
        args.gamma = gamma
        train(args)


def sweep_hparams_for_dip_vae_i(args, train):
    lambda_od_choices = [1, 2, 5, 10, 20, 50]
    for lambda_od in lambda_od_choices:
        print('===== lambda_od: %d =====' % lambda_od)
        args.lambda_od = lambda_od
        args.lambda_d_factor = 10
        train(args)


def sweep_hparams_for_dip_vae_ii(args, train):
    lambda_od_choices = [1, 2, 5, 10, 20, 50]
    for lambda_od in lambda_od_choices:
        print('===== lambda_od: %d =====' % lambda_od)
        args.lambda_od = lambda_od
        args.lambda_d_factor = 1
        train(args)


sweep_train_fns = {'beta_vae': sweep_hparams_for_beta_vae,
                   'annealed_vae': sweep_hparams_for_annealed_vae,
                   'beta_tc_vae': sweep_hparams_for_beta_tc_vae,
                   'factor_vae': sweep_hparams_for_factor_vae,
                   'dip_vae_i': sweep_hparams_for_dip_vae_i,
                   'dip_vae_ii': sweep_hparams_for_dip_vae_ii}


def sweep_train(model_type, args, train):
    sweep_train_fns[model_type](args, train)

if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    T.args.version = 'v2'
    T_pd.args.version = 'v3'
    T_pd.args.detach_z = True
    T_pd.args.scale_z = True

    long_train_params = T.args.long_train_params
    sp = long_train_params.split(',')
    dataset = sp[0]
    model_type_list = sp[1:]
    seeds = range(N_SEEDS)
    for seed in seeds:
        for model_type in model_type_list:

            T.args.dataset = dataset
            T.args.model_type = model_type
            print('')
            print('===== name: %s =====' % T.args.name)
            print('===== model: %s =====' % model_type)
            print('===== dataset: %s =====' % dataset)
            print('===== seed: %d =====' % seed)
            T.args.seed = seed
            T.args.experiment = seed

            sweep_train(model_type, T.args, T.train)

            T_pd.args.dataset = dataset
            T_pd.args.objective_type = model_type
            print('')
            print('===== name: %s =====' % T_pd.args.name)
            print('===== objective: %s =====' % model_type)
            print('===== dataset: %s =====' % dataset)
            print('===== seed: %d =====' % seed)
            T_pd.args.seed = seed
            T_pd.args.experiment = seed

            LT_pd.sweep_train(model_type, T_pd.args, T_pd.train)
