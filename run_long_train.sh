#!/usr/bin/env bash

SG1_CUDA0_PARAMS="cars3d,beta_tc_vae|dsprites_full,dip_vae_i|dsprites_full,annealed_vae"
SG2_CUDA0_PARAMS="mpi3d_toy,beta_tc_vae|dsprites_full,beta_vae|dsprites_full,dip_vae_ii"
SG3_CUDA0_PARAMS="mpi3d_toy,dip_vae_i|cars3d,annealed_vae|dsprites_full,beta_tc_vae"

MG1_CUDA0_PARAMS="mpi3d_toy,annealed_vae|color_dsprites,beta_tc_vae|color_dsprites,dip_vae_i"
MG1_CUDA1_PARAMS="cars3d,dip_vae_i|color_dsprites,factor_vae|color_dsprites,annealed_vae"
MG1_CUDA2_PARAMS="smallnorb,beta_tc_vae|color_dsprites,beta_vae|color_dsprites,dip_vae_ii"
MG1_CUDA3_PARAMS="smallnorb,dip_vae_i|noisy_dsprites,beta_tc_vae|noisy_dsprites,beta_vae"

MG2_CUDA0_PARAMS="mpi3d_toy,beta_vae|noisy_dsprites,annealed_vae|noisy_dsprites,dip_vae_ii"
MG2_CUDA1_PARAMS="cars3d,beta_vae|noisy_dsprites,factor_vae|noisy_dsprites,dip_vae_i"
MG2_CUDA2_PARAMS="smallnorb,annealed_vae|scream_dsprites,beta_tc_vae|scream_dsprites,dip_vae_i"
MG2_CUDA3_PARAMS="smallnorb,beta_vae|scream_dsprites,factor_vae|scream_dsprites,dip_vae_ii"

MG3_CUDA0_PARAMS="mpi3d_toy,dip_vae_ii|dsprites_full,factor_vae"
MG3_CUDA1_PARAMS="cars3d,dip_vae_ii|cars3d,factor_vae"
MG3_CUDA2_PARAMS="smallnorb,dip_vae_ii|mpi3d_toy,factor_vae"
MG3_CUDA3_PARAMS="smallnorb,factor_vae|scream_dsprites,annealed_vae|scream_dsprites,beta_vae"

hostname=`hostname`

if [ "$hostname" == "reco-research-xiaoran-single-gpu1" ]; then
    echo $hostname
    echo $SG1_CUDA0_PARAMS
    IFS='|'
    read -ra params <<< "$SG1_CUDA0_PARAMS"
    for p in "${params[@]}"; do
        echo $p
        nohup python -u long_train_benchmark.py -p $p -c 0 > benchmark-${p}.log 2>&1 </dev/null &
        nohup python -u long_train_pdvae.py -p $p -c 0 > pdvae-${p}.log 2>&1 </dev/null &
    done
fi

if [ "$hostname" == "reco-research-xiaoran-single-gpu2" ]; then
    echo $hostname
    echo $SG2_CUDA0_PARAMS
    IFS='|'
    read -ra params <<< "$SG2_CUDA0_PARAMS"
    for p in "${params[@]}"; do
        echo $p
        nohup python -u long_train_benchmark.py -p $p -c 0 > benchmark-${p}.log 2>&1 </dev/null &
        nohup python -u long_train_pdvae.py -p $p -c 0 > pdvae-${p}.log 2>&1 </dev/null &
    done
fi

if [ "$hostname" == "reco-research-xiaoran-single-gpu3" ]; then
    echo $hostname
    echo $SG3_CUDA0_PARAMS
    IFS='|'
    read -ra params <<< "$SG3_CUDA0_PARAMS"
    for p in "${params[@]}"; do
        echo $p
        nohup python -u long_train_benchmark.py -p $p -c 0 > benchmark-${p}.log 2>&1 </dev/null &
        nohup python -u long_train_pdvae.py -p $p -c 0 > pdvae-${p}.log 2>&1 </dev/null &
    done
fi

if [ "$hostname" == "reco-research-xiaoran-multi-gpu1" ]; then
    echo $hostname
    echo $MG1_CUDA0_PARAMS
    IFS='|'
    read -ra params <<< "$MG1_CUDA0_PARAMS"
    for p in "${params[@]}"; do
        echo $p
        nohup python -u long_train_benchmark.py -p $p -c 0 > benchmark-${p}.log 2>&1 </dev/null &
        nohup python -u long_train_pdvae.py -p $p -c 0 > pdvae-${p}.log 2>&1 </dev/null &
    done
    echo $MG1_CUDA1_PARAMS
    IFS='|'
    read -ra params <<< "$MG1_CUDA1_PARAMS"
    for p in "${params[@]}"; do
        echo $p
        nohup python -u long_train_benchmark.py -p $p -c 1 > benchmark-${p}.log 2>&1 </dev/null &
        nohup python -u long_train_pdvae.py -p $p -c 1 > pdvae-${p}.log 2>&1 </dev/null &
    done
    echo $MG1_CUDA2_PARAMS
    IFS='|'
    read -ra params <<< "$MG1_CUDA2_PARAMS"
    for p in "${params[@]}"; do
        echo $p
        nohup python -u long_train_benchmark.py -p $p -c 2 > benchmark-${p}.log 2>&1 </dev/null &
        nohup python -u long_train_pdvae.py -p $p -c 2 > pdvae-${p}.log 2>&1 </dev/null &
    done
    echo $MG1_CUDA3_PARAMS
    IFS='|'
    read -ra params <<< "$MG1_CUDA3_PARAMS"
    for p in "${params[@]}"; do
        echo $p
        nohup python -u long_train_benchmark.py -p $p -c 3 > benchmark-${p}.log 2>&1 </dev/null &
        nohup python -u long_train_pdvae.py -p $p -c 3 > pdvae-${p}.log 2>&1 </dev/null &
    done
fi

if [ "$hostname" == "reco-research-xiaoran-multi-gpu2" ]; then
    echo $hostname
    echo $MG2_CUDA0_PARAMS
    IFS='|'
    read -ra params <<< "$MG2_CUDA0_PARAMS"
    for p in "${params[@]}"; do
        echo $p
        nohup python -u long_train_benchmark.py -p $p -c 0 > benchmark-${p}.log 2>&1 </dev/null &
        nohup python -u long_train_pdvae.py -p $p -c 0 > pdvae-${p}.log 2>&1 </dev/null &
    done
    echo $MG2_CUDA1_PARAMS
    IFS='|'
    read -ra params <<< "$MG2_CUDA1_PARAMS"
    for p in "${params[@]}"; do
        echo $p
        nohup python -u long_train_benchmark.py -p $p -c 1 > benchmark-${p}.log 2>&1 </dev/null &
        nohup python -u long_train_pdvae.py -p $p -c 1 > pdvae-${p}.log 2>&1 </dev/null &
    done
    echo $MG2_CUDA2_PARAMS
    IFS='|'
    read -ra params <<< "$MG2_CUDA2_PARAMS"
    for p in "${params[@]}"; do
        echo $p
        nohup python -u long_train_benchmark.py -p $p -c 2 > benchmark-${p}.log 2>&1 </dev/null &
        nohup python -u long_train_pdvae.py -p $p -c 2 > pdvae-${p}.log 2>&1 </dev/null &
    done
    echo $MG2_CUDA3_PARAMS
    IFS='|'
    read -ra params <<< "$MG2_CUDA3_PARAMS"
    for p in "${params[@]}"; do
        echo $p
        nohup python -u long_train_benchmark.py -p $p -c 3 > benchmark-${p}.log 2>&1 </dev/null &
        nohup python -u long_train_pdvae.py -p $p -c 3 > pdvae-${p}.log 2>&1 </dev/null &
    done
fi

if [ "$hostname" == "reco-research-xiaoran-multi-gpu3" ]; then
    echo $hostname
    echo $MG3_CUDA0_PARAMS
    IFS='|'
    read -ra params <<< "$MG3_CUDA0_PARAMS"
    for p in "${params[@]}"; do
        echo $p
        nohup python -u long_train_benchmark.py -p $p -c 0 > benchmark-${p}.log 2>&1 </dev/null &
        nohup python -u long_train_pdvae.py -p $p -c 0 > pdvae-${p}.log 2>&1 </dev/null &
    done
    echo $MG3_CUDA1_PARAMS
    IFS='|'
    read -ra params <<< "$MG3_CUDA1_PARAMS"
    for p in "${params[@]}"; do
        echo $p
        nohup python -u long_train_benchmark.py -p $p -c 1 > benchmark-${p}.log 2>&1 </dev/null &
        nohup python -u long_train_pdvae.py -p $p -c 1 > pdvae-${p}.log 2>&1 </dev/null &
    done
    echo $MG3_CUDA2_PARAMS
    IFS='|'
    read -ra params <<< "$MG3_CUDA2_PARAMS"
    for p in "${params[@]}"; do
        echo $p
        nohup python -u long_train_benchmark.py -p $p -c 2 > benchmark-${p}.log 2>&1 </dev/null &
        nohup python -u long_train_pdvae.py -p $p -c 2 > pdvae-${p}.log 2>&1 </dev/null &
    done
    echo $MG3_CUDA3_PARAMS
    IFS='|'
    read -ra params <<< "$MG3_CUDA3_PARAMS"
    for p in "${params[@]}"; do
        echo $p
        nohup python -u long_train_benchmark.py -p $p -c 3 > benchmark-${p}.log 2>&1 </dev/null &
        nohup python -u long_train_pdvae.py -p $p -c 3 > pdvae-${p}.log 2>&1 </dev/null &
    done
fi
