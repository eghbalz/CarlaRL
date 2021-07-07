#!/usr/bin/env bash

GPU=$1
SEED=$2
export OMP_NUM_THREADS=2
CUDA_VISIBLE_DEVICES=$GPU python -m carla.train_agent --norm_obs --seed $SEED --exp_name darla_beta_trainvae --hid_context_net 0 \
--contextual --context_encoder_model_path models/VAE/Beta2.5Gauss/2021-4-26_9-4-5.815669 --conditioning darla --vae_train