#!/usr/bin/env bash
# seeds 44864 26912 94869 88994 24946 34416 73735 65066
GPU=$1
SEED=$2
export OMP_NUM_THREADS=2
CUDA_VISIBLE_DEVICES=$GPU python -m carla.train_agent --norm_obs --seed $SEED --exp_name carlac_geco_trainvae --hid_context_net 0 \
--contextual --context_encoder_model_path models/VAE/GECO6/2021-4-26_14-7-18.150064 --conditioning carlac --vae_train