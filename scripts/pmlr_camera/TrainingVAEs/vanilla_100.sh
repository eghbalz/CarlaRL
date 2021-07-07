#!/usr/bin/env bash

GPU=$1
EPOCH=$2

echo "CUDA_VISIBLE_DEVICES:" $GPU
echo "EPOCH:" $EPOCH

CUDA_VISIBLE_DEVICES=$GPU python -m VAE.train -save-dir models/VAE/ \
-ds-name carla45fully48px -img-size 48 -dec-out-nonlin none -prior gauss -init kaiming -num-epochs $EPOCH \
-latent-dim 100 -loss-type beta -lr 0.0005 -batch-size 128 -vae-reduction norm_batch

