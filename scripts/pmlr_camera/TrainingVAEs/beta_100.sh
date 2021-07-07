#!/usr/bin/env bash

GPU=$1
EPOCH=$2
BETA=$3
LR=$4

echo "CUDA_VISIBLE_DEVICES:" $GPU
echo "EPOCH:" $EPOCH
echo "BETA:" $BETA
echo "LR:" $LR

CUDA_VISIBLE_DEVICES=$GPU python -m VAE.train -save-dir models/VAE/ \
-ds-name carla45fully48px -img-size 48 -dec-out-nonlin none -prior gauss -init kaiming -num-epochs $EPOCH \
-latent-dim 100 -loss-type beta -vae-beta $BETA -lr $LR -batch-size 128 -vae-reduction norm_batch
