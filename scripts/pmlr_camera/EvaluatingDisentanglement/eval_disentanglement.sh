#!/usr/bin/env bash
GPU=$1
VAEUID=$2

echo "GPU:" $GPU
echo "VAEUID:" $VAEUID

CUDA_VISIBLE_DEVICES=$GPU python -m VAE.disentangle_eval \
-vae-model-path models/VAE/GECO6/ \
-save-dir results -ds-name carla45fully48pxfactors -img-size 48 \
--fully_obs --tile_size 8 --grid_size 6 -alt_img_count 3 -vae-uid $VAEUID