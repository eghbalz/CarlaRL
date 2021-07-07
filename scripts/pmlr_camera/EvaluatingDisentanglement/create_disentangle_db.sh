#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=$1 python -m VAE.create_disentangle_dataset \
--root_path  data/carla45fully48pxfactorsdb -save-dir results --fully_obs --tile_size 8 --grid_size 6 \
--context_config pmlr_all.yaml