#!/usr/bin/env bash
# bash scripts/pmlr_all/eval/context_clf/context_clf_eval.sh 1 2021-4-26_14-7-18.150064 linear 0.001 100 none
# bash scripts/pmlr_all/eval/context_clf/context_clf_eval.sh 1 2021-4-26_14-7-18.150064 linear 0.001 100 step
GPU=$1
VAEUID=$2
CLFTYP=$3
LR=$4
NEPOCH=$5
SCHEDUL=$6

echo "GPU:" $GPU
echo "VAE UID:" $VAEUID
echo "CLF TYPE:" $CLFTYP
echo "LR:" $LR
echo "NEPOCH:" $NEPOCH
echo "SCHEDUL:" $SCHEDUL

CUDA_VISIBLE_DEVICES=$GPU python -m Context_Clf.context_clf_eval \
-vae-model-path models/VAE/ -vae-uid $VAEUID \
-save-dir /home/hamid/results -ds-name carla45fully48px -img-size 48 \
-clf-type $CLFTYP -lr $LR -num-epochs $NEPOCH -schedule-type $SCHEDUL