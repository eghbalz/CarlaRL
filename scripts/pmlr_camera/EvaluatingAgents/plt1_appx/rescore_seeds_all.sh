#!/usr/bin/env bash
# Rescoring seen
GPU=$1
MODEL_ITR=$2
#
echo "CUDA_VISIBLE_DEVICES:" $GPU
echo "Model is chosen based on " $MODEL_ITR
#
echo "Start evaluating on seen env ..."
bash scripts/pmlr_camera/EvaluatingAgents/plt1_appx/rescore_all.sh model/agents $GPU $MODEL_ITR 0. 1. 50 44864 26912 94869 88994 24946 34416 73735 65066
bash scripts/pmlr_camera/EvaluatingAgents/plt1_appx/rescore_all.sh model/agents $GPU $MODEL_ITR 1. 0. 50 44864 26912 94869 88994 24946 34416 73735 65066
bash scripts/pmlr_camera/EvaluatingAgents/plt1_appx/rescore_all.sh model/agents $GPU $MODEL_ITR 0.5 0.5 50 44864 26912 94869 88994 24946 34416 73735 65066

