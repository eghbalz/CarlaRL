#!/usr/bin/env bash
#!/usr/bin/env bash
DIR=$1
GPU=$2
MODEL_ITR=$3
w_G="$4"
w_OG="$5"
EPISODES=$6
SEEDS=($7 $8 $9 ${10} ${11} ${12} ${13} ${14})

echo "CUDA_VISIBLE_DEVICES:" $GPU
echo "Model iter:" $MODEL_ITR
echo "DIR:" $DIR
echo "w_G:" $w_G
echo "w_OG:" $w_OG
echo "EPISODES:" $EPISODES
echo "SEEDS:" ${SEEDS[@]}


CUDA_VISIBLE_DEVICES=$GPU python -m carla.evaluate_agent \
$DIR/nc \
$DIR/onlinecarla_geco \
$DIR/carlac_geco_frozenvae \
--legend NC CARLA-CG-Onl. CARLA-CG \
--exp-id "plt3_main{$MODEL_ITR}" \
--episodes $EPISODES --save_plot --context_config pmlr_all.yaml \
--result_dir $DIR/results/ \
--plot_dir $DIR/final_plots/ \
--seeds ${SEEDS[@]} \
--title "" \
--w_G $w_G \
--w_OG $w_OG \
--itr $MODEL_ITR \
--plot_return