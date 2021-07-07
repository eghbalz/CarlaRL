#!/usr/bin/env bash
echo "plt1_appx (last iter)"
bash scripts/pmlr_camera/EvaluatingAgents/plt1_appx/rescore_seeds_all.sh 1 last
echo "plt1_main (last iter)"
bash scripts/pmlr_camera/EvaluatingAgents/plt1_main/rescore_seeds_all.sh 1 last
echo "plt2_appx (last iter)"
bash scripts/pmlr_camera/EvaluatingAgents/plt2_appx/rescore_seeds_all.sh 1 last
echo "plt2_main (last iter)"
bash scripts/pmlr_camera/EvaluatingAgents/plt2_main/rescore_seeds_all.sh 1 last
echo "plt3_main (last iter)"
bash scripts/pmlr_camera/EvaluatingAgents/plt3_main/rescore_seeds_all.sh 1 last
