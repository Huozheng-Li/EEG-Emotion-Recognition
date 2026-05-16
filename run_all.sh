#!/bin/bash
# Run all training tasks sequentially and save logs separately
# Usage: bash run_all.sh
set -e

cd /root/autodl-tmp/eeg_emotion_recog
source activate eeg

echo "===== [1/4] DEAP Pretrain ====="
python -m src.train.pretrain
echo "Done."

echo "===== [2/4] TSception from scratch ====="
python -m src.train.finetune --pretrained none
mkdir -p checkpoints/logs_scratch
mv checkpoints/logs/finetune_fold*.npz checkpoints/logs_scratch/ 2>/dev/null
echo "Done."

echo "===== [3/4] TSception + DEAP Finetune ====="
python -m src.train.finetune --pretrained checkpoints/tsception_deap_pretrain.pt
mkdir -p checkpoints/logs_pretrained
mv checkpoints/logs/finetune_fold*.npz checkpoints/logs_pretrained/ 2>/dev/null
echo "Done."

echo "===== [4/4] LightGBM ====="
python -m src.train.lightgbm_baseline
echo "Done."

echo "===== All tasks complete! ====="
ls -lh checkpoints/logs*/ checkpoints/*.pt checkpoints/*.pkl
