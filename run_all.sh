#!/bin/bash
# Run all training tasks sequentially and save logs separately
# Usage: bash run_all.sh
set -e

cd /root/autodl-tmp/eeg_emotion_recog
source activate eeg

echo "===== [1/6] EEGNet DEAP Pretrain ====="
python -m src.train.pretrain --model eegnet
echo "Done."

echo "===== [2/6] EEGNet from scratch ====="
python -m src.train.finetune --model eegnet --pretrained none
mkdir -p checkpoints/logs_eegnet_scratch
mv checkpoints/logs/finetune_fold*.npz checkpoints/logs_eegnet_scratch/ 2>/dev/null
echo "Done."

echo "===== [3/6] EEGNet + DEAP Finetune ====="
python -m src.train.finetune --model eegnet --pretrained checkpoints/eegnet_deap_pretrain.pt
mkdir -p checkpoints/logs_eegnet_pretrained
mv checkpoints/logs/finetune_fold*.npz checkpoints/logs_eegnet_pretrained/ 2>/dev/null
echo "Done."

echo "===== [4/6] TSception from scratch ====="
python -m src.train.finetune --model tsception --pretrained none
mkdir -p checkpoints/logs_scratch
mv checkpoints/logs/finetune_fold*.npz checkpoints/logs_scratch/ 2>/dev/null
echo "Done."

echo "===== [5/6] TSception + DEAP Finetune ====="
python -m src.train.finetune --model tsception --pretrained checkpoints/tsception_deap_pretrain.pt
mkdir -p checkpoints/logs_pretrained
mv checkpoints/logs/finetune_fold*.npz checkpoints/logs_pretrained/ 2>/dev/null
echo "Done."

echo "===== [6/6] LightGBM ====="
python -m src.train.lightgbm_baseline
echo "Done."

echo "===== All tasks complete! ====="
ls -lh checkpoints/logs*/ checkpoints/*.pt checkpoints/*.pkl
