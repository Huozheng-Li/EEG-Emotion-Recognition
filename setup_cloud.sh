#!/bin/bash
# AutoDL cloud setup script — run once after connecting via SSH
set -e

echo "=== Setting up EEG Emotion Recognition on AutoDL ==="

# 1. Create conda env (AutoDL base has Python 3.10 + CUDA 11.8)
echo "[1/4] Creating conda environment..."
conda create -n eeg python=3.10 -y 2>/dev/null
source activate eeg

# 2. Install PyTorch for CUDA 11.8
echo "[2/4] Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install other dependencies
echo "[3/4] Installing other packages..."
pip install mne scikit-learn pandas matplotlib seaborn tqdm openpyxl h5py

# 4. Verify
echo "[4/4] Verifying..."
python -c "
import torch, mne, sklearn
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.0f}GB')
print('All packages OK!')
"

echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Upload data files to ~/eeg_emotion_recog/"
echo "  2. Run: python src/train/pretrain.py then src/train/finetune.py"
