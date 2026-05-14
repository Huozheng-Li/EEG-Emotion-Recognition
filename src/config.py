"""
Project configuration — paths, model hyperparams, training settings.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
DEAP_DIR = DATA_DIR / "DEAP"            # Place DEAP .dat files here
COMP_DIR = ROOT / "PROBLEM"             # Competition data
TRAIN_DIR = COMP_DIR / "训练集"
TEST_DIR = COMP_DIR / "公开测试集"
CHECKPOINT_DIR = ROOT / "checkpoints"
PROCESSED_DIR = DATA_DIR / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── EEG Settings ────────────────────────────────────────────────────
N_CHANNELS = 30
DEAP_SFREQ = 128        # DEAP sampling rate
COMP_SFREQ = 250        # Competition sampling rate
TRIAL_LENGTH_SEC = 5    # Segment length in seconds
STRIDE_SEC = 2.5        # Sliding window stride (50% overlap)
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 50.0

# ── Model: TSception ────────────────────────────────────────────────
TSCEPTION = {
    "sampling_rate": 250,
    "num_T": 15,               # Temporal conv branches
    "num_S": 15,               # Spatial conv branches
    "hid_channels": 32,        # Hidden channels per branch
    "dropout": 0.5,
    "num_classes": 2,
}

# ── Training: Pretrain on DEAP ──────────────────────────────────────
PRETRAIN = {
    "batch_size": 64,
    "epochs": 100,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "val_split": 0.15,
    "patience": 15,
    "num_workers": 4,
    "label_threshold": 5.0,    # Valence > 5 → positive
}

# ── Training: Finetune on Competition ──────────────────────────────
FINETUNE = {
    "batch_size": 32,
    "epochs": 80,
    "lr": 1e-4,
    "weight_decay": 1e-3,
    "val_split": 0.2,
    "patience": 15,
    "num_workers": 4,
    "n_folds": 5,
}
