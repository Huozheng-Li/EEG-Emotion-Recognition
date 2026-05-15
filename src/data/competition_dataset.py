"""
Competition EEG dataset loader.

Training format (.mat v7.3 via h5py):
  Each subject file contains:
    - EEG_data_neu: (50000, 30) float32  → neutral emotion
    - EEG_data_pos: (50000, 30) float32  → positive emotion

Test format (.mat v5 via scipy):
    - test_eeg_c: (30, 20000) float64
"""
import numpy as np
import h5py
import scipy.io as sio
from pathlib import Path
from tqdm import tqdm

from src.data.preprocessing import bandpass_filter, segment_epochs, standardize_epochs
from src.config import TRAIN_DIR, TEST_DIR, COMP_SFREQ, TRIAL_LENGTH_SEC, \
    STRIDE_SEC, BANDPASS_LOW, BANDPASS_HIGH, N_CHANNELS, PROCESSED_DIR


def load_competition_subject(filepath: Path) -> tuple:
    """
    Load one training subject's .mat file.

    Returns:
        neu_data: (50000, 30) float32
        pos_data: (50000, 30) float32
    """
    with h5py.File(filepath, "r") as f:
        neu = f["EEG_data_neu"][:].astype(np.float32)
        pos = f["EEG_data_pos"][:].astype(np.float32)
    return neu, pos


def process_competition_subject(filepath: Path) -> tuple:
    """
    Preprocess one competition subject:
    bandpass filter → segment → standardize → combine both classes.

    Returns:
        epochs: (n_epochs, 30, trial_samples)
        labels: (n_epochs,)  0=neutral, 1=positive
    """
    neu, pos = load_competition_subject(filepath)

    epochs_list, labels_list = [], []

    for data, label in [(neu, 0), (pos, 1)]:
        data = bandpass_filter(data, COMP_SFREQ, BANDPASS_LOW, BANDPASS_HIGH)
        epochs = segment_epochs(data, COMP_SFREQ, TRIAL_LENGTH_SEC, STRIDE_SEC)
        epochs = standardize_epochs(epochs)
        epochs_list.append(epochs)
        labels_list.append(np.full(len(epochs), label, dtype=np.int64))

    X = np.concatenate(epochs_list, axis=0).astype(np.float32)
    y = np.concatenate(labels_list, axis=0)
    return X, y


def build_competition_dataset(cache: bool = True) -> tuple:
    """
    Load and preprocess all competition training subjects.

    Returns:
        X: (n_total_epochs, 30, trial_samples)
        y: (n_total_epochs,)
        subject_ids: (n_total_epochs,) 1-indexed subject ID
        groups: (n_total_epochs,) group label (0=HC, 1=DEP)
    """
    cache_file = PROCESSED_DIR / "competition_train.npz"
    if cache and cache_file.exists():
        print(f"[COMP] Loading cached data from {cache_file}")
        cached = np.load(cache_file)
        return (cached["X"], cached["y"],
                cached["subject_ids"], cached["groups"])

    X_list, y_list, subj_list, group_list = [], [], [], []
    subj_counter = 1

    for folder, group_id in [("正常人", 0), ("抑郁症患者", 1)]:
        folder_path = TRAIN_DIR / folder
        files = sorted(folder_path.glob("*.mat"))
        for fpath in tqdm(files, desc=f"  Loading {folder}", ncols=80):
            X_s, y_s = process_competition_subject(fpath)
            X_list.append(X_s)
            y_list.append(y_s)
            subj_list.append(np.full(len(y_s), subj_counter, dtype=np.int16))
            group_list.append(np.full(len(y_s), group_id, dtype=np.int16))
            subj_counter += 1

    print("  Concatenating...")
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    subj = np.concatenate(subj_list, axis=0)
    groups = np.concatenate(group_list, axis=0)

    if cache:
        print("  Caching to disk...")
        np.savez_compressed(cache_file, X=X, y=y,
                            subject_ids=subj, groups=groups)
        print(f"[COMP] Cached {len(y)} epochs to {cache_file}")

    return X, y, subj, groups


def load_test_subject(filepath: Path) -> np.ndarray:
    """
    Load one test subject .mat file, return (n_channels, n_times).
    """
    mat = sio.loadmat(filepath)
    for k, v in mat.items():
        if not k.startswith("__"):
            return v.astype(np.float64)  # (30, 20000)


def process_test_subject(filepath: Path) -> np.ndarray:
    """
    Preprocess one test subject → epochs.

    Returns:
        epochs: (n_epochs, 30, trial_samples)
    """
    data = load_test_subject(filepath)  # (30, 20000)
    data = bandpass_filter(data, COMP_SFREQ, BANDPASS_LOW, BANDPASS_HIGH)
    # segment_epochs expects (n_times, n_channels)
    data_t = data.T  # (20000, 30)
    epochs = segment_epochs(data_t, COMP_SFREQ, TRIAL_LENGTH_SEC, STRIDE_SEC)
    epochs = standardize_epochs(epochs)
    return epochs.astype(np.float32)


if __name__ == "__main__":
    X, y, subj, groups = build_competition_dataset()
    print(f"\nTotal epochs: {X.shape[0]}, pos ratio: {y.mean():.3f}")
    print(f"Subjects: {np.unique(subj)}")
    print(f"HC: {(groups == 0).sum()}, DEP: {(groups == 1).sum()}")
