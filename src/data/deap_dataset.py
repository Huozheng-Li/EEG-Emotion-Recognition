"""
DEAP dataset loader.

DEAP format (per subject .dat file):
  - data:   (40, 40, 8064) → 40 trials × 40 channels × 8064 samples
    First 32 channels are EEG, last 8 are peripheral.
  - labels: (40, 4) → valence, arousal, dominance, liking (1–9)

We use valence > threshold → positive (1), else neutral (0).
"""
import pickle
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from src.data.preprocessing import bandpass_filter, segment_epochs, standardize_epochs
from src.config import DEAP_DIR, DEAP_SFREQ, TRIAL_LENGTH_SEC, STRIDE_SEC, \
    BANDPASS_LOW, BANDPASS_HIGH, PRETRAIN, N_CHANNELS


def load_deap_subject(subj_id: int, deap_dir: Path = None) -> dict:
    """
    Load a single DEAP subject's preprocessed .dat file.

    Returns dict with keys: 'data' (40,40,8064), 'labels' (40,4)
    """
    if deap_dir is None:
        deap_dir = DEAP_DIR
    fpath = deap_dir / f"s{subj_id:02d}.dat"
    if not fpath.exists():
        raise FileNotFoundError(
            f"DEAP file not found: {fpath}\n"
            "Download DEAP from https://www.eecs.qmul.ac.uk/mmv/datasets/deap/\n"
            "Place .dat files in data/DEAP/"
        )
    with open(fpath, "rb") as f:
        subject = pickle.load(f, encoding="latin1")
    return subject


def process_deap_subject(subj_id: int, deap_dir: Path = None) -> tuple:
    """
    Preprocess one DEAP subject: take first 30 EEG channels,
    bandpass filter, segment into trials, standardize.

    Returns:
        epochs: (n_epochs, 30, trial_samples)
        labels: (n_epochs,) binary (0 = neutral, 1 = positive)
    """
    raw = load_deap_subject(subj_id, deap_dir)
    data = raw["data"]          # (40, 40, 8064)
    labels = raw["labels"]      # (40, 4)

    # Only first 30 EEG channels (exclude 8 peripheral channels)
    data = data[:, :N_CHANNELS, :]

    # Extract valence and binarize
    valence = labels[:, 0]                    # (40,)
    threshold = PRETRAIN["label_threshold"]
    bin_labels = (valence > threshold).astype(np.int64)  # 1=positive, 0=neutral

    # Drop ambiguous trials (valence == threshold)
    valid = valence != threshold
    data = data[valid]
    bin_labels = bin_labels[valid]

    all_epochs, all_labels = [], []
    for i in range(len(data)):
        trial_data = data[i]  # (30, 8064)
        trial_data = trial_data.T  # (8064, 30) for segment_epochs

        # Bandpass filter
        trial_data = bandpass_filter(trial_data, DEAP_SFREQ,
                                     BANDPASS_LOW, BANDPASS_HIGH)

        # Segment into smaller epochs
        epochs = segment_epochs(trial_data, DEAP_SFREQ,
                                TRIAL_LENGTH_SEC, STRIDE_SEC)
        all_epochs.append(epochs)
        all_labels.extend([bin_labels[i]] * len(epochs))

    all_epochs = np.concatenate(all_epochs, axis=0)
    all_labels = np.array(all_labels)

    # Standardize per channel
    all_epochs = standardize_epochs(all_epochs)

    return all_epochs.astype(np.float32), all_labels.astype(np.int64)


def build_deap_dataset(deap_dir: Path = None, cache: bool = True) -> tuple:
    """
    Load and preprocess all DEAP subjects.

    Returns:
        X: (n_total_epochs, 30, trial_samples)
        y: (n_total_epochs,)
        subject_ids: (n_total_epochs,) which subject each epoch belongs to
    """
    from src.config import PROCESSED_DIR

    cache_file = PROCESSED_DIR / "deap_processed.npz"
    if cache and cache_file.exists():
        print(f"[DEAP] Loading cached data from {cache_file}")
        cached = np.load(cache_file)
        return cached["X"], cached["y"], cached["subject_ids"]

    X_list, y_list, subj_list = [], [], []
    for subj_id in range(1, 33):
        try:
            X_s, y_s = process_deap_subject(subj_id, deap_dir)
            X_list.append(X_s)
            y_list.append(y_s)
            subj_list.append(np.full(len(y_s), subj_id, dtype=np.int16))
            print(f"  DEAP s{subj_id:02d}: {len(y_s)} epochs, "
                  f"pos={y_s.mean():.2%}")
        except FileNotFoundError:
            print(f"  DEAP s{subj_id:02d}: file not found, skipping...")

    if not X_list:
        raise RuntimeError("No DEAP data loaded. Place .dat files in data/DEAP/")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    subj = np.concatenate(subj_list, axis=0)

    if cache:
        np.savez_compressed(cache_file, X=X, y=y, subject_ids=subj)
        print(f"[DEAP] Cached {len(y)} epochs to {cache_file}")

    return X, y, subj


if __name__ == "__main__":
    # Test: load and print stats
    try:
        X, y, subj = build_deap_dataset()
        print(f"\nTotal: {X.shape}, pos ratio: {y.mean():.3f}")
        print(f"Subjects: {np.unique(subj)}")
    except RuntimeError as e:
        print(f"DEAP data not available: {e}")
