"""
EEG preprocessing: bandpass filtering, segmentation, standardization.
Works on both DEAP (128 Hz) and Competition (250 Hz) data.
"""
import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler

# EEG frequency bands
BANDS = [(1, 4), (4, 8), (8, 14), (14, 31), (31, 50)]
BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]


def bandpass_filter(data: np.ndarray, sfreq: float,
                    low: float = 0.5, high: float = 50.0) -> np.ndarray:
    """
    Apply Butterworth bandpass filter.

    Args:
        data: (n_channels, n_times) or (n_times, n_channels)
        sfreq: sampling frequency in Hz
        low, high: cutoff frequencies
    """
    nyq = sfreq / 2.0
    b, a = signal.butter(4, [low / nyq, high / nyq], btype="band")

    if data.shape[0] < data.shape[1]:  # (channels, times) → filter along time
        return signal.filtfilt(b, a, data, axis=-1)
    else:  # (times, channels) → filter along time
        return signal.filtfilt(b, a, data, axis=0)


def segment_epochs(data: np.ndarray, sfreq: float,
                   trial_len: float = 10.0, stride: float = 5.0,
                   ensure_2d: bool = True) -> np.ndarray:
    """
    Segment continuous EEG into overlapping trials.

    Args:
        data: (n_times, n_channels) or (n_channels, n_times)
        sfreq: sampling rate
        trial_len: trial duration in seconds
        stride: stride between trials in seconds

    Returns:
        epochs: (n_epochs, n_channels, trial_samples)
    """
    # Input: (n_times, n_channels). If channels > times, transpose.
    if data.shape[1] > data.shape[0]:
        data = data.T
    n_times, n_ch = data.shape
    trial_samples = int(trial_len * sfreq)
    stride_samples = int(stride * sfreq)

    epochs = []
    for start in range(0, n_times - trial_samples + 1, stride_samples):
        epoch = data[start:start + trial_samples, :].T  # (n_ch, trial_samples)
        epochs.append(epoch)

    return np.stack(epochs, axis=0)  # (n_epochs, n_ch, trial_samples)


def standardize_epochs(epochs: np.ndarray) -> np.ndarray:
    """
    Z-score standardize each epoch per channel.

    Args:
        epochs: (n_epochs, n_channels, n_times)
    Returns:
        standardized epochs, same shape
    """
    n_epochs, n_ch, n_times = epochs.shape
    out = np.zeros_like(epochs)
    for i in range(n_epochs):
        for c in range(n_ch):
            ch_data = epochs[i, c, :]
            mu, std = ch_data.mean(), ch_data.std()
            if std > 1e-8:
                out[i, c, :] = (ch_data - mu) / std
    return out


def extract_de_features(epochs: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Extract Differential Entropy (DE) features per frequency band.
    DE = 1/2 * log(2πeσ²) per band per channel.

    Bands: delta (1-4), theta (4-8), alpha (8-14), beta (14-31), gamma (31-50)

    Args:
        epochs: (n_epochs, n_channels, n_times)
        sfreq: sampling rate
    Returns:
        features: (n_epochs, n_channels * 5)
    """
    bands = [(1, 4), (4, 8), (8, 14), (14, 31), (31, 50)]
    n_epochs, n_ch, _ = epochs.shape
    features = np.zeros((n_epochs, n_ch * len(bands)))

    for i in range(n_epochs):
        for j, (low, high) in enumerate(bands):
            filtered = bandpass_filter(epochs[i], sfreq, low, high)
            var = filtered.var(axis=-1)
            var = np.maximum(var, 1e-12)
            de = 0.5 * np.log(2 * np.pi * np.e * var)
            features[i, j * n_ch:(j + 1) * n_ch] = de

    return features


def extract_psd_features(epochs: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Extract PSD band power features using Welch's method.

    Returns:
        features: (n_epochs, n_channels * 5)
    """
    bands = [(1, 4), (4, 8), (8, 14), (14, 31), (31, 50)]
    n_epochs, n_ch, _ = epochs.shape
    features = np.zeros((n_epochs, n_ch * len(bands)))

    for i in range(n_epochs):
        for c in range(n_ch):
            freqs, psd = signal.welch(epochs[i, c], fs=sfreq, nperseg=min(256, epochs.shape[-1]))
            for j, (low, high) in enumerate(bands):
                mask = (freqs >= low) & (freqs <= high)
                features[i, j * n_ch + c] = np.log(psd[mask].sum() + 1e-12)

    return features
