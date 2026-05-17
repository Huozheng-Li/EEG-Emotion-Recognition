"""
Test set inference: EEGNet + LightGBM ensemble → submission xlsx.
"""
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import joblib
import openpyxl

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import (
    CHECKPOINT_DIR, COMP_SFREQ, N_CHANNELS, TRIAL_LENGTH_SEC, STRIDE_SEC,
    BANDPASS_LOW, BANDPASS_HIGH, TEST_DIR, ROOT, EEGNET, TSCEPTION,
)
from src.models.eegnet import EEGNet
from src.data.preprocessing import (
    bandpass_filter, segment_epochs, standardize_epochs,
    extract_de_features, extract_psd_features, BANDS, BAND_NAMES,
)


def extract_hjorth(epochs):
    """Fast Hjorth extraction matching lightgbm_baseline."""
    n_epochs, n_ch, _ = epochs.shape
    feats = np.zeros((n_epochs, n_ch * 3))
    for i in range(n_epochs):
        for c in range(n_ch):
            x = epochs[i, c, :]
            d1 = np.diff(x)
            d2 = np.diff(d1)
            act = np.var(x)
            mob = np.sqrt(np.var(d1) / max(act, 1e-12))
            comp = np.sqrt(np.var(d2) / max(np.var(d1), 1e-12)) / max(mob, 1e-12)
            feats[i, 0 * n_ch + c] = act
            feats[i, 1 * n_ch + c] = mob
            feats[i, 2 * n_ch + c] = comp
    return feats


def preprocess_test_subject(filepath: Path) -> np.ndarray:
    """
    Load and preprocess one test subject.
    Test data: 8 video clips × 10s = 20000 samples.
    We split each 10s clip into 2 non-overlapping 5s sub-epochs,
    then average predictions per clip to get 8 trial predictions.
    Returns: (16, 30, 1250) — 16 five-second epochs.
    """
    import scipy.io as sio
    mat = sio.loadmat(filepath)
    for k, v in mat.items():
        if not k.startswith("__"):
            data = v.astype(np.float64)
            break
    # data: (30, 20000)
    data = bandpass_filter(data, COMP_SFREQ, BANDPASS_LOW, BANDPASS_HIGH)
    # Cut into 16 fixed 5s segments (no overlap): 20000 / 1250 = 16
    n_times = data.shape[1]
    trial_samples = 1250  # Hard-coded 5s at 250Hz (model was trained on 5s)
    epochs = []
    for start in range(0, n_times, trial_samples):
        epoch = data[:, start:start + trial_samples]  # (30, 1250)
        if epoch.shape[1] == trial_samples:
            epochs.append(epoch)
    epochs = np.stack(epochs, axis=0)  # (16, 30, 1250)
    epochs = standardize_epochs(epochs)
    return epochs.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eegnet_ckpt", type=str,
                        default=str(CHECKPOINT_DIR / "eegnet_competition_final.pt"))
    parser.add_argument("--lgb_ckpt", type=str,
                        default=str(CHECKPOINT_DIR / "lightgbm_baseline.pkl"))
    parser.add_argument("--output", type=str,
                        default=str(ROOT / "submission.xlsx"))
    parser.add_argument("--template", type=str,
                        default=str(ROOT / "PROBLEM" / "测试结果模板.xlsx"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load models ─────────────────────────────────────────────
    print("Loading models...")
    # EEGNet
    eegnet = EEGNet(n_channels=N_CHANNELS, n_times=1250,  # 5s at 250Hz — must match training
                    n_classes=2, F1=EEGNET["F1"], D=EEGNET["D"],
                    dropout=EEGNET["dropout"]).to(device)
    ckpt = torch.load(args.eegnet_ckpt, map_location=device, weights_only=True)
    eegnet.load_state_dict(ckpt["model_state_dict"])
    eegnet.eval()

    # LightGBM
    lgb_data = joblib.load(args.lgb_ckpt)
    lgb_model = lgb_data["model"]
    feature_names = lgb_data["feature_names"]

    # ── Process each test subject ───────────────────────────────
    test_files = sorted(TEST_DIR.glob("P_test*.mat"))
    print(f"Test subjects: {len(test_files)}")

    rows = []
    for fpath in tqdm(test_files, desc="Processing test subjects", ncols=80):
        user_id = fpath.stem  # e.g., "P_test1"
        epochs = preprocess_test_subject(fpath)  # (16, 30, 1250) — 8 trials × 2 halves

        # EEGNet predictions
        X_tensor = torch.from_numpy(epochs).to(device)
        with torch.no_grad():
            logits = eegnet(X_tensor)
            eegnet_probs = torch.softmax(logits, dim=1).cpu().numpy()  # (16, 2)

        # LightGBM predictions
        de = extract_de_features(epochs, COMP_SFREQ)
        psd = extract_psd_features(epochs, COMP_SFREQ)
        hj = extract_hjorth(epochs)
        feats = np.concatenate([de, psd, hj], axis=1).astype(np.float32)
        lgb_probs = lgb_model.predict_proba(feats)  # (16, 2)

        # Soft voting ensemble on each 5s sub-epoch
        ensemble_probs = (eegnet_probs + lgb_probs) / 2.0  # (16, 2)

        # Average every 2 consecutive sub-epochs → 8 trial probabilities
        trial_pos_probs = ensemble_probs.reshape(8, 2, 2).mean(axis=1)[:, 1]  # (8,)

        # Each test subject has exactly 4 positive + 4 negative videos.
        # Pick top 4 highest positive-prob trials as positive.
        preds = np.zeros(8, dtype=int)
        top4_idx = np.argsort(trial_pos_probs)[-4:]  # indices of 4 highest
        preds[top4_idx] = 1

        for t in range(8):
            rows.append([user_id, t + 1, int(preds[t])])

    # ── Write submission ────────────────────────────────────────
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["user_id", "trial_id", "Emotion_label"])
    for row in rows:
        ws.append(row)
    wb.save(args.output)

    # Stats
    labels = np.array([r[2] for r in rows])
    print(f"\nSubmission: {len(rows)} trials (8 per subject × 10 subjects) → {args.output}")
    print(f"Predicted: pos={labels.mean():.1%} ({labels.sum()}/{len(labels)})")


if __name__ == "__main__":
    main()
