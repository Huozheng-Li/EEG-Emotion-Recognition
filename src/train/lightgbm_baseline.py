"""
Handcrafted EEG features + LightGBM baseline.

Extracts DE, PSD, Hjorth, and asymmetry features from EEG epochs,
then trains LightGBM with GroupKFold subject-wise cross-validation.
"""
import sys
import argparse
import time
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import (
    CHECKPOINT_DIR, PROCESSED_DIR, COMP_SFREQ, N_CHANNELS, FINETUNE,
)
from src.data.competition_dataset import build_competition_dataset
from src.data.preprocessing import (
    extract_de_features, extract_psd_features, BANDS, BAND_NAMES,
)


def extract_features(X: np.ndarray, sfreq: float) -> tuple:
    """
    Extract comprehensive features from EEG epochs.

    Args:
        X: (n_epochs, n_channels, n_times)
        sfreq: sampling rate

    Returns:
        features: (n_epochs, n_features) concatenated feature vector
        feature_names: list of feature name strings
    """
    n_epochs, n_ch, _ = X.shape
    print(f"  Extracting DE features...")
    de = extract_de_features(X, sfreq)   # (n_epochs, n_ch * 5)
    de_names = [f"DE_{b}_{c}" for b in BAND_NAMES for c in range(1, n_ch + 1)]

    print(f"  Extracting PSD features...")
    psd = extract_psd_features(X, sfreq)  # (n_epochs, n_ch * 5)
    psd_names = [f"PSD_{b}_{c}" for b in BAND_NAMES for c in range(1, n_ch + 1)]

    print(f"  Extracting Hjorth features...")
    hj = extract_hjorth_features(X)       # (n_epochs, n_ch * 3)
    hj_names = [f"Hjorth_{p}_{c}" for p in ["act", "mob", "comp"]
                for c in range(1, n_ch + 1)]

    feats = np.concatenate([de, psd, hj], axis=1)
    names = de_names + psd_names + hj_names

    return feats.astype(np.float32), names


def extract_hjorth_features(epochs: np.ndarray) -> np.ndarray:
    """
    Hjorth parameters: Activity, Mobility, Complexity.

    Activity  = var(signal)
    Mobility  = std(derivative) / std(signal)
    Complexity = Mobility(derivative) / Mobility(signal)

    Returns:
        (n_epochs, n_channels * 3)
    """
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


def train_fold(fold, train_idx, val_idx, X_feat, y, model_params):
    """Train LightGBM on one fold, return validation accuracy."""
    import lightgbm as lgb
    X_tr, y_tr = X_feat[train_idx], y[train_idx]
    X_va, y_va = X_feat[val_idx], y[val_idx]

    model = lgb.LGBMClassifier(**model_params, verbose=-1)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)],
    )
    acc = model.score(X_va, y_va)
    return acc, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_folds", type=int, default=FINETUNE["n_folds"])
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--num_leaves", type=int, default=63)
    parser.add_argument("--max_bin", type=int, default=255)
    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────────
    print("=" * 60)
    print("LightGBM + Handcrafted EEG Features Baseline")
    print("=" * 60)

    X_raw, y, subj_ids, _ = build_competition_dataset()
    n_subjects = len(np.unique(subj_ids))
    n_epochs = X_raw.shape[0]
    print(f"Data: {n_epochs} epochs x {X_raw.shape[1]}ch x {X_raw.shape[2]}pt")
    print(f"Subjects: {n_subjects}, pos ratio: {y.mean():.1%}")

    # ── Extract features (with caching) ──────────────────────────
    cache_file = PROCESSED_DIR / "competition_features.npz"
    if cache_file.exists():
        print(f"Loading cached features from {cache_file}")
        cached = np.load(cache_file)
        X_feat, feature_names = cached["X"], list(cached["names"])
    else:
        print("Extracting features...")
        t0 = time.time()
        X_feat, feature_names = extract_features(X_raw, COMP_SFREQ)
        print(f"  Done in {time.time() - t0:.1f}s | "
              f"Features: {X_feat.shape[1]}")
        np.savez_compressed(
            cache_file,
            X=X_feat,
            names=np.array(feature_names),
        )
        print(f"  Cached to {cache_file}")

    # ── GroupKFold CV ────────────────────────────────────────────
    lgb_params = {
        "n_estimators": args.n_estimators,
        "learning_rate": args.lr,
        "num_leaves": args.num_leaves,
        "max_bin": args.max_bin,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "force_col_wise": True,
        "verbosity": -1,
    }

    gkf = GroupKFold(n_splits=args.n_folds)
    fold_accs = []

    print(f"\nStarting {args.n_folds}-fold subject-wise CV")
    print(f"n_estimators={args.n_estimators}, lr={args.lr}, "
          f"num_leaves={args.num_leaves}")

    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(X_feat, y, groups=subj_ids), 1
    ):
        val_subjs = np.unique(subj_ids[val_idx])
        print(f"  Fold {fold}: val subjects = {val_subjs.tolist()}")

        acc, model = train_fold(fold, train_idx, val_idx,
                                X_feat, y, lgb_params)
        fold_accs.append(acc)

        # Top features
        importance = model.feature_importances_
        top_idx = np.argsort(importance)[-10:][::-1]
        top_str = ", ".join(
            f"{feature_names[i]}({importance[i]:.0f})" for i in top_idx
        )
        print(f"    acc={acc:.2%} | top features: {top_str}")

    print(f"\nCV: {[f'{a:.2%}' for a in fold_accs]}")
    print(f"Mean: {np.mean(fold_accs):.2%}  Std: {np.std(fold_accs):.2%}")

    # ── Train full model & save ─────────────────────────────────
    print("\nTraining final model on all data...")
    import lightgbm as lgb
    final_model = lgb.LGBMClassifier(**lgb_params, verbose=-1)
    final_model.fit(X_feat, y)
    train_acc = final_model.score(X_feat, y)
    print(f"  Train acc: {train_acc:.2%}")

    import joblib
    model_path = CHECKPOINT_DIR / "lightgbm_baseline.pkl"
    joblib.dump({"model": final_model, "feature_names": feature_names},
                model_path)
    print(f"  Saved to {model_path}")


if __name__ == "__main__":
    main()
