"""
Evaluate EEGNet + LightGBM ensemble with GroupKFold CV.
Fast: LightGBM trains in seconds, EEGNet ~3min/fold.
"""
import sys, argparse, numpy as np, torch, time, joblib
from pathlib import Path
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import CHECKPOINT_DIR, FINETUNE, EEGNET, COMP_SFREQ, N_CHANNELS
from src.models.eegnet import EEGNet
from src.data.competition_dataset import build_competition_dataset
from src.data.preprocessing import extract_de_features, extract_psd_features
from torch.utils.data import DataLoader, TensorDataset


def extract_hjorth(epochs):
    n, c, _ = epochs.shape
    f = np.zeros((n, c * 3))
    for i in range(n):
        for j in range(c):
            x = epochs[i, j]; d1 = np.diff(x); d2 = np.diff(d1)
            act = np.var(x)
            mob = np.sqrt(np.var(d1) / max(act, 1e-12))
            comp = np.sqrt(np.var(d2) / max(np.var(d1), 1e-12)) / max(mob, 1e-12)
            f[i, 0*c+j] = act; f[i, 1*c+j] = mob; f[i, 2*c+j] = comp
    return f


def train_eegnet_fold(X_tr, y_tr, X_va, y_va, n_times, device):
    """Train EEGNet on one fold, return validation predictions."""
    model = EEGNet(N_CHANNELS, n_times, 2, EEGNET["F1"], EEGNET["D"],
                   EEGNET["dropout"]).to(device)
    ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    loader = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc, patience = 0, 0
    for epoch in range(1, 101):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward(); opt.step()
        # Validate
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(X_va).to(device))
            acc = (logits.argmax(1).cpu() == torch.from_numpy(y_va)).float().mean().item()
        if acc > best_acc:
            best_acc = acc; patience = 0
        else:
            patience += 1
            if patience >= 15: break
    # Return probs
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(X_va).to(device)), 1).cpu().numpy()
    return probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_folds", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y, subj_ids, _ = build_competition_dataset()
    n_times = X.shape[-1]

    # Extract LightGBM features on all data
    print("Extracting features for LightGBM...")
    de = extract_de_features(X, COMP_SFREQ)
    psd = extract_psd_features(X, COMP_SFREQ)
    hj = extract_hjorth(X)
    X_feat = np.concatenate([de, psd, hj], axis=1).astype(np.float32)

    gkf = GroupKFold(n_splits=args.n_folds)
    fold_accs_eeg = []
    fold_accs_lgb = []
    fold_accs_ens = []

    for fold, (tr, va) in enumerate(gkf.split(X, y, groups=subj_ids), 1):
        print(f"\nFold {fold}:")
        # LightGBM
        import lightgbm as lgb
        lgb_model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05,
                                       num_leaves=63, max_bin=255, min_child_samples=20,
                                       subsample=0.8, colsample_bytree=0.8,
                                       reg_alpha=0.1, reg_lambda=0.1, random_state=42,
                                       n_jobs=-1, force_col_wise=True, verbose=-1)
        lgb_model.fit(X_feat[tr], y[tr],
                      eval_set=[(X_feat[va], y[va])],
                      eval_metric="binary_logloss",
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        lgb_probs = lgb_model.predict_proba(X_feat[va])
        lgb_acc = (lgb_probs.argmax(1) == y[va]).mean()

        # EEGNet
        print(f"  Training EEGNet...")
        eeg_probs = train_eegnet_fold(X[tr], y[tr], X[va], y[va], n_times, device)
        eeg_acc = (eeg_probs.argmax(1) == y[va]).mean()

        # Ensemble
        ens_probs = (eeg_probs + lgb_probs) / 2.0
        ens_acc = (ens_probs.argmax(1) == y[va]).mean()

        fold_accs_eeg.append(eeg_acc)
        fold_accs_lgb.append(lgb_acc)
        fold_accs_ens.append(ens_acc)
        print(f"  EEGNet={eeg_acc:.2%}  LGB={lgb_acc:.2%}  Ensemble={ens_acc:.2%}")

    print(f"\nEEGNet:     {[f'{a:.2%}' for a in fold_accs_eeg]}")
    print(f"LightGBM:   {[f'{a:.2%}' for a in fold_accs_lgb]}")
    print(f"Ensemble:   {[f'{a:.2%}' for a in fold_accs_ens]}")
    print(f"EEGNet Mean={np.mean(fold_accs_eeg):.2%}  LGB Mean={np.mean(fold_accs_lgb):.2%}  Ensemble Mean={np.mean(fold_accs_ens):.2%}")


if __name__ == "__main__":
    main()
