"""
Fine-tune TSception on competition dataset.

Uses pretrained DEAP weights, then fine-tunes with
cross-subject validation (LOSO or k-fold).
"""
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import (
    CHECKPOINT_DIR, FINETUNE, TSCEPTION, COMP_SFREQ, N_CHANNELS,
)
from src.models.tsception import TSception
from src.data.competition_dataset import build_competition_dataset


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


def run_fold(fold: int, train_idx, val_idx, X, y, pretrained_path: Path,
             device, args):
    """Train/eval a single CV fold."""
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True,
                              num_workers=FINETUNE["num_workers"],
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=FINETUNE["num_workers"],
                            pin_memory=True)

    n_times = X.shape[-1]
    model = TSception(
        n_channels=N_CHANNELS, n_times=n_times,
        num_T=TSCEPTION["num_T"], num_S=TSCEPTION["num_S"],
        hid_channels=TSCEPTION["hid_channels"],
        num_classes=TSCEPTION["num_classes"],
        dropout=TSCEPTION["dropout"],
        sampling_rate=COMP_SFREQ,
    ).to(device)

    # Load pretrained weights
    if pretrained_path and pretrained_path.exists():
        ckpt = torch.load(pretrained_path, map_location=device,
                          weights_only=True)
        # Filter out classifier weights (they may differ)
        pretrained_dict = ckpt["model_state_dict"]
        model_dict = model.state_dict()
        compatible = {k: v for k, v in pretrained_dict.items()
                      if k in model_dict and v.shape == model_dict[k].shape
                      and not k.startswith("classifier")}
        model_dict.update(compatible)
        model.load_state_dict(model_dict)
        print(f"  Fold {fold}: Loaded {len(compatible)}/{len(model_dict)} "
              f"layers from pretrained weights")
    else:
        print(f"  Fold {fold}: Training from scratch (no pretrained weights)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=FINETUNE["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer,
                                            criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch}/{args.epochs}: "
                  f"train={train_loss:.3f}/{train_acc:.1%} "
                  f"val={val_loss:.3f}/{val_acc:.1%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= FINETUNE["patience"]:
                break

    return best_val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str,
                        default=str(CHECKPOINT_DIR / "tsception_deap_pretrain.pt"),
                        help="Path to pretrained weights (set to 'none' to skip)")
    parser.add_argument("--epochs", type=int, default=FINETUNE["epochs"])
    parser.add_argument("--batch_size", type=int, default=FINETUNE["batch_size"])
    parser.add_argument("--lr", type=float, default=FINETUNE["lr"])
    parser.add_argument("--n_folds", type=int, default=FINETUNE["n_folds"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FINETUNE] Device: {device}")

    # ── Load competition data ───────────────────────────────────
    X, y, subj_ids, groups = build_competition_dataset()
    n_subjects = len(np.unique(subj_ids))
    print(f"[FINETUNE] Data: {X.shape}, subjects: {n_subjects}, "
          f"pos={y.mean():.3f}")

    # ── Cross-validation ────────────────────────────────────────
    pretrained_path = Path(args.pretrained) if args.pretrained.lower() != "none" else None

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    fold_accs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        acc = run_fold(fold, train_idx, val_idx, X, y,
                       pretrained_path, device, args)
        fold_accs.append(acc)
        print(f"  Fold {fold}: val_acc={acc:.3%}")

    print(f"\n[FINETUNE] {args.n_folds}-fold CV: "
          f"mean={np.mean(fold_accs):.3%} "
          f"std={np.std(fold_accs):.3%}")

    # ── Train final model on full dataset ───────────────────────
    print("\n[FINETUNE] Training final model on all data...")
    n_times = X.shape[-1]
    final_model = TSception(
        n_channels=N_CHANNELS, n_times=n_times,
        num_T=TSCEPTION["num_T"], num_S=TSCEPTION["num_S"],
        hid_channels=TSCEPTION["hid_channels"],
        num_classes=TSCEPTION["num_classes"],
        dropout=TSCEPTION["dropout"],
        sampling_rate=COMP_SFREQ,
    ).to(device)

    if pretrained_path and pretrained_path.exists():
        ckpt = torch.load(pretrained_path, map_location=device,
                          weights_only=True)
        pretrained_dict = ckpt["model_state_dict"]
        model_dict = final_model.state_dict()
        compatible = {k: v for k, v in pretrained_dict.items()
                      if k in model_dict and v.shape == model_dict[k].shape
                      and not k.startswith("classifier")}
        model_dict.update(compatible)
        final_model.load_state_dict(model_dict)

    full_ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    full_loader = DataLoader(full_ds, batch_size=args.batch_size,
                             shuffle=True, drop_last=True,
                             num_workers=FINETUNE["num_workers"],
                             pin_memory=True)

    optimizer = torch.optim.AdamW(final_model.parameters(), lr=args.lr,
                                  weight_decay=FINETUNE["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        loss, acc = train_epoch(final_model, full_loader, optimizer,
                                criterion, device)
        scheduler.step()
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}: train_loss={loss:.4f} train_acc={acc:.3%}")

    final_path = CHECKPOINT_DIR / "tsception_competition_final.pt"
    torch.save({
        "model_state_dict": final_model.state_dict(),
        "config": TSCEPTION,
    }, final_path)
    print(f"[FINETUNE] Final model saved to {final_path}")


if __name__ == "__main__":
    main()
