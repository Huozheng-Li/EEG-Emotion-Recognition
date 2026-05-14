"""
Pretrain TSception on DEAP dataset.

Saves checkpoint to checkpoints/tsception_deap_pretrain.pt for
subsequent fine-tuning on the competition dataset.
"""
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import (
    CHECKPOINT_DIR, PRETRAIN, TSCEPTION, COMP_SFREQ, N_CHANNELS,
)
from src.models.tsception import TSception
from src.data.deap_dataset import build_deap_dataset


def create_dataloaders(X, y, batch_size: int, val_split: float,
                       num_workers: int = 0):
    """Split into train/val and create DataLoaders."""
    # Stratified split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, stratify=y, random_state=42
    )

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deap_dir", type=str, default=None,
                        help="Path to DEAP .dat files")
    parser.add_argument("--epochs", type=int, default=PRETRAIN["epochs"])
    parser.add_argument("--batch_size", type=int, default=PRETRAIN["batch_size"])
    parser.add_argument("--lr", type=float, default=PRETRAIN["lr"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PRETRAIN] Device: {device}")

    # ── Load DEAP ───────────────────────────────────────────────
    deap_dir = Path(args.deap_dir) if args.deap_dir else None
    X, y, _ = build_deap_dataset(deap_dir)

    n_times = X.shape[-1]
    print(f"[PRETRAIN] DEAP data: {X.shape}, pos={y.mean():.3f}")

    train_loader, val_loader = create_dataloaders(
        X, y, args.batch_size, PRETRAIN["val_split"], PRETRAIN["num_workers"]
    )

    # ── Model ───────────────────────────────────────────────────
    model = TSception(
        n_channels=N_CHANNELS,
        n_times=n_times,
        num_T=TSCEPTION["num_T"],
        num_S=TSCEPTION["num_S"],
        hid_channels=TSCEPTION["hid_channels"],
        num_classes=TSCEPTION["num_classes"],
        dropout=TSCEPTION["dropout"],
        sampling_rate=128,  # DEAP sampling rate
    ).to(device)

    print(f"[PRETRAIN] Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── Optimizer & Loss ────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=PRETRAIN["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion = nn.CrossEntropyLoss()

    # ── Training Loop ───────────────────────────────────────────
    best_val_acc = 0
    patience_counter = 0
    best_path = CHECKPOINT_DIR / "tsception_deap_pretrain.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer,
                                            criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs} | "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.3%} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.3%} | "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "config": TSCEPTION,
            }, best_path)
        else:
            patience_counter += 1
            if patience_counter >= PRETRAIN["patience"]:
                print(f"[PRETRAIN] Early stopping at epoch {epoch}")
                break

    print(f"[PRETRAIN] Best val acc: {best_val_acc:.3%}")
    print(f"[PRETRAIN] Model saved to {best_path}")


if __name__ == "__main__":
    main()
