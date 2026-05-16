"""
Pretrain TSception on DEAP dataset.
Saves to checkpoints/tsception_deap_pretrain.pt for fine-tuning.
"""
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import (
    CHECKPOINT_DIR, PRETRAIN, TSCEPTION, COMP_SFREQ, N_CHANNELS, DEAP_SFREQ,
)
from src.models.tsception import TSception, count_parameters
from src.data.deap_dataset import build_deap_dataset


def gpu_mem() -> str:
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"{used:.1f}G/{total:.0f}G"
    return "N/A"


def train_epoch(model, loader, optimizer, criterion, device, pbar_desc="train"):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc=pbar_desc, leave=False, ncols=100)
    for X_batch, y_batch in pbar:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += len(y_batch)
        pbar.set_postfix({"loss": f"{loss.item():.3f}", "acc": f"{correct / total:.2%}"})
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, pbar_desc="val"):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc=pbar_desc, leave=False, ncols=100)
    for X_batch, y_batch in pbar:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += len(y_batch)
        pbar.set_postfix({"loss": f"{loss.item():.3f}", "acc": f"{correct / total:.2%}"})
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
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device} ({gpu_name})")

    # ── Load DEAP ───────────────────────────────────────────────
    deap_dir = Path(args.deap_dir) if args.deap_dir else None
    X, y, _ = build_deap_dataset(deap_dir)

    n_times = X.shape[-1]
    print(f"DEAP data: {X.shape[0]} epochs x {X.shape[1]}ch x {X.shape[2]}pt | "
          f"pos={y.mean():.1%}")

    train_loader, val_loader = create_dataloaders(
        X, y, args.batch_size, PRETRAIN["val_split"], PRETRAIN["num_workers"]
    )

    # ── Model ───────────────────────────────────────────────────
    model = TSception(
        n_channels=N_CHANNELS, n_times=n_times,
        num_T=TSCEPTION["num_T"], num_S=TSCEPTION["num_S"],
        hid_channels=TSCEPTION["hid_channels"],
        num_classes=TSCEPTION["num_classes"],
        dropout=TSCEPTION["dropout"],
        sampling_rate=DEAP_SFREQ,
    ).to(device)

    n_params = count_parameters(model)
    print(f"Model: TSception, {n_params:,} params | Batch: {args.batch_size} | "
          f"LR: {args.lr} | GPU: {gpu_mem()}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=PRETRAIN["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # ── Training ────────────────────────────────────────────────
    best_val_acc = 0
    patience_counter = 0
    best_path = CHECKPOINT_DIR / "tsception_deap_pretrain.pt"

    print(f"\n{'─'*60}")
    print(f"{'Epoch':>5} {'train_loss':>10} {'train_acc':>10} "
          f"{'val_loss':>10} {'val_acc':>10} {'lr':>8}  {'best':>10}")
    print(f"{'─'*60}")

    t_start = time.time()
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            pbar_desc=f"Pretrain E{epoch:02d}")
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device,
            pbar_desc=f"PreVal   E{epoch:02d}")
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        marker = " *" if val_acc > best_val_acc else ""
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

        print(f"{epoch:5d} {train_loss:10.4f} {train_acc:9.2%} "
              f"{val_loss:10.4f} {val_acc:9.2%} "
              f"{scheduler.get_last_lr()[0]:8.2e}  {best_val_acc:9.2%}{marker}")

        if patience_counter >= PRETRAIN["patience"]:
            print(f"Early stopping @ epoch {epoch}")
            break

    elapsed = time.time() - t_start
    print(f"{'─'*60}")
    print(f"Done in {elapsed/60:.1f}min | Best val_acc: {best_val_acc:.2%}")
    print(f"Saved to {best_path}")

    # Save training history for plotting
    logs_dir = CHECKPOINT_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(logs_dir / "pretrain_history.npz",
                        epochs=np.arange(1, len(history["train_loss"]) + 1),
                        train_loss=history["train_loss"],
                        train_acc=history["train_acc"],
                        val_loss=history["val_loss"],
                        val_acc=history["val_acc"])
    print(f"History saved to {logs_dir / 'pretrain_history.npz'}")


def create_dataloaders(X, y, batch_size, val_split, num_workers=0):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, stratify=y, random_state=42
    )
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


if __name__ == "__main__":
    main()
