"""
Fine-tune TSception on competition dataset.
Cross-subject validation (StratifiedKFold).
"""
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, GroupKFold
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import (
    CHECKPOINT_DIR, FINETUNE, TSCEPTION, EEGNET, COMP_SFREQ, N_CHANNELS,
)
from src.models.tsception import TSception
from src.models.eegnet import EEGNet


def build_model(model_name: str, n_channels: int, n_times: int, num_classes: int,
                sampling_rate: int):
    if model_name == "eegnet":
        return EEGNet(n_channels=n_channels, n_times=n_times, n_classes=num_classes,
                      F1=EEGNET["F1"], D=EEGNET["D"], dropout=EEGNET["dropout"])
    else:
        return TSception(n_channels=n_channels, n_times=n_times,
                         num_T=TSCEPTION["num_T"], num_S=TSCEPTION["num_S"],
                         hid_channels=TSCEPTION["hid_channels"],
                         num_classes=num_classes, dropout=TSCEPTION["dropout"],
                         sampling_rate=sampling_rate)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
from src.data.competition_dataset import build_competition_dataset


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


def run_fold(fold: int, train_idx, val_idx, X, y, pretrained_path: Path,
             device, args, model_name: str = "tsception"):
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True,
                              num_workers=FINETUNE["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=FINETUNE["num_workers"], pin_memory=True)

    n_times = X.shape[-1]
    model = build_model(model_name, N_CHANNELS, n_times,
                        TSCEPTION["num_classes"], COMP_SFREQ).to(device)

    n_params = count_parameters(model)

    if pretrained_path and pretrained_path.exists():
        ckpt = torch.load(pretrained_path, map_location=device, weights_only=True)
        pretrained_dict = ckpt["model_state_dict"]
        model_dict = model.state_dict()
        compatible = {k: v for k, v in pretrained_dict.items()
                      if k in model_dict and v.shape == model_dict[k].shape
                      and not k.startswith("classifier")}
        model_dict.update(compatible)
        model.load_state_dict(model_dict)
        ckpt_info = f"pretrained={pretrained_path.name} ({len(compatible)} layers)"
    else:
        ckpt_info = "pretrained=none (from scratch)"

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=FINETUNE["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    print(f"\n{'═'*80}")
    print(f" Fold {fold}/{args.n_folds} | {ckpt_info}")
    print(f" Train: {len(train_ds)} trials | Val: {len(val_ds)} trials")
    print(f" Params: {n_params:,} | Batch: {args.batch_size} | "
          f"LR: {args.lr} | GPU: {gpu_mem()}")
    print(f"{'─'*80}")
    print(f"{'Epoch':>5} {'train_loss':>10} {'train_acc':>10} "
          f"{'val_loss':>10} {'val_acc':>10} {'lr':>8}  {'best':>10}")
    print(f"{'─'*80}")

    best_val_acc = 0
    patience_counter = 0
    t_start = time.time()
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            pbar_desc=f"Fold{fold} E{epoch:02d} train")
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device,
            pbar_desc=f"Fold{fold} E{epoch:02d} val  ")
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        marker = " *" if val_acc > best_val_acc else ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"{epoch:5d} {train_loss:10.4f} {train_acc:9.2%} "
              f"{val_loss:10.4f} {val_acc:9.2%} "
              f"{scheduler.get_last_lr()[0]:8.2e}  {best_val_acc:9.2%}{marker}")

        if patience_counter >= FINETUNE["patience"]:
            print(f"  Early stopping @ epoch {epoch}")
            break

    elapsed = time.time() - t_start
    print(f"{'─'*80}")
    print(f" Fold {fold} done | best_val_acc={best_val_acc:.2%} | "
          f"time={elapsed/60:.1f}min")

    # Save training history for plotting
    logs_dir = CHECKPOINT_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(logs_dir / f"finetune_fold{fold}.npz",
                        train_loss=history["train_loss"],
                        train_acc=history["train_acc"],
                        val_loss=history["val_loss"],
                        val_acc=history["val_acc"],
                        best_val_acc=best_val_acc)
    print(f"  History saved to {logs_dir / f'finetune_fold{fold}.npz'}")
    print(f"{'═'*80}")
    return best_val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str,
                        default=str(CHECKPOINT_DIR / "tsception_deap_pretrain.pt"),
                        help="Path to pretrained weights (set to 'none' to skip)")
    parser.add_argument("--model", type=str, default="tsception",
                        choices=["tsception", "eegnet"])
    parser.add_argument("--epochs", type=int, default=FINETUNE["epochs"])
    parser.add_argument("--batch_size", type=int, default=FINETUNE["batch_size"])
    parser.add_argument("--lr", type=float, default=FINETUNE["lr"])
    parser.add_argument("--n_folds", type=int, default=FINETUNE["n_folds"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device} ({gpu_name})")

    # ── Load data ────────────────────────────────────────────────
    X, y, subj_ids, groups = build_competition_dataset()
    n_subjects = len(np.unique(subj_ids))
    print(f"Data: {X.shape[0]} trials x {X.shape[1]}ch x {X.shape[2]}pt | "
          f"{n_subjects} subjects | pos={y.mean():.1%}")
    print(f"HC subjects: {(groups==0).astype(int).sum()//(X.shape[0]//n_subjects)} | "
          f"DEP subjects: {(groups==1).astype(int).sum()//(X.shape[0]//n_subjects)}")

    # ── Cross-validation ─────────────────────────────────────────
    pretrained_path = Path(args.pretrained) if args.pretrained.lower() != "none" else None

    gkf = GroupKFold(n_splits=args.n_folds)
    fold_accs = []

    print(f"\n{'═'*80}")
    print(f" Starting {args.n_folds}-fold subject-wise cross-validation")
    print(f" Epochs: {args.epochs} | Patience: {FINETUNE['patience']} | "
          f"Batch: {args.batch_size}")
    print(f"{'═'*80}")

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=subj_ids), 1):
        val_subjs = np.unique(subj_ids[val_idx])
        print(f"  Fold {fold}: val subjects = {val_subjs.tolist()}")
        acc = run_fold(fold, train_idx, val_idx, X, y,
                       pretrained_path, device, args, args.model)
        fold_accs.append(acc)

    print(f"\n{'═'*80}")
    print(f" CV Results: {args.n_folds}-fold")
    print(f" Per-fold: {[f'{a:.2%}' for a in fold_accs]}")
    print(f" Mean: {np.mean(fold_accs):.2%}  Std: {np.std(fold_accs):.2%}")
    print(f"{'═'*80}")

    # ── Train final model ────────────────────────────────────────
    print("\nTraining final model on all data for inference...")
    n_times = X.shape[-1]
    final_model = build_model(args.model, N_CHANNELS, n_times,
                              TSCEPTION["num_classes"], COMP_SFREQ).to(device)

    if pretrained_path and pretrained_path.exists():
        ckpt = torch.load(pretrained_path, map_location=device, weights_only=True)
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
                             num_workers=FINETUNE["num_workers"], pin_memory=True)

    optimizer = torch.optim.AdamW(final_model.parameters(), lr=args.lr,
                                  weight_decay=FINETUNE["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        loss, acc = train_epoch(final_model, full_loader, optimizer, criterion,
                                device, pbar_desc=f"Final E{epoch:02d}")
        scheduler.step()
        if epoch % 10 == 0:
            print(f"  Final Epoch {epoch:3d}: loss={loss:.4f} acc={acc:.2%}")

    final_path = CHECKPOINT_DIR / f"{args.model}_competition_final.pt"
    torch.save({"model_state_dict": final_model.state_dict(), "config": TSCEPTION}, final_path)
    print(f"Final model saved to {final_path}")


if __name__ == "__main__":
    main()
