"""
Quick train final model on all competition data (skip CV).
Usage: python -m src.train.train_final --model eegnet [--pretrained path/to/ckpt.pt]
"""
import sys, argparse, torch, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import (
    CHECKPOINT_DIR, FINETUNE, TSCEPTION, EEGNET, COMP_SFREQ, N_CHANNELS,
)
from src.models.eegnet import EEGNet
from src.models.tsception import TSception
from src.data.competition_dataset import build_competition_dataset
from src.train.finetune import train_epoch
from torch.utils.data import DataLoader, TensorDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="eegnet", choices=["tsception","eegnet"])
    parser.add_argument("--pretrained", type=str, default="none")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=FINETUNE["lr"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y, _, _ = build_competition_dataset()
    n_times = X.shape[-1]

    if args.model == "eegnet":
        model = EEGNet(N_CHANNELS, n_times, 2, EEGNET["F1"], EEGNET["D"], EEGNET["dropout"])
    else:
        model = TSception(N_CHANNELS, n_times, TSCEPTION["num_T"], TSCEPTION["num_S"],
                          TSCEPTION["hid_channels"], 2, TSCEPTION["dropout"], COMP_SFREQ)
    model = model.to(device)

    if args.pretrained != "none":
        ckpt = torch.load(args.pretrained, map_location=device, weights_only=True)
        pdict = ckpt["model_state_dict"]
        mdict = model.state_dict()
        compat = {k: v for k, v in pdict.items() if k in mdict and v.shape == mdict[k].shape and not k.startswith("classifier")}
        mdict.update(compat)
        model.load_state_dict(mdict)
        print(f"Loaded {len(compat)} pretrained layers")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} | Params: {n_params:,} | Epochs: {args.epochs}")

    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=FINETUNE["weight_decay"])
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        loss, acc = train_epoch(model, loader, opt, criterion, device, f"Final E{epoch:02d}")
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}: loss={loss:.4f} acc={acc:.2%}")

    out = CHECKPOINT_DIR / f"{args.model}_competition_final.pt"
    torch.save({"model_state_dict": model.state_dict()}, out)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
