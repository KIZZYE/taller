"""Entrena y evalúa uno o ambos modelos desde la terminal.

    python train.py --model mask2former --epochs 30
    python train.py --model effps --epochs 40
    python train.py --model both
"""
import argparse

import torch

from models.effps_model import EffPS
from models.mask2former_model import Mask2FormerPS
from src.config import DATA_DIR, IMG_SIZE, OUTPUTS_DIR
from src.data import make_loaders
from src.trainer import fit, test_best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mask2former", "effps", "both"], default="both")
    ap.add_argument("--data", default=str(DATA_DIR))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    ap.add_argument("--patience", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Dispositivo:", device)
    _, loaders = make_loaders(args.data, args.batch, args.img_size, splits_json=OUTPUTS_DIR / "splits.json")

    todo = ["mask2former", "effps"] if args.model == "both" else [args.model]
    for name in todo:
        cls = Mask2FormerPS if name == "mask2former" else EffPS
        model = cls.from_pretrained(device) if name == "mask2former" else EffPS.build(device, img_size=args.img_size)
        fit(model, loaders, device, epochs=args.epochs, lr=args.lr, patience=args.patience)
        test_best(cls, loaders, device, name)


if __name__ == "__main__":
    main()
