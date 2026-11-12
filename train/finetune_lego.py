#!/usr/bin/env python3
# Copyright (c) 2026 Eoin Jordan
# SPDX-License-Identifier: Apache-2.0
"""Fine-tune BrickNet on pictures from the board, mixed with synthetic ones.

The pictures come from tools/lego_capture.py, one folder per label:

    data/lego/real/background/   every cell is background
    data/lego/real/yellow/       cells are background or yellow, and at least
    data/lego/real/white/        one cell is the colour of the folder; where
    data/lego/real/red/          exactly is left to the network
    data/lego/real/bin/
    data/lego/real/mixed/        used only with a points file

A folder name is a weak label: it says what is in the picture, not where.
For exact targets put a text file next to a picture, same name with .txt,
with one object a line as "class x y [radius]", in picture coordinates
from 0 to 1 (class is yellow, white, red or bin; radius defaults to one
cell). Cells inside the radius take the class, a ring around it is left
out of the loss, the rest is background. A picture with a points file is
trained on its points whatever folder it is in.

Half of every batch is synthetic, so the network keeps what it learned.

    python train/finetune_lego.py                # fine-tune, then export
    python train/finetune_lego.py --epochs 0     # only score the real pictures
    python train/export_lego.py

The real pictures are also the training pictures here; the scores printed
for them are not a measure of accuracy on new ones. Keep some folders back
(--real with another path and --epochs 0) for that.
"""

import argparse
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import lego_synth  # noqa: E402
from lego_model import CLASSES, GRID, SIDE, BrickNet, decode  # noqa: E402
from train_lego import (STATE, augment, loss_fn, predict, report, synth, tensors,  # noqa: E402
                        to_input, validation_set)

REPO = pathlib.Path(__file__).resolve().parents[1]
FOLDERS = ("background", "yellow", "white", "red", "bin", "mixed")
NAME_TO_CLASS = {"yellow": 1, "white": 2, "red": 3, "bin": 4}
WEAK = 254      # cell label: background or the class of the folder


def cells_from_points(path: pathlib.Path) -> np.ndarray:
    cells = np.zeros((GRID, GRID), np.uint8)
    centre = (np.arange(GRID) + 0.5) / GRID
    for line in path.read_text().splitlines():
        tok = line.split()
        if not tok or tok[0].startswith("#"):
            continue
        cls = NAME_TO_CLASS[tok[0]]
        x, y = float(tok[1]), float(tok[2])
        r = float(tok[3]) if len(tok) > 3 else 0.5 / GRID
        dist = np.hypot(centre[None, :] - x, centre[:, None] - y)
        ring = (dist <= r + 1.0 / GRID) & (cells == 0)
        cells[ring] = lego_synth.IGNORE
        cells[dist <= max(r, 0.71 / GRID)] = cls
    return cells


def load_real(root: pathlib.Path):
    """Pictures, cell labels (with WEAK and IGNORE), folder class and path."""
    images, cells, folder_cls, paths = [], [], [], []
    for name in FOLDERS:
        for png in sorted((root / name).glob("*.png")):
            img = Image.open(png).convert("RGB")
            if img.size != (SIDE, SIDE):
                img = img.resize((SIDE, SIDE), Image.BOX)
            points = png.with_suffix(".txt")
            if points.exists():
                c = cells_from_points(points)
            elif name == "background":
                c = np.zeros((GRID, GRID), np.uint8)
            elif name == "mixed":
                print(f"skipping {png}: a mixed picture needs {points.name}")
                continue
            else:
                c = np.full((GRID, GRID), WEAK, np.uint8)
            images.append(np.asarray(img))
            cells.append(c)
            folder_cls.append(NAME_TO_CLASS.get(name, 0))
            paths.append(png)
    if not images:
        return None
    x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous()
    return x, torch.from_numpy(np.stack(cells)).long(), torch.tensor(folder_cls), paths


def real_loss(logits: torch.Tensor, y: torch.Tensor, folder_cls: torch.Tensor) -> torch.Tensor:
    """Cross entropy where the cells are known, the folder label where they are not."""
    weak = y == WEAK
    exact = y.clone()
    exact[weak] = lego_synth.IGNORE
    loss = torch.zeros(())
    if (exact != lego_synth.IGNORE).any():
        loss = loss + loss_fn(logits, exact)
    logp = F.log_softmax(logits, dim=1)
    for i in range(len(y)):
        c = int(folder_cls[i])
        if c == 0 or not weak[i].any():
            continue
        lp = logp[i][:, weak[i]]                                  # (5, cells)
        allowed = torch.logsumexp(lp[[0, c]], dim=0)              # background or the colour
        loss = loss + (-allowed.mean() - lp[c].max()) / len(y)    # and the colour somewhere
    return loss


def score_real(model, real) -> None:
    """Per folder: pictures where the model sees what the folder says and nothing else."""
    x, _, folder_cls, paths = real
    logits = predict(model, x)
    rows = {}
    for z, c, path in zip(logits, folder_cls.tolist(), paths):
        bricks, the_bin = decode(z)
        seen = {d.cls for d in bricks} | ({4} if the_bin is not None else set())
        name = path.parent.name
        row = rows.setdefault(name, [0, 0, 0])
        row[0] += 1
        if name == "mixed":
            continue
        row[1] += int(seen == ({c} if c else set()))
        row[2] += int(bool(seen - {c}))
    print("\nreal pictures (the folder label against what the model finds)")
    print("  folder      pictures  as labelled  other class found")
    for name, (n, ok, other) in rows.items():
        print(f"  {name:10s}  {n:8d}  {ok:11d}  {other:17d}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real", type=pathlib.Path, default=REPO / "data" / "lego" / "real")
    p.add_argument("--state-dict", type=pathlib.Path, default=STATE, help="weights to start from")
    p.add_argument("--out", type=pathlib.Path, default=STATE, help="weights to write")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--synthetic", type=int, default=4000, help="synthetic pictures to mix in")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--val", type=int, default=1000)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()

    real = load_real(args.real)
    if real is None:
        sys.exit(f"no pictures under {args.real}; capture some with tools/lego_capture.py")
    xr, yr, cr, _ = real
    print(f"{len(xr)} real pictures from {args.real}")

    model = BrickNet()
    model.load_state_dict(torch.load(args.state_dict, map_location="cpu"))
    model.eval()
    print("\n== before ==")
    score_real(model, real)
    if args.epochs == 0:
        return

    xs, ys = tensors(synth(args.synthetic, 550001, args.workers))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    half = args.batch // 2
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(xs))
        total = 0.0
        for i in range(0, len(xs), half):
            idx = order[i:i + half]
            ridx = torch.randint(0, len(xr), (half,))
            xb, yb = augment(to_input(xs[idx]), ys[idx])
            xrb, yrb = augment(to_input(xr[ridx]), yr[ridx])
            opt.zero_grad()
            out = model(torch.cat([xb, xrb]))
            loss = loss_fn(out[:len(idx)], yb) + real_loss(out[len(idx):], yrb, cr[ridx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        print(f"epoch {epoch + 1}  loss {total / len(xs):.4f}", flush=True)

    model.eval()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"wrote {args.out}")
    print("\n== after ==")
    score_real(model, real)
    val = validation_set(args.workers, args.val)
    xv, _ = tensors(val)
    report(predict(model, xv), val)
    print("\nnow run train/export_lego.py and rebuild the firmware")


if __name__ == "__main__":
    main()
