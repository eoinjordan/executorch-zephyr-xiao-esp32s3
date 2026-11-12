#!/usr/bin/env python3
# Copyright (c) 2026 Eoin Jordan
# SPDX-License-Identifier: Apache-2.0
"""Train BrickNet on synthetic desk scenes and score it on a held-out set.

Nothing is downloaded and nothing is stored: the pictures come from
train/lego_synth.py and stay in memory (about 28 KB each).

    python train/train_lego.py                 # about ten minutes on a laptop CPU
    python train/export_lego.py                # .pte files and the self-test header

Writes train/out/lego_bricknet.pt. The scores printed at the end are for
synthetic pictures drawn by the same generator with another seed. They say
that the network learned the generator, not how it does on a real desk.
"""

import argparse
import multiprocessing
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import lego_synth  # noqa: E402
from lego_model import BIN, CLASSES, GRID, NUM_CLASSES, BrickNet, decode  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
STATE = REPO / "train" / "out" / "lego_bricknet.pt"
VAL_SEED = 990001
VAL_COUNT = 2000


def _chunk(args):
    count, seed = args
    return lego_synth.make_set(count, seed)


def synth(count: int, seed: int, workers: int):
    """count samples; the result depends on seed only, not on the worker count."""
    per = 250
    jobs = [(min(per, count - i), seed + i) for i in range(0, count, per)]
    if workers <= 1:
        parts = [_chunk(j) for j in jobs]
    else:
        with multiprocessing.Pool(workers) as pool:
            parts = pool.map(_chunk, jobs)
    return [s for part in parts for s in part]


def tensors(samples):
    x = torch.from_numpy(np.stack([s.image for s in samples])).permute(0, 3, 1, 2).contiguous()
    y = torch.from_numpy(np.stack([s.cells for s in samples])).long()
    return x, y


def to_input(x_u8: torch.Tensor) -> torch.Tensor:
    return x_u8.float() / 255.0


def augment(x: torch.Tensor, y: torch.Tensor):
    """Flips and quarter turns, the same for a picture and its cells."""
    if torch.rand(1).item() < 0.5:
        x, y = x.flip(3), y.flip(2)
    if torch.rand(1).item() < 0.5:
        x, y = x.flip(2), y.flip(1)
    if torch.rand(1).item() < 0.5:
        x, y = x.transpose(2, 3), y.transpose(1, 2)
    return x, y


CLASS_WEIGHT = torch.tensor([1.0, 3.0, 4.0, 3.0, 2.0])


def loss_fn(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, y, weight=CLASS_WEIGHT, ignore_index=lego_synth.IGNORE)


@torch.no_grad()
def predict(module, x_u8: torch.Tensor, batch: int = 250, nhwc: bool = False) -> np.ndarray:
    """Logits (N, 5, 12, 12) of a module for uint8 NCHW pictures."""
    out = []
    for i in range(0, len(x_u8), batch):
        x = to_input(x_u8[i:i + batch])
        if nhwc:
            x = x.to(memory_format=torch.channels_last)
        out.append(module(x).float().numpy())
    return np.concatenate(out)


def score(logits: np.ndarray, samples, keep=None) -> dict:
    """Cell and object scores per class over the samples keep() accepts.

    A decoded object is right when it is the nearest unmatched object of its
    class and its centroid is within 1.5 cells of the true one, or within
    half the size of the object if that is more.
    """
    cell_tp = np.zeros(NUM_CLASSES)
    cell_fp = np.zeros(NUM_CLASSES)
    cell_fn = np.zeros(NUM_CLASSES)
    tp = np.zeros(NUM_CLASSES)
    fp = np.zeros(NUM_CLASSES)
    fn = np.zeros(NUM_CLASSES)
    err = [[] for _ in range(NUM_CLASSES)]
    used = 0
    for z, s in zip(logits, samples):
        if keep is not None and not keep(s):
            continue
        used += 1
        pred = z.argmax(axis=0)
        valid = s.cells != lego_synth.IGNORE
        for c in range(1, NUM_CLASSES):
            cell_tp[c] += ((pred == c) & (s.cells == c)).sum()
            cell_fp[c] += ((pred == c) & (s.cells != c) & valid).sum()
            cell_fn[c] += ((pred != c) & (s.cells == c)).sum()

        bricks, the_bin = decode(z)
        dets = bricks + ([the_bin] if the_bin is not None else [])
        truth = list(s.objects)
        taken = [False] * len(truth)
        for d in dets:
            best, best_dist = -1, 1e9
            for i, o in enumerate(truth):
                if taken[i] or o.cls != d.cls:
                    continue
                dist = float(np.hypot(d.x - o.x, d.y - o.y))
                limit = max(1.5 / GRID, 0.5 * np.sqrt(o.area) / lego_synth.SIDE)
                if dist <= limit and dist < best_dist:
                    best, best_dist = i, dist
            if best >= 0:
                taken[best] = True
                tp[d.cls] += 1
                err[d.cls].append(best_dist)
            else:
                gy = min(int(d.y * GRID), GRID - 1)
                gx = min(int(d.x * GRID), GRID - 1)
                # A detection on a cell the labels leave open (a sliver of
                # a brick, the edge of one) is neither right nor wrong.
                if s.cells[gy, gx] != lego_synth.IGNORE:
                    fp[d.cls] += 1
        for i, o in enumerate(truth):
            if not taken[i]:
                fn[o.cls] += 1

    def ratio(a, b):
        return float(a / (a + b)) if a + b > 0 else float("nan")

    rows = {}
    for c in range(1, NUM_CLASSES):
        rows[CLASSES[c]] = {
            "cell_precision": ratio(cell_tp[c], cell_fp[c]),
            "cell_recall": ratio(cell_tp[c], cell_fn[c]),
            "precision": ratio(tp[c], fp[c]),
            "recall": ratio(tp[c], fn[c]),
            "objects": int(tp[c] + fn[c]),
            "centroid_error": float(np.mean(err[c])) if err[c] else float("nan"),
        }
    rows["pictures"] = used
    return rows


def print_scores(title: str, rows: dict) -> None:
    print(f"\n{title} ({rows['pictures']} pictures)")
    print("  class   objects  precision  recall  cell-prec  cell-rec  centroid error")
    for name in CLASSES[1:]:
        r = rows[name]
        print(f"  {name:7s} {r['objects']:7d}  {r['precision']:9.3f}  {r['recall']:6.3f}"
              f"  {r['cell_precision']:9.3f}  {r['cell_recall']:8.3f}"
              f"  {r['centroid_error'] * GRID:6.2f} cells")


def report(logits: np.ndarray, samples) -> None:
    """All of the held-out set, then the two cases that are hard for colour."""
    print_scores("held-out synthetic set", score(logits, samples))
    print_scores("pale desk only (white bricks on white)",
                 score(logits, samples, lambda s: s.bg == "pale"))
    print_scores("warm light only (white turns yellow)",
                 score(logits, samples, lambda s: s.cast == "warm"))
    print_scores("pale desk under warm light",
                 score(logits, samples, lambda s: s.bg == "pale" and s.cast == "warm"))


def validation_set(workers: int, count: int = VAL_COUNT):
    return synth(count, VAL_SEED, workers)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", type=int, default=16000, help="synthetic training pictures")
    p.add_argument("--val", type=int, default=VAL_COUNT, help="held-out synthetic pictures")
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=4e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=min(4, multiprocessing.cpu_count() or 1))
    p.add_argument("--state-dict", type=pathlib.Path, default=STATE)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    t0 = time.time()
    train = synth(args.train, 1000 + args.seed * 100000, args.workers)
    val = validation_set(args.workers, args.val)
    x, y = tensors(train)
    xv, _ = tensors(val)
    print(f"{len(train)} training and {len(val)} held-out pictures in {time.time() - t0:.0f} s")

    model = BrickNet()
    print(f"BrickNet: {sum(p.numel() for p in model.parameters())} parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * ((len(x) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)

    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(x))
        total = 0.0
        for i in range(0, len(x), args.batch):
            idx = order[i:i + args.batch]
            xb, yb = augment(to_input(x[idx]), y[idx])
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        model.eval()
        rows = score(predict(model, xv[:500]), val[:500])
        recall = " ".join(f"{n} {rows[n]['recall']:.2f}/{rows[n]['precision']:.2f}"
                          for n in CLASSES[1:])
        print(f"epoch {epoch + 1:2d}  loss {total / len(x):.4f}  recall/precision: {recall}"
              f"  ({time.time() - t0:.0f} s)", flush=True)

    model.eval()
    args.state_dict.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.state_dict)
    print(f"wrote {args.state_dict}")
    report(predict(model, xv), val)


if __name__ == "__main__":
    main()
