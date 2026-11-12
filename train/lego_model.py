# Copyright (c) 2026 Eoin Jordan
# SPDX-License-Identifier: Apache-2.0
"""BrickNet: a fully convolutional centroid detector for bricks by colour.

In the style of FOMO: the network reduces a 96 x 96 RGB picture by a factor
of eight and gives every cell of the 12 x 12 grid a class, background,
yellow, white, red or bin. Touching cells of one class make one object,
and its centroid is the target for the arm. There are no boxes and no
anchors: the network is a stack of plain and depthwise convolutions and the
decoding is a flood fill.

The firmware does the same decoding in firmware/src/lego.cpp; keep the
two in step, together with the constants below.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

SIDE = 96
GRID = 12
CLASSES = ("background", "yellow", "white", "red", "bin")
NUM_CLASSES = len(CLASSES)
BIN = 4
INPUT_SHAPE = (1, 3, SIDE, SIDE)

# Decoding. A cell counts when its class has at least this probability.
THRESHOLD = 0.6
MAX_BRICKS = 8


def conv_bn(cin: int, cout: int, kernel: int, stride: int, groups: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel, stride, kernel // 2, groups=groups, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(),
    )


class BrickNet(nn.Module):
    """Pixels scaled to 0..1 in, (N, 5, 12, 12) logits out."""

    def __init__(self, width: int = 16) -> None:
        super().__init__()
        w = width
        self.features = nn.Sequential(
            conv_bn(3, w // 2, 3, 2),                 # 96 -> 48
            conv_bn(w // 2, w // 2, 3, 2, w // 2),    # 48 -> 24, depthwise
            conv_bn(w // 2, w, 1, 1),
            conv_bn(w, w, 3, 2, w),                   # 24 -> 12, depthwise
            conv_bn(w, 2 * w, 1, 1),
            # Three rounds at 12 x 12: cheap there, and each widens what a
            # cell sees by two cells, which is what tells a white brick
            # with its shadow from a sheet of paper.
            conv_bn(2 * w, 2 * w, 3, 1, 2 * w),       # depthwise
            conv_bn(2 * w, 2 * w, 1, 1),
            conv_bn(2 * w, 2 * w, 3, 1, 2 * w),
            conv_bn(2 * w, 2 * w, 1, 1),
            conv_bn(2 * w, 2 * w, 3, 1, 2 * w),
            conv_bn(2 * w, 2 * w, 1, 1),
        )
        self.head = nn.Conv2d(2 * w, NUM_CLASSES, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

    def fused(self) -> "BrickNet":
        """A copy for export with every batch norm folded into its convolution."""
        import copy

        from torch.nn.utils.fusion import fuse_conv_bn_eval

        net = copy.deepcopy(self).eval()
        for i, block in enumerate(net.features):
            net.features[i] = nn.Sequential(fuse_conv_bn_eval(block[0], block[1]), nn.ReLU())
        return net


@dataclass
class Detection:
    cls: int
    x: float       # 0..1 from the left of the picture
    y: float       # 0..1 from the top
    conf: float    # highest cell probability of the object
    cells: int

    @property
    def name(self) -> str:
        return "bin" if self.cls == BIN else CLASSES[self.cls]


def decode(logits: np.ndarray, threshold: float = THRESHOLD):
    """(5, 12, 12) logits to (bricks, bin). bin is None or one Detection."""
    z = logits - logits.max(axis=0, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=0, keepdims=True)
    cls = p.argmax(axis=0)
    conf = p.max(axis=0)
    cls[conf < threshold] = 0

    seen = np.zeros((GRID, GRID), bool)
    found = []
    for gy in range(GRID):
        for gx in range(GRID):
            c = int(cls[gy, gx])
            if c == 0 or seen[gy, gx]:
                continue
            stack = [(gy, gx)]
            seen[gy, gx] = True
            sw = sx = sy = best = 0.0
            n = 0
            while stack:
                y, x = stack.pop()
                w = float(conf[y, x])
                sw += w
                sx += w * (x + 0.5)
                sy += w * (y + 0.5)
                best = max(best, w)
                n += 1
                # Eight neighbours: a brick lying across the grid touches
                # its next cell by a corner as often as by a side.
                for ny in range(max(y - 1, 0), min(y + 2, GRID)):
                    for nx in range(max(x - 1, 0), min(x + 2, GRID)):
                        if not seen[ny, nx] and cls[ny, nx] == c:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            found.append(Detection(c, sx / sw / GRID, sy / sw / GRID, best, n))

    bins = [d for d in found if d.cls == BIN]
    bricks = sorted((d for d in found if d.cls != BIN), key=lambda d: -d.conf)[:MAX_BRICKS]
    the_bin = max(bins, key=lambda d: (d.cells, d.conf)) if bins else None
    return bricks, the_bin


def det_line(bricks, the_bin, ms: int | None = None) -> str:
    """The line the firmware prints for a frame."""
    parts = [f"DET n={len(bricks)}"]
    for d in bricks:
        parts.append(f"{d.name} {d.x:.2f} {d.y:.2f} {d.conf:.2f}")
    if the_bin is not None:
        parts.append(f"bin {the_bin.x:.2f} {the_bin.y:.2f} {the_bin.conf:.2f}")
    if ms is not None:
        parts.append(f"ms={ms}")
    return " ".join(parts)
