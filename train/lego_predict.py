#!/usr/bin/env python3
# Copyright (c) 2026 Eoin Jordan
# SPDX-License-Identifier: Apache-2.0
"""Run BrickNet on the host for pictures captured from the board.

Prints the DET line of the fp32 model and of the quantised model for each
PNG, next to the line the board printed for the same frame when the folder
has a detections.txt from tools/lego_capture.py. The board runs the int8
program, so its line should agree with the quantised model here up to
rounding. --sheet writes the pictures with the cells the quantised model
marked, to look at.

    python train/lego_predict.py data/lego/real/background --sheet sheet.png
"""

import argparse
import pathlib
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import lego_synth  # noqa: E402
from export_lego import CALIBRATION_SEED, NHWC, quantise  # noqa: E402
from lego_model import SIDE, THRESHOLD, BrickNet, decode, det_line  # noqa: E402
from train_lego import STATE, predict, synth, tensors  # noqa: E402


def same(a: str, b: str) -> bool:
    """Two DET lines name the same objects within 0.02, the time aside."""
    ta = [t for t in a.split() if not t.startswith("ms=")]
    tb = [t for t in b.split() if not t.startswith("ms=")]
    if len(ta) != len(tb):
        return False
    for u, v in zip(ta, tb):
        try:
            if abs(float(u) - float(v)) > 0.021:
                return False
        except ValueError:
            if u != v:
                return False
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", type=pathlib.Path, nargs="+", help="PNG files or folders of them")
    p.add_argument("--state-dict", type=pathlib.Path, default=STATE)
    p.add_argument("--sheet", type=pathlib.Path, help="write the annotated pictures to this PNG")
    args = p.parse_args()

    files = []
    for path in args.path:
        files += sorted(path.glob("*.png")) if path.is_dir() else [path]
    if not files:
        sys.exit("no PNG files")

    device = {}
    for folder in {f.parent for f in files}:
        log = folder / "detections.txt"
        if log.exists():
            for line in log.read_text().splitlines():
                name, _, det = line.partition(" ")
                device[folder / name] = det

    net = BrickNet()
    net.load_state_dict(torch.load(args.state_dict, map_location="cpu"))
    model = net.fused()
    xc, _ = tensors(synth(500, CALIBRATION_SEED, 4))
    converted = quantise(net.fused().to(memory_format=NHWC), xc)

    images = [np.asarray(Image.open(f).convert("RGB").resize((SIDE, SIDE), Image.BOX))
              for f in files]
    x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous()
    z = predict(model, x)
    zq = predict(converted, x, nhwc=True)

    agree = compared = 0
    tiles = []
    for f, img, a, b in zip(files, images, z, zq):
        line = det_line(*decode(a))
        line_q = det_line(*decode(b))
        print(f"{f}")
        print(f"  fp32   {line}")
        print(f"  int8   {line_q}")
        if f in device:
            compared += 1
            ok = same(device[f], line_q)
            agree += ok
            print(f"  board  {device[f]}{'' if ok else '   <-- differs from int8'}")
        e = np.exp(b - b.max(axis=0, keepdims=True))
        prob = e / e.sum(axis=0, keepdims=True)
        cells = prob.argmax(axis=0).astype(np.uint8)
        cells[prob.max(axis=0) < THRESHOLD] = 0
        tiles.append(lego_synth.annotate(img, cells))
    if compared:
        print(f"\nboard and host int8 agree on {agree} of {compared} frames")

    if args.sheet:
        cols = min(8, len(tiles))
        rows = (len(tiles) + cols - 1) // cols
        w = tiles[0].width
        sheet = Image.new("RGB", (cols * w, rows * w))
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % cols) * w, (i // cols) * w))
        args.sheet.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(args.sheet)
        print(f"wrote {args.sheet}")


if __name__ == "__main__":
    main()
