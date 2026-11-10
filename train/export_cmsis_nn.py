#!/usr/bin/env python3
# Copyright (c) 2026 Eoin Jordan
# SPDX-License-Identifier: Apache-2.0
"""Quantise TinyDigitNet to int8 and export it for the CMSIS-NN kernels.

The program goes through ExecuTorch's Cortex-M quantizer and pass manager, so
its convolutions, pooling and linear layers become cortex_m operators that
call CMSIS-NN. CMSIS-NN has three code paths: Helium (MVE), the DSP extension
and plain C. The plain C one ("scalar") needs nothing from Arm, which is what
lets the same kernels build for the Xtensa LX7 of the ESP32-S3. The export
targets that path, because buffer sizes are fixed for a path at export time.

Reads (written by train/train_mnist.py):
  train/out/mnist_tiny.pt            trained weights
  train/data/MNIST/raw/              the MNIST files

Writes:
  firmware/model/mnist_tiny_cmsis_nn.pte   int8 program, cortex_m operators
  firmware/src/test_input_q.h              reference logits of the quantised
                                           model for the image in test_input.h

Usage:
  python train/train_mnist.py        # once, to train and download MNIST
  python train/export_cmsis_nn.py
"""

import argparse
import gzip
import pathlib
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import INPUT_SHAPE, MEAN, STD, TinyDigitNet  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]

# CMSIS-NN convolutions and pooling work on NHWC data. The quantizer only
# hands a convolution to them when the model is in PyTorch's channels-last
# memory format; in the default format it stays a float convolution.
NHWC = torch.channels_last


def example_input():
    return (torch.zeros(*INPUT_SHAPE).to(memory_format=NHWC),)


def read_idx(path: pathlib.Path) -> np.ndarray:
    """An MNIST IDX file (plain or gzipped) as a uint8 array."""
    if not path.exists() and path.with_name(path.name + ".gz").exists():
        path = path.with_name(path.name + ".gz")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        data = f.read()
    _, _, dtype, ndim = struct.unpack(">BBBB", data[:4])
    if dtype != 0x08:
        raise ValueError(f"{path}: expected unsigned bytes")
    shape = struct.unpack(">" + "I" * ndim, data[4 : 4 + 4 * ndim])
    return np.frombuffer(data, np.uint8, offset=4 + 4 * ndim).reshape(shape)


def load_mnist(raw: pathlib.Path, split: str):
    """Images normalised as in training, NCHW float32, and labels."""
    images = read_idx(raw / f"{split}-images-idx3-ubyte")
    labels = read_idx(raw / f"{split}-labels-idx1-ubyte")
    x = (torch.from_numpy(images.copy()).float() / 255.0 - MEAN) / STD
    return x[:, None], torch.from_numpy(labels.copy()).long()


@torch.no_grad()
def accuracy(module, x, y, batch: int = 500) -> float:
    correct = 0
    for i in range(0, len(x), batch):
        out = module(x[i : i + batch].to(memory_format=NHWC))
        correct += (out.argmax(1) == y[i : i + batch]).sum().item()
    return correct / len(x)


def quantise(model, calibration):
    """PT2E quantisation with the Cortex-M quantizer. Returns the converted module."""
    from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    with torch.no_grad():
        exported = torch.export.export(model, example_input(), strict=False)
        try:
            # The guard that .module() adds cannot pass through the edge passes.
            captured = exported.module(check_guards=False)
        except TypeError:
            captured = exported.module()
        prepared = prepare_pt2e(captured, CortexMQuantizer())
        for i in range(0, len(calibration), 100):
            prepared(calibration[i : i + 100].to(memory_format=NHWC))
        return convert_pt2e(prepared)


def export_pte(converted, out: pathlib.Path):
    """The quantised module as an ExecuTorch program with cortex_m operators."""
    from executorch.backends.cortex_m.edge_compile_config import cortex_m_edge_compile_config
    from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager
    from executorch.backends.cortex_m.target_config import CortexM, CortexMTargetConfig
    from executorch.exir import to_edge

    # A Cortex-M0+ has neither DSP nor Helium, so this selects the plain C
    # kernels of CMSIS-NN, the ones that also compile for Xtensa.
    target = CortexMTargetConfig(cpu=CortexM.M0PLUS)
    with torch.no_grad():
        exported = torch.export.export(converted, example_input(), strict=False)
        edge = to_edge(exported, compile_config=cortex_m_edge_compile_config())
        edge._edge_programs["forward"] = CortexMPassManager(
            edge.exported_program(), target_config=target
        ).transform()
        program = edge.to_executorch()

    ops = sorted(
        {
            str(n.target)
            for n in program.exported_program().graph.nodes
            if n.op == "call_function" and hasattr(n.target, "_schema")
        }
    )
    print(f"CMSIS-NN code path: {target.backend.name}")
    print("operators in program:")
    for op in ops:
        print(f"  {op}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(program.buffer)
    print(f"wrote {out} ({out.stat().st_size} bytes)")


def output_step(converted) -> float:
    """The quantisation step of the logits: the scale of the last dequantize."""
    scale = None
    for node in converted.graph.nodes:
        if node.op == "call_function" and "dequantize_per_tensor" in str(node.target):
            scale = float(node.args[1])
    if scale is None:
        raise RuntimeError("the quantised graph has no dequantize node")
    return scale


def write_header(converted, image, label: int, index: int, out: pathlib.Path) -> None:
    """Reference logits of the quantised model for the self-test image."""
    with torch.no_grad():
        logits = converted(image[None].to(memory_format=NHWC))[0]
    step = output_step(converted)
    lines = [
        "/* Generated by train/export_cmsis_nn.py. Do not edit.",
        f" * MNIST test index {index}, label {label}: the image in test_input.h.",
        " * Logits of the int8 model, computed in PyTorch from the quantised graph.",
        " * The integer kernels on the device may differ from them by a rounding",
        " * step of the output scale, never by more than a few.",
        " */",
        "#pragma once",
        "",
        "/* One step of the int8 output: logits are multiples of this. */",
        f"#define TEST_LOGIT_STEP_Q {step:.6f}f",
        "",
        "static const float kTestExpectedLogitsQ[10] = {",
        "    " + ", ".join(f"{v:.6f}f" for v in logits.tolist()) + ",",
        "};",
        "",
    ]
    out.write_text("\n".join(lines), newline="\n")
    print(f"wrote {out} (label={label}, argmax={int(logits.argmax())}, output step {step:.6f})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state-dict", type=pathlib.Path, default=REPO / "train" / "out" / "mnist_tiny.pt")
    p.add_argument("--data-dir", type=pathlib.Path, default=REPO / "train" / "data")
    p.add_argument("--pte", type=pathlib.Path, default=REPO / "firmware" / "model" / "mnist_tiny_cmsis_nn.pte")
    p.add_argument("--header", type=pathlib.Path, default=REPO / "firmware" / "src" / "test_input_q.h")
    p.add_argument("--calibration", type=int, default=1000, help="training images used to set the ranges")
    p.add_argument("--test-index", type=int, default=0, help="must match train_mnist.py")
    args = p.parse_args()

    raw = args.data_dir / "MNIST" / "raw"
    if not args.state_dict.exists() or not raw.exists():
        sys.exit("run train/train_mnist.py first: it trains the model and downloads MNIST")

    torch.manual_seed(0)
    model = TinyDigitNet()
    model.load_state_dict(torch.load(args.state_dict, map_location="cpu"))
    model = model.eval().to(memory_format=NHWC)

    train_x, _ = load_mnist(raw, "train")
    test_x, test_y = load_mnist(raw, "t10k")

    converted = quantise(model, train_x[: args.calibration])
    acc_f = accuracy(model, test_x, test_y)
    acc_q = accuracy(converted, test_x, test_y)
    print(f"test accuracy: fp32 {acc_f:.4f}, int8 {acc_q:.4f} ({len(test_x)} images, PyTorch)")

    export_pte(converted, args.pte)
    write_header(converted, test_x[args.test_index], int(test_y[args.test_index]), args.test_index, args.header)


if __name__ == "__main__":
    main()
