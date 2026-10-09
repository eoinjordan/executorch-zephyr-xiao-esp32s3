#!/usr/bin/env python3
# Copyright (c) 2026 Eoin Jordan
# SPDX-License-Identifier: Apache-2.0
"""Run the exported .pte on the host with the ExecuTorch runtime.

Confirms the program loads and that its logits match the embedded test input
before you spend time flashing. Needs the `executorch` pip package.
"""

import argparse
import pathlib
import re

import torch
from executorch.runtime import Runtime

REPO = pathlib.Path(__file__).resolve().parents[1]


def read_header(path: pathlib.Path):
    text = path.read_text()
    label = int(re.search(r"#define TEST_INPUT_LABEL (\d+)", text).group(1))
    body = text.split("kTestInput[TEST_INPUT_NUMEL] = {")[1].split("};")[0]
    pixels = [float(v) for v in re.findall(r"(-?\d+\.\d+)f", body)]
    body = text.split("kTestExpectedLogits[10] = {")[1].split("};")[0]
    expected = [float(v) for v in re.findall(r"(-?\d+\.\d+)f", body)]
    return torch.tensor(pixels).reshape(1, 1, 28, 28), label, torch.tensor(expected)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pte", type=pathlib.Path, default=REPO / "firmware" / "model" / "mnist_tiny.pte")
    p.add_argument("--header", type=pathlib.Path, default=REPO / "firmware" / "src" / "test_input.h")
    args = p.parse_args()

    x, label, expected = read_header(args.header)
    program = Runtime.get().load_program(args.pte)
    method = program.load_method("forward")
    logits = method.execute([x])[0][0]

    print("logits:", [f"{v:.4f}" for v in logits.tolist()])
    print(f"argmax={int(logits.argmax())} label={label}")
    diff = (logits - expected).abs().max().item()
    print(f"max |pte - pytorch| = {diff:.2e}")
    assert int(logits.argmax()) == label, "prediction does not match label"
    assert diff < 1e-3, "pte output drifted from PyTorch"
    print("OK")


if __name__ == "__main__":
    main()
