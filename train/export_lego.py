#!/usr/bin/env python3
# Copyright (c) 2026 Eoin Jordan
# SPDX-License-Identifier: Apache-2.0
"""Export BrickNet for the firmware: fp32, int8 and the self-test picture.

Reads train/out/lego_bricknet.pt (train/train_lego.py or
train/finetune_lego.py) and writes:

  firmware/model/lego_bricknet.pte            fp32, portable kernels
  firmware/model/lego_bricknet_cmsis_nn.pte   int8, cortex_m operators on
                                              CMSIS-NN, plain C path
  firmware/src/lego_test_input.h              one synthetic picture, the
                                              logits of both models for it
                                              and the int8 input scale

The int8 program takes the picture as int8, not float: the quantize operator
at its input is removed, so the firmware hands over camera bytes and the
27,648 byte input is the largest tensor instead of a 110 KB float one.

Both models are scored on the held-out synthetic set, so the cost of
quantising shows next to the fp32 numbers.

    python train/train_lego.py
    python train/export_lego.py
"""

import argparse
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import lego_synth  # noqa: E402
from lego_model import INPUT_SHAPE, BrickNet, decode, det_line  # noqa: E402
from train_lego import STATE, predict, report, synth, tensors, to_input, validation_set  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
NHWC = torch.channels_last
CALIBRATION_SEED = 770001


def program_ops(program) -> list[str]:
    return sorted({str(n.target) for n in program.exported_program().graph.nodes
                   if n.op == "call_function" and hasattr(n.target, "_schema")})


def write_pte(program, out: pathlib.Path) -> None:
    print("operators in program:")
    for op in program_ops(program):
        print(f"  {op}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(program.buffer)
    print(f"wrote {out} ({out.stat().st_size} bytes)")


def export_fp32(model, out: pathlib.Path) -> None:
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    with torch.no_grad():
        exported = torch.export.export(model, (torch.zeros(*INPUT_SHAPE),))
        program = to_edge_transform_and_lower(
            exported, compile_config=EdgeCompileConfig(_check_ir_validity=False)
        ).to_executorch()
    write_pte(program, out)


def quantise(model, calibration_u8: torch.Tensor):
    """PT2E quantisation with the Cortex-M quantizer. Returns the converted module."""
    from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    example = (torch.zeros(*INPUT_SHAPE).to(memory_format=NHWC),)
    with torch.no_grad():
        exported = torch.export.export(model, example, strict=False)
        try:
            captured = exported.module(check_guards=False)
        except TypeError:
            captured = exported.module()
        prepared = prepare_pt2e(captured, CortexMQuantizer())
        for i in range(0, len(calibration_u8), 50):
            prepared(to_input(calibration_u8[i:i + 50]).to(memory_format=NHWC))
        return convert_pt2e(prepared)


def qparams(converted, which: str):
    """(scale, zero point) of the first quantize or the last dequantize node."""
    found = None
    for node in converted.graph.nodes:
        if node.op != "call_function":
            continue
        name = str(node.target)
        if which == "input" and "quantize_per_tensor" in name and "dequantize" not in name:
            return float(node.args[1]), int(node.args[2])
        if which == "output" and "dequantize_per_tensor" in name:
            found = (float(node.args[1]), int(node.args[2]))
    if found is None:
        raise RuntimeError(f"the quantised graph has no {which} quantisation node")
    return found


def export_int8(converted, out: pathlib.Path) -> None:
    from executorch.backends.cortex_m.edge_compile_config import cortex_m_edge_compile_config
    from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager
    from executorch.backends.cortex_m.target_config import CortexM, CortexMTargetConfig
    from executorch.exir import to_edge
    from executorch.exir.passes.quantize_io_pass import QuantizeInputs

    # A Cortex-M0+ has neither DSP nor Helium: the plain C kernels of
    # CMSIS-NN, the ones that also compile for Xtensa.
    target = CortexMTargetConfig(cpu=CortexM.M0PLUS)
    example = (torch.zeros(*INPUT_SHAPE).to(memory_format=NHWC),)
    with torch.no_grad():
        exported = torch.export.export(converted, example, strict=False)
        edge = to_edge(exported, compile_config=cortex_m_edge_compile_config())
        edge = edge.transform([QuantizeInputs(edge, [0])])
        edge._edge_programs["forward"] = CortexMPassManager(
            edge.exported_program(), target_config=target
        ).transform()
        program = edge.to_executorch()
    print(f"CMSIS-NN code path: {target.backend.name}")
    write_pte(program, out)


def pick_test(samples, logits, logits_q):
    """Index of a held-out picture with bricks of two colours and the bin, all found by
    both models."""
    for i, (s, z, zq) in enumerate(zip(samples, logits, logits_q)):
        bricks, the_bin = decode(z)
        bricks_q, bin_q = decode(zq)
        if bin_q is None or sorted(d.cls for d in bricks_q) != sorted(d.cls for d in bricks):
            continue
        truth = [o for o in s.objects if o.cls != 4]
        if the_bin is None or len(bricks) < 2 or len(bricks) != len(truth):
            continue
        if len({d.cls for d in bricks}) < 2 or min(d.conf for d in bricks) < 0.9:
            continue
        if sorted(d.cls for d in bricks) == sorted(o.cls for o in truth):
            return i
    print("warning: no held-out picture has two colours and the bin found by both models;"
          " the self-test uses the first picture")
    return 0


def c_array(values, per_line: int, fmt) -> list[str]:
    return ["    " + ", ".join(fmt(v) for v in values[i:i + per_line]) + ","
            for i in range(0, len(values), per_line)]


def write_header(sample, z_fp32, z_int8, in_q, out_q, out: pathlib.Path) -> None:
    bricks, the_bin = decode(z_fp32)
    bricks_q, bin_q = decode(z_int8)
    lines = [
        "/* Generated by train/export_lego.py. Do not edit.",
        " * One held-out synthetic picture and the logits PyTorch gives for it.",
        f" * fp32: {det_line(bricks, the_bin)}",
        f" * int8: {det_line(bricks_q, bin_q)}",
        " */",
        "#pragma once",
        "",
        "#include <stdint.h>",
        "",
        f"#define LEGO_TEST_BRICKS {len(bricks)}",
        f"#define LEGO_TEST_BIN {1 if the_bin is not None else 0}",
        "",
        "/* The int8 program takes round(pixel / 255 / scale) + zero_point. */",
        f"#define LEGO_Q_INPUT_SCALE {in_q[0]:.9f}f",
        f"#define LEGO_Q_INPUT_ZERO_POINT {in_q[1]}",
        "/* One step of the int8 output: its logits are multiples of this. */",
        f"#define LEGO_Q_LOGIT_STEP {out_q[0]:.6f}f",
        "",
        "/* RGB, row by row, three bytes a pixel: [96][96][3]. */",
        "static const uint8_t kLegoTestImage[96 * 96 * 3] = {",
        *c_array(sample.image.flatten().tolist(), 24, str),
        "};",
        "",
        "/* fp32 model, [class][row][column]. */",
        "static const float kLegoTestLogits[5 * 12 * 12] = {",
        *c_array(z_fp32.flatten().tolist(), 8, lambda v: f"{v:.5f}f"),
        "};",
        "",
        "/* Quantised model computed in PyTorch, same layout. The integer kernels",
        " * on the device may differ from it by a rounding step or two. */",
        "static const float kLegoTestLogitsQ[5 * 12 * 12] = {",
        *c_array(z_int8.flatten().tolist(), 8, lambda v: f"{v:.5f}f"),
        "};",
        "",
    ]
    out.write_text("\n".join(lines), newline="\n")
    print(f"wrote {out}")
    print(f"  fp32: {det_line(bricks, the_bin)}")
    print(f"  int8: {det_line(bricks_q, bin_q)}")


def check_on_host(pte: pathlib.Path, sample, expected: np.ndarray) -> None:
    """Run the fp32 program with the ExecuTorch Python runtime."""
    from executorch.runtime import Runtime

    x = to_input(torch.from_numpy(sample.image).permute(2, 0, 1)[None].contiguous())
    method = Runtime.get().load_program(pte).load_method("forward")
    got = method.execute([x])[0][0].numpy()
    diff = float(np.abs(got - expected).max())
    print(f"{pte.name} on the host runtime: max |pte - pytorch| = {diff:.2e}")
    assert diff < 1e-3, "pte output drifted from PyTorch"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state-dict", type=pathlib.Path, default=STATE)
    p.add_argument("--pte", type=pathlib.Path,
                   default=REPO / "firmware" / "model" / "lego_bricknet.pte")
    p.add_argument("--pte-int8", type=pathlib.Path,
                   default=REPO / "firmware" / "model" / "lego_bricknet_cmsis_nn.pte")
    p.add_argument("--header", type=pathlib.Path,
                   default=REPO / "firmware" / "src" / "lego_test_input.h")
    p.add_argument("--val", type=int, default=2000, help="held-out synthetic pictures to score")
    p.add_argument("--calibration", type=int, default=500)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()

    if not args.state_dict.exists():
        sys.exit(f"{args.state_dict} not found: run train/train_lego.py first")
    net = BrickNet()
    net.load_state_dict(torch.load(args.state_dict, map_location="cpu"))
    model = net.fused()

    val = validation_set(args.workers, args.val)
    xv, _ = tensors(val)
    xc, _ = tensors(synth(args.calibration, CALIBRATION_SEED, args.workers))

    z_fp32 = predict(model, xv)
    print("\n== fp32 ==")
    report(z_fp32, val)
    export_fp32(model, args.pte)

    print("\n== int8 ==")
    converted = quantise(model.to(memory_format=NHWC), xc)
    z_int8 = predict(converted, xv, nhwc=True)
    report(z_int8, val)
    export_int8(converted, args.pte_int8)

    i = pick_test(val, z_fp32, z_int8)
    sample = val[i]
    write_header(sample, z_fp32[i], z_int8[i], qparams(converted, "input"),
                 qparams(converted, "output"), args.header)
    check_on_host(args.pte, sample, z_fp32[i])


if __name__ == "__main__":
    main()
