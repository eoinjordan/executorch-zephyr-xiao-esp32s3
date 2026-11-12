#!/usr/bin/env python3
# Copyright (c) 2026 Eoin Jordan
# SPDX-License-Identifier: Apache-2.0
"""Read the DET lines of the brick detector and choose the next target.

A starting point for the arm side of a pick-and-place demo. The firmware
prints one line for every camera frame:

    DET n=2 yellow 0.42 0.61 0.93 red 0.70 0.33 0.88 bin 0.15 0.20 0.97 ms=41

    n=<bricks>                   number of bricks that follow, 0 to 8
    <colour> <x> <y> <conf>      one per brick: yellow, white or red; the
                                 most confident first
    bin <x> <y> <conf>           only when the bin is in view
    ms=<time>                    the inference, in milliseconds

x and y are the centroid in the picture the model sees (the centre square
of the camera frame): x from 0 at the left to 1 at the right, y from 0 at
the top to 1 at the bottom. conf is the highest class probability among
the cells of the object, 0.60 to 1.00.

The rule here: with the gripper empty, go for the brick nearest to the
point of the picture where the gripper closes; with a brick held, go for
the bin. Bricks that already lie in the bin are left alone. Moving the arm
is not part of this file: next_target() returns where to go in the picture
and the caller turns that into joint angles.

    python tools/lego_reader.py                 # print the target for each frame
    python tools/lego_reader.py --holding       # as if a brick is in the gripper
    python tools/lego_reader.py --demo          # parse the example line, no board

Needs pyserial, except with --demo.
"""

import argparse
import math
import sys
from dataclasses import dataclass, field

ESPRESSIF_VID = 0x303A
COLOURS = ("yellow", "white", "red")
EXAMPLE = "DET n=2 yellow 0.42 0.61 0.93 red 0.70 0.33 0.88 bin 0.15 0.20 0.97 ms=41"


@dataclass
class Target:
    name: str
    x: float
    y: float
    conf: float


@dataclass
class Frame:
    bricks: list = field(default_factory=list)
    bin: Target | None = None
    ms: int = 0


def parse_det(line: str) -> Frame | None:
    """A DET line as a Frame, or None for any other line and for a damaged one."""
    tok = line.split()
    if len(tok) < 3 or tok[0] != "DET" or not tok[1].startswith("n="):
        return None
    try:
        n = int(tok[1][2:])
        frame = Frame()
        i = 2
        while i < len(tok) and not tok[i].startswith("ms="):
            t = Target(tok[i], float(tok[i + 1]), float(tok[i + 2]), float(tok[i + 3]))
            if t.name == "bin":
                frame.bin = t
            elif t.name in COLOURS:
                frame.bricks.append(t)
            else:
                return None
            i += 4
        frame.ms = int(tok[i][3:])
    except (ValueError, IndexError):
        return None
    return frame if len(frame.bricks) == n else None


def next_target(frame: Frame, holding: bool, gripper=(0.5, 0.5), min_conf: float = 0.8,
                bin_radius: float = 0.18) -> Target | None:
    """Where to go next, or None when there is nothing to do in this frame."""
    if holding:
        return frame.bin
    best, best_dist = None, math.inf
    for b in frame.bricks:
        if b.conf < min_conf:
            continue
        if frame.bin is not None and math.hypot(b.x - frame.bin.x, b.y - frame.bin.y) < bin_radius:
            continue    # already delivered
        dist = math.hypot(b.x - gripper[0], b.y - gripper[1])
        if dist < best_dist:
            best, best_dist = b, dist
    return best


def describe(frame: Frame, holding: bool, gripper) -> str:
    t = next_target(frame, holding, gripper)
    if t is None:
        what = "bin not in view" if holding else "no brick to pick"
        return f"{what} ({len(frame.bricks)} bricks, {frame.ms} ms)"
    dx, dy = t.x - gripper[0], t.y - gripper[1]
    verb = "drop at" if holding else "pick"
    return (f"{verb} {t.name} at {t.x:.2f} {t.y:.2f} (conf {t.conf:.2f}), "
            f"offset from the gripper {dx:+.2f} {dy:+.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", help="serial port; found by USB id when not given")
    p.add_argument("--holding", action="store_true", help="a brick is in the gripper")
    p.add_argument("--gripper", type=float, nargs=2, default=(0.5, 0.5), metavar=("X", "Y"),
                   help="where the gripper closes, in picture coordinates")
    p.add_argument("--demo", action="store_true", help="parse the example line and stop")
    args = p.parse_args()
    gripper = tuple(args.gripper)

    if args.demo:
        frame = parse_det(EXAMPLE)
        assert frame is not None and len(frame.bricks) == 2 and frame.bin is not None
        assert parse_det("DET n=3 yellow 0.1 0.2 0.9 ms=40") is None    # count and list differ
        print(EXAMPLE)
        print("empty gripper:", describe(frame, False, gripper))
        print("holding:      ", describe(frame, True, gripper))
        return

    import serial
    from serial.tools import list_ports

    name = args.port
    if name is None:
        ports = [q.device for q in list_ports.comports() if q.vid == ESPRESSIF_VID]
        if len(ports) != 1:
            sys.exit(f"found {len(ports)} Espressif serial ports; pass --port")
        name = ports[0]
    port = serial.Serial()
    port.port = name
    port.baudrate = 115200
    port.timeout = 1.0
    port.dtr = False    # do not reset the board on open
    port.rts = False
    port.open()
    print(f"port {name}; Ctrl-C to stop")
    try:
        while True:
            line = port.readline().decode("ascii", "replace").strip()
            frame = parse_det(line)
            if frame is not None:
                print(describe(frame, args.holding, gripper))
    except KeyboardInterrupt:
        pass
    finally:
        port.close()


if __name__ == "__main__":
    main()
