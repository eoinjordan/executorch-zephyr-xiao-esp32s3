#!/usr/bin/env python3
# Copyright (c) 2026 Eoin Jordan
# SPDX-License-Identifier: Apache-2.0
"""Pull pictures from the brick detector firmware over its USB serial console.

The firmware (firmware/src/lego.cpp) answers one-character requests:

    c   the next camera frame as the model gets it (96 x 96), then its DET line
    s   the next screen as it is drawn (240 x 240)
    i   the start-up report again: sizes, self-tests and timings
    t   one frame from the synthetic picture in the firmware, not the camera

Pictures for training go into one folder per label, which is all the
labelling a first round of fine-tuning needs (train/finetune_lego.py):

    data/lego/real/background/   no brick and no bin in view
    data/lego/real/yellow/       only yellow bricks in view
    data/lego/real/white/        only white bricks
    data/lego/real/red/          only red bricks
    data/lego/real/bin/          only the bin
    data/lego/real/mixed/        anything else; needs a points file to be used

Each folder also gets detections.txt: the DET line the board printed for
every picture, to see what the model on the device made of it.

    python tools/lego_capture.py --label background --count 20
    python tools/lego_capture.py --label yellow --count 30 --interval 1.5
    python tools/lego_capture.py --screen screen.png
    python tools/lego_capture.py --screen selftest.png --test
    python tools/lego_capture.py --console console.txt --info --seconds 15

The port is found by the USB vendor id of the board (Espressif, 303A).
Needs pyserial and Pillow.
"""

import argparse
import pathlib
import sys
import time

import serial
from serial.tools import list_ports

ESPRESSIF_VID = 0x303A
LABELS = ("background", "yellow", "white", "red", "bin", "mixed")


def find_port() -> str:
    ports = [p for p in list_ports.comports() if p.vid == ESPRESSIF_VID]
    if not ports:
        sys.exit("no serial port with the Espressif USB vendor id 303A; pass --port")
    if len(ports) > 1:
        names = ", ".join(p.device for p in ports)
        sys.exit(f"more than one Espressif port ({names}); pass --port")
    return ports[0].device


def open_port(name: str) -> serial.Serial:
    port = serial.Serial()
    port.port = name
    port.baudrate = 115200
    port.timeout = 0.5
    # Opening must not pull the lines that reset the chip.
    port.dtr = False
    port.rts = False
    port.open()
    return port


def reset(port: serial.Serial) -> None:
    """Restart the firmware: RTS is the reset line of the USB serial bridge in the chip."""
    port.dtr = False
    port.rts = True
    time.sleep(0.1)
    port.rts = False


def read_line(port: serial.Serial, deadline: float) -> str | None:
    while time.time() < deadline:
        raw = port.readline()
        if raw.endswith(b"\n"):
            return raw.decode("ascii", "replace").rstrip("\r\n")
        if raw:
            # A partial line: wait for the rest of it.
            rest = port.read_until(b"\n")
            return (raw + rest).decode("ascii", "replace").rstrip("\r\n")
    return None


def read_block(port: serial.Serial, tag: str, request: bytes, timeout: float = 15.0):
    """Send the request and return (image, lines seen before the block)."""
    from PIL import Image

    port.reset_input_buffer()
    port.write(request)
    deadline = time.time() + timeout
    before = []
    while True:
        line = read_line(port, deadline)
        if line is None:
            sys.exit(f"no {tag} block from the board in {timeout:.0f} s; is the brick firmware "
                     "running (lego.conf)?")
        if line.startswith(tag + " "):
            break
        before.append(line)
    _, w, h, fmt = line.split()
    w, h = int(w), int(h)
    if fmt != "rgb565":
        sys.exit(f"unknown pixel format {fmt}")
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        row = read_line(port, deadline)
        if row is None or len(row) != 4 * w:
            sys.exit(f"{tag} row {y} is damaged ({0 if row is None else len(row)} characters)")
        for x in range(w):
            v = int(row[4 * x:4 * x + 4], 16)
            r, g, b = (v >> 11) & 0x1F, (v >> 5) & 0x3F, v & 0x1F
            px[x, y] = ((r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2))
    end = read_line(port, deadline)
    if end != f"END {tag}":
        sys.exit(f"{tag} block did not end properly: {end!r}")
    return img, before


def capture_frame(port: serial.Serial):
    """One camera frame and the DET line of the same frame."""
    img, _ = read_block(port, "IMG", b"c")
    deadline = time.time() + 5.0
    while True:
        line = read_line(port, deadline)
        if line is None:
            return img, ""
        if line.startswith("DET "):
            return img, line


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", help="serial port; found by USB id when not given")
    p.add_argument("--label", choices=LABELS, help="capture camera frames into this folder")
    p.add_argument("--count", type=int, default=10, help="frames to capture")
    p.add_argument("--interval", type=float, default=1.0, help="seconds between frames")
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/lego/real"))
    p.add_argument("--screen", type=pathlib.Path, help="save what the display shows to this PNG")
    p.add_argument("--console", type=pathlib.Path, help="save the console text to this file")
    p.add_argument("--seconds", type=float, default=10.0, help="how long to log the console")
    p.add_argument("--test", action="store_true",
                   help="with --screen: show the synthetic self-test picture for that frame")
    p.add_argument("--info", action="store_true",
                   help="with --console: ask for the start-up report first")
    p.add_argument("--reset", action="store_true", help="restart the firmware first")
    args = p.parse_args()
    if not (args.label or args.screen or args.console):
        p.error("give --label, --screen or --console")

    name = args.port or find_port()
    port = open_port(name)
    print(f"port {name}")
    if args.reset:
        reset(port)

    if args.console:
        if args.info:
            port.reset_input_buffer()
            port.write(b"i")
        deadline = time.time() + args.seconds
        lines = []
        while time.time() < deadline:
            line = read_line(port, deadline)
            if line is not None:
                lines.append(line)
        args.console.parent.mkdir(parents=True, exist_ok=True)
        args.console.write_text("\n".join(lines) + "\n", newline="\n")
        print(f"wrote {args.console} ({len(lines)} lines)")

    if args.label:
        folder = args.out / args.label
        folder.mkdir(parents=True, exist_ok=True)
        log = folder / "detections.txt"
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for i in range(args.count):
            img, det = capture_frame(port)
            path = folder / f"{stamp}-{i:03d}.png"
            img.save(path)
            with log.open("a", newline="\n") as f:
                f.write(f"{path.name} {det}\n")
            print(f"{path}  {det}")
            if i + 1 < args.count:
                time.sleep(args.interval)

    if args.screen:
        img, _ = read_block(port, "SCR", b"ts" if args.test else b"s", timeout=30.0)
        args.screen.parent.mkdir(parents=True, exist_ok=True)
        img.save(args.screen)
        print(f"wrote {args.screen}")
    port.close()


if __name__ == "__main__":
    main()
