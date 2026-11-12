#!/usr/bin/env python3
# Copyright (c) 2026 Eoin Jordan
# SPDX-License-Identifier: Apache-2.0
"""Synthetic desk scenes with LEGO-like bricks and a blue parts bin.

There is no real dataset for the brick detector, so training starts from
pictures drawn here: bricks as boxes with studs, side faces and a cast
shadow in yellow, white and red, a blue open bin, and things that are not
targets (skin, wood, orange, green, grey and black shapes, bricks in other
colours, gripper fingers at the edge of the frame). Every picture then gets
a desk, a lighting colour cast, an exposure, blur, sensor noise and the
RGB565 rounding of the camera.

Each sample is a 96 x 96 RGB picture, a 12 x 12 grid of cell labels and the
list of objects with their centroids:

    0 background, 1 yellow, 2 white, 3 red, 4 bin, 255 ignore

A cell takes the class that covers at least half of it. The cell under the
centroid of a small brick takes the class of the brick as well. Cells that
are only partly covered are ignored by the loss.

    python train/lego_synth.py --preview sheet.png     # look at 48 samples
"""

import argparse
import colorsys
import math
import pathlib
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SIDE = 96          # picture given to the model
GRID = 12          # cells per side
CELL = SIDE // GRID
SS = 2             # the scene is drawn at SS times the size and reduced
CANVAS = SIDE * SS

CLASSES = ("background", "yellow", "white", "red", "bin")
NUM_CLASSES = len(CLASSES)
IGNORE = 255

BG_FAMILIES = ("pale", "light_wood", "dark_wood", "grey", "black", "green_mat", "tinted")
CASTS = ("neutral", "warm", "cool", "green")


@dataclass
class Obj:
    cls: int
    x: float      # centroid, 0..1 from the left
    y: float      # centroid, 0..1 from the top
    area: float   # visible pixels at 96 x 96


@dataclass
class Sample:
    image: np.ndarray                    # (96, 96, 3) uint8
    cells: np.ndarray                    # (12, 12) uint8
    objects: list = field(default_factory=list)
    bg: str = ""
    cast: str = ""


def hsv(h_deg: float, s: float, v: float) -> np.ndarray:
    return np.array(colorsys.hsv_to_rgb((h_deg % 360.0) / 360.0, s, v)) * 255.0


def target_colour(rng, cls: int) -> np.ndarray:
    if cls == 1:    # yellow
        return hsv(rng.uniform(40, 55), rng.uniform(0.72, 1.0), rng.uniform(0.80, 1.0))
    if cls == 2:    # white
        return hsv(rng.uniform(0, 360), rng.uniform(0.0, 0.06), rng.uniform(0.84, 1.0))
    if cls == 3:    # red
        return hsv(rng.uniform(-6, 8), rng.uniform(0.78, 1.0), rng.uniform(0.55, 0.92))
    # the bin
    return hsv(rng.uniform(203, 232), rng.uniform(0.65, 1.0), rng.uniform(0.45, 0.92))


def distractor_colour(rng) -> np.ndarray:
    kind = rng.choice(
        ["skin", "wood", "orange", "green", "grey", "black", "pink", "purple", "cyan", "brown"]
    )
    if kind == "skin":
        return hsv(rng.uniform(12, 30), rng.uniform(0.22, 0.55), rng.uniform(0.55, 0.97))
    if kind == "wood":
        return hsv(rng.uniform(20, 40), rng.uniform(0.3, 0.6), rng.uniform(0.4, 0.8))
    if kind == "orange":
        return hsv(rng.uniform(16, 31), rng.uniform(0.8, 1.0), rng.uniform(0.75, 1.0))
    if kind == "green":
        return hsv(rng.uniform(80, 165), rng.uniform(0.4, 1.0), rng.uniform(0.3, 0.9))
    if kind == "grey":
        return hsv(0, 0.0, rng.uniform(0.3, 0.75))
    if kind == "black":
        return hsv(0, 0.0, rng.uniform(0.03, 0.18))
    if kind == "pink":
        return hsv(rng.uniform(315, 345), rng.uniform(0.25, 0.6), rng.uniform(0.7, 1.0))
    if kind == "purple":
        return hsv(rng.uniform(260, 300), rng.uniform(0.4, 0.9), rng.uniform(0.3, 0.8))
    if kind == "cyan":
        return hsv(rng.uniform(165, 192), rng.uniform(0.3, 0.9), rng.uniform(0.5, 0.95))
    return hsv(rng.uniform(10, 30), rng.uniform(0.5, 0.8), rng.uniform(0.2, 0.45))


def background(rng, family: str) -> np.ndarray:
    """A desk: base colour, texture and uneven light, float32 (CANVAS, CANVAS, 3)."""
    n = CANVAS
    if family == "pale":
        base = hsv(rng.uniform(0, 360), rng.uniform(0.0, 0.08), rng.uniform(0.80, 1.0))
    elif family == "light_wood":
        base = hsv(rng.uniform(24, 46), rng.uniform(0.15, 0.45), rng.uniform(0.6, 0.92))
    elif family == "dark_wood":
        base = hsv(rng.uniform(14, 36), rng.uniform(0.3, 0.62), rng.uniform(0.22, 0.5))
    elif family == "grey":
        base = hsv(rng.uniform(0, 360), rng.uniform(0.0, 0.08), rng.uniform(0.3, 0.78))
    elif family == "black":
        base = hsv(rng.uniform(0, 360), rng.uniform(0.0, 0.15), rng.uniform(0.04, 0.2))
    elif family == "green_mat":
        base = hsv(rng.uniform(120, 172), rng.uniform(0.4, 0.8), rng.uniform(0.28, 0.6))
    else:
        base = hsv(rng.uniform(0, 360), rng.uniform(0.05, 0.25), rng.uniform(0.35, 0.9))

    img = np.ones((n, n, 3), np.float32) * base.astype(np.float32)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)

    if family in ("light_wood", "dark_wood"):
        # Grain: stripes along one direction, wobbling a little.
        a = rng.uniform(0, math.pi)
        t = xx * math.cos(a) + yy * math.sin(a)
        u = -xx * math.sin(a) + yy * math.cos(a)
        grain = np.sin(t * rng.uniform(0.25, 0.9) + np.sin(u * rng.uniform(0.01, 0.05)) * 3.0)
        grain += 0.5 * np.sin(t * rng.uniform(1.0, 2.2) + rng.uniform(0, 6))
        img *= 1.0 + rng.uniform(0.03, 0.12) * grain[..., None]
    if family == "green_mat" and rng.random() < 0.7:
        step = rng.integers(12, 40)
        off = rng.integers(0, step)
        line = ((xx.astype(int) + off) % step == 0) | ((yy.astype(int) + off) % step == 0)
        img[line] = img[line] * 0.5 + 0.5 * np.array([230, 230, 200], np.float32)

    # Low-frequency mottling.
    coarse = rng.normal(0, 1, (6, 6)).astype(np.float32)
    coarse = np.asarray(
        Image.fromarray(coarse).resize((n, n), Image.BICUBIC), dtype=np.float32
    )
    img *= 1.0 + rng.uniform(0.0, 0.06) * coarse[..., None]

    # A second surface behind a straight edge: the end of the desk, a sheet.
    if rng.random() < 0.3:
        a = rng.uniform(0, 2 * math.pi)
        d = rng.uniform(-0.35, 0.35) * n
        side = (xx - n / 2) * math.cos(a) + (yy - n / 2) * math.sin(a) > d
        other = background_colour_only(rng)
        img[side] = img[side] * 0.0 + other

    # Uneven light: a ramp and a soft spot.
    a = rng.uniform(0, 2 * math.pi)
    ramp = ((xx - n / 2) * math.cos(a) + (yy - n / 2) * math.sin(a)) / n
    img *= 1.0 + rng.uniform(0.0, 0.5) * ramp[..., None]
    if rng.random() < 0.5:
        cx, cy = rng.uniform(0, n, 2)
        r = rng.uniform(0.3, 0.9) * n
        spot = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))
        img *= 1.0 + rng.uniform(-0.25, 0.35) * spot[..., None]
    return img


def background_colour_only(rng) -> np.ndarray:
    kind = rng.choice(["pale", "grey", "wood", "dark"])
    if kind == "pale":
        c = hsv(rng.uniform(0, 360), rng.uniform(0.0, 0.07), rng.uniform(0.8, 1.0))
    elif kind == "grey":
        c = hsv(0, 0.0, rng.uniform(0.3, 0.75))
    elif kind == "wood":
        c = hsv(rng.uniform(18, 42), rng.uniform(0.2, 0.55), rng.uniform(0.3, 0.85))
    else:
        c = hsv(rng.uniform(0, 360), rng.uniform(0.0, 0.2), rng.uniform(0.05, 0.25))
    return c.astype(np.float32)


def rgb(c) -> tuple:
    return tuple(int(min(max(v, 0.0), 255.0)) for v in c)


def rot(points, angle, cx, cy):
    ca, sa = math.cos(angle), math.sin(angle)
    return [(cx + x * ca - y * sa, cy + x * sa + y * ca) for x, y in points]


class Scene:
    """Colour layer, shadow layer and instance map of one picture."""

    def __init__(self, rng, bg: np.ndarray):
        self.rng = rng
        self.bg = bg
        n = CANVAS
        self.paint = Image.new("RGBA", (n, n), (0, 0, 0, 0))
        self.draw = ImageDraw.Draw(self.paint)
        self.shadow = Image.new("L", (n, n), 0)
        self.sdraw = ImageDraw.Draw(self.shadow)
        self.inst = Image.new("L", (n, n), 0)
        self.idraw = ImageDraw.Draw(self.inst)
        self.inst_cls = [0]     # class of each instance id
        # One light for the whole picture.
        a = rng.uniform(0, 2 * math.pi)
        self.shadow_dir = (math.cos(a), math.sin(a))
        self.shadow_len = rng.uniform(0.1, 0.9)
        self.shadow_dark = rng.uniform(0.15, 0.6)
        self.shadow_blur = rng.uniform(1.0, 6.0)
        # The way the camera leans: side faces show towards this direction.
        a = rng.uniform(0, 2 * math.pi)
        lean = rng.uniform(0.0, 0.9)
        self.lean = (math.cos(a) * lean, math.sin(a) * lean)

    def new_instance(self, cls: int) -> int:
        self.inst_cls.append(cls)
        return len(self.inst_cls) - 1

    def poly(self, pts, colour, inst_id=None):
        self.draw.polygon(pts, fill=rgb(colour) + (255,))
        if inst_id is not None:
            self.idraw.polygon(pts, fill=inst_id)

    def ellipse(self, box, colour, inst_id=None):
        self.draw.ellipse(box, fill=rgb(colour) + (255,))
        if inst_id is not None:
            self.idraw.ellipse(box, fill=inst_id)

    def brick(self, colour, cls, cx, cy, unit, nx, ny, angle):
        """A brick nx by ny studs. cls 0 is a brick in a colour that is not a target."""
        rng = self.rng
        # Instance 0 keeps the map clear where a non-target covers a target.
        inst_id = self.new_instance(cls) if cls else 0
        hw, hh = nx * unit / 2, ny * unit / 2
        top = rot([(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)], angle, cx, cy)
        height = 1.2 * unit
        ex, ey = self.lean[0] * height, self.lean[1] * height
        foot = [(x + ex, y + ey) for x, y in top]

        # Shadow of the footprint, pushed away from the light.
        sl = self.shadow_len * height
        sx, sy = self.shadow_dir[0] * sl, self.shadow_dir[1] * sl
        dark = int(255 * self.shadow_dark)
        self.sdraw.polygon([(x + sx, y + sy) for x, y in foot], fill=dark)
        self.sdraw.polygon(foot, fill=dark)

        # Side faces that the lean exposes, each with its own shade.
        for i in range(4):
            p, q = top[i], top[(i + 1) % 4]
            nxn, nyn = q[1] - p[1], -(q[0] - p[0])
            if nxn * ex + nyn * ey > 0:
                shade = rng.uniform(0.5, 0.9)
                self.poly([p, q, (q[0] + ex, q[1] + ey), (p[0] + ex, p[1] + ey)],
                          colour * shade, inst_id)
        top_shade = rng.uniform(0.9, 1.08)
        self.poly(top, colour * top_shade, inst_id)

        # Studs: a darker ring towards the lean, a lighter top.
        r = 0.3 * unit
        if r >= 1.2:
            sxo, syo = self.lean[0] * 0.2 * unit, self.lean[1] * 0.2 * unit
            for ix in range(nx):
                for iy in range(ny):
                    px = (ix + 0.5) * unit - hw
                    py = (iy + 0.5) * unit - hh
                    (qx, qy), = rot([(px, py)], angle, cx, cy)
                    self.ellipse([qx - r, qy - r, qx + r, qy + r], colour * top_shade * 0.78)
                    self.ellipse([qx - r - sxo, qy - r - syo, qx + r - sxo, qy + r - syo],
                                 colour * min(top_shade * 1.06, 1.12))
        # Gloss: a light streak across the top face.
        if rng.random() < 0.35:
            t = rng.uniform(0.2, 0.8)
            w = rng.uniform(0.04, 0.12)
            a = [(1 - t + w) * top[0][k] + (t - w) * top[1][k] for k in (0, 1)]
            b = [(1 - t - w) * top[0][k] + (t + w) * top[1][k] for k in (0, 1)]
            c = [(1 - t - w) * top[3][k] + (t + w) * top[2][k] for k in (0, 1)]
            d = [(1 - t + w) * top[3][k] + (t - w) * top[2][k] for k in (0, 1)]
            gloss = colour * 0.6 + 255.0 * 0.4
            self.draw.polygon([tuple(a), tuple(b), tuple(c), tuple(d)], fill=rgb(gloss) + (255,))
        return inst_id

    def bin(self, cx, cy, w, h, angle):
        """An open blue box seen from above: rim, walls and a darker floor."""
        rng = self.rng
        colour = target_colour(rng, 4)
        inst_id = self.new_instance(4)
        outer = rot([(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)],
                    angle, cx, cy)
        sl = self.shadow_len * 0.25 * min(w, h)
        self.sdraw.polygon([(x + self.shadow_dir[0] * sl, y + self.shadow_dir[1] * sl)
                            for x, y in outer], fill=int(255 * self.shadow_dark))
        self.poly(outer, colour * rng.uniform(0.9, 1.1), inst_id)
        # Walls slope to a smaller floor that sits off centre with the lean.
        k = rng.uniform(0.62, 0.86)
        ox, oy = self.lean[0] * 0.12 * min(w, h), self.lean[1] * 0.12 * min(w, h)
        rim = rng.uniform(0.03, 0.09) * min(w, h)
        inner = rot([(-w / 2 + rim, -h / 2 + rim), (w / 2 - rim, -h / 2 + rim),
                     (w / 2 - rim, h / 2 - rim), (-w / 2 + rim, h / 2 - rim)], angle, cx, cy)
        floor = rot([(-w * k / 2, -h * k / 2), (w * k / 2, -h * k / 2),
                     (w * k / 2, h * k / 2), (-w * k / 2, h * k / 2)], angle, cx + ox, cy + oy)
        for i in range(4):
            j = (i + 1) % 4
            self.poly([inner[i], inner[j], floor[j], floor[i]],
                      colour * rng.uniform(0.45, 0.95), inst_id)
        self.poly(floor, colour * rng.uniform(0.35, 0.8), inst_id)
        return inst_id, (cx + ox, cy + oy, w * k, h * k)

    def blob(self, colour):
        """Something that is not a target: an ellipse, a polygon or a thick stroke."""
        rng = self.rng
        n = CANVAS
        kind = rng.choice(["ellipse", "poly", "stroke"])
        cx, cy = rng.uniform(-0.1 * n, 1.1 * n, 2)
        if kind == "ellipse":
            rx, ry = rng.uniform(0.03 * n, 0.25 * n, 2)
            self.draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=rgb(colour) + (255,))
            self.idraw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=0)
        elif kind == "poly":
            k = rng.integers(3, 7)
            r = rng.uniform(0.04 * n, 0.25 * n)
            ang = np.sort(rng.uniform(0, 2 * math.pi, k))
            pts = [(cx + math.cos(a) * r * rng.uniform(0.5, 1.0),
                    cy + math.sin(a) * r * rng.uniform(0.5, 1.0)) for a in ang]
            self.draw.polygon(pts, fill=rgb(colour) + (255,))
            self.idraw.polygon(pts, fill=0)
        else:
            x2, y2 = rng.uniform(-0.1 * n, 1.1 * n, 2)
            wd = int(rng.uniform(0.03 * n, 0.14 * n))
            self.draw.line([cx, cy, x2, y2], fill=rgb(colour) + (255,), width=wd)
            self.idraw.line([cx, cy, x2, y2], fill=0, width=wd)
            for px, py in ((cx, cy), (x2, y2)):
                self.draw.ellipse([px - wd / 2, py - wd / 2, px + wd / 2, py + wd / 2],
                                  fill=rgb(colour) + (255,))
                self.idraw.ellipse([px - wd / 2, py - wd / 2, px + wd / 2, py + wd / 2], fill=0)

    def fingers(self):
        """The gripper: two bars coming in from one edge of the frame."""
        rng = self.rng
        n = CANVAS
        colour = rng.choice([0, 1, 2])
        colour = [hsv(0, 0, rng.uniform(0.03, 0.2)), hsv(0, 0, rng.uniform(0.4, 0.8)),
                  hsv(rng.uniform(16, 30), rng.uniform(0.8, 1.0), rng.uniform(0.7, 1.0))][colour]
        gap = rng.uniform(0.2, 0.6) * n
        wd = rng.uniform(0.06, 0.16) * n
        ln = rng.uniform(0.1, 0.45) * n
        mid = rng.uniform(0.35, 0.65) * n
        edge = rng.integers(0, 4)
        for s in (-1, 1):
            a = mid + s * gap / 2
            box = [a - wd / 2, n - ln, a + wd / 2, n + 4]
            pts = [(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])]
            pts = rot([(x - n / 2, y - n / 2) for x, y in pts], edge * math.pi / 2, n / 2, n / 2)
            self.draw.polygon(pts, fill=rgb(colour * rng.uniform(0.85, 1.1)) + (255,))
            self.idraw.polygon(pts, fill=0)

    def compose(self) -> np.ndarray:
        shadow = self.shadow.filter(ImageFilter.GaussianBlur(self.shadow_blur))
        shade = 1.0 - np.asarray(shadow, np.float32)[..., None] / 255.0
        paint = np.asarray(self.paint, np.float32)
        alpha = paint[..., 3:4] / 255.0
        return self.bg * shade * (1.0 - alpha) + paint[..., :3] * alpha


BRICK_SIZES = ((1, 1), (1, 2), (2, 2), (2, 2), (2, 3), (2, 4), (2, 4), (1, 4), (1, 3))


def photometric(rng, img: np.ndarray, cast: str) -> np.ndarray:
    """Light colour, exposure, lens and sensor: float (H, W, 3) 0..255 in, uint8 out."""
    if cast == "warm":
        gains = (rng.uniform(1.0, 1.22), rng.uniform(0.92, 1.04), rng.uniform(0.5, 0.85))
    elif cast == "cool":
        gains = (rng.uniform(0.72, 0.92), rng.uniform(0.95, 1.05), rng.uniform(1.04, 1.25))
    elif cast == "green":
        gains = (rng.uniform(0.82, 0.96), rng.uniform(1.0, 1.12), rng.uniform(0.8, 0.96))
    else:
        gains = tuple(rng.uniform(0.95, 1.05, 3))
    x = img / 255.0 * np.array(gains, np.float32)

    # Saturation of the camera pipeline.
    grey = x.mean(axis=2, keepdims=True)
    x = grey + (x - grey) * rng.uniform(0.6, 1.3)
    # Exposure, with clipping, then contrast and gamma.
    x = x * rng.uniform(0.5, 1.5)
    x = (x - 0.5) * rng.uniform(0.8, 1.25) + 0.5 + rng.uniform(-0.06, 0.06)
    x = np.clip(x, 0.0, 1.0) ** rng.uniform(0.75, 1.3)

    # Vignette.
    n = x.shape[0]
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    r2 = ((xx - n / 2) ** 2 + (yy - n / 2) ** 2) / (n * n / 2)
    x = x * (1.0 - rng.uniform(0.0, 0.45) * r2[..., None])

    im = Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8))
    im = im.resize((SIDE, SIDE), Image.BOX)
    blur = rng.uniform(0.0, 1.3)
    if blur > 0.25:
        im = im.filter(ImageFilter.GaussianBlur(blur))
    out = np.asarray(im, np.float32)
    out = out + rng.normal(0, rng.uniform(0.5, 8.0), out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    # RGB565, as the camera delivers it.
    out[..., 0] = (out[..., 0] & 0xF8) | (out[..., 0] >> 5)
    out[..., 1] = (out[..., 1] & 0xFC) | (out[..., 1] >> 6)
    out[..., 2] = (out[..., 2] & 0xF8) | (out[..., 2] >> 5)
    return out


def labels(inst: np.ndarray, inst_cls: list):
    """Cell labels and the object list from the instance map at CANVAS size."""
    cells = np.zeros((GRID, GRID), np.uint8)
    step = CANVAS // GRID
    cls_lut = np.array(inst_cls, np.uint8)
    cls_map = cls_lut[inst]
    onehot = np.stack([(cls_map == c) for c in range(NUM_CLASSES)], axis=0).astype(np.float32)
    cover = onehot.reshape(NUM_CLASSES, GRID, step, GRID, step).mean(axis=(2, 4))
    best = cover[1:].argmax(axis=0) + 1
    amount = cover[1:].max(axis=0)
    cells[amount >= 0.5] = best[amount >= 0.5]
    cells[(amount >= 0.12) & (amount < 0.5)] = IGNORE

    objects = []
    for i in range(1, len(inst_cls)):
        ys, xs = np.nonzero(inst == i)
        area = len(ys) / (SS * SS)
        if area == 0:
            continue
        cx, cy = xs.mean(), ys.mean()
        gx, gy = int(cx) // step, int(cy) // step
        if area < 20:
            # Too little of it shows to ask for a detection.
            if cells[gy, gx] == 0:
                cells[gy, gx] = IGNORE
            continue
        if inst_cls[i] != 4 and cover[inst_cls[i], gy, gx] >= 0.12:
            cells[gy, gx] = inst_cls[i]
        objects.append(Obj(inst_cls[i], cx / CANVAS, cy / CANVAS, area))
    return cells, objects


def make_sample(rng, bg: str | None = None, cast: str | None = None) -> Sample:
    n = CANVAS
    if bg is None:
        bg = rng.choice(BG_FAMILIES, p=[0.28, 0.16, 0.12, 0.16, 0.08, 0.08, 0.12])
    if cast is None:
        cast = rng.choice(CASTS, p=[0.4, 0.3, 0.18, 0.12])
    scene = Scene(rng, background(rng, bg))

    # Sheets of paper: pale rectangles that are not bricks.
    for _ in range(rng.integers(0, 3) if rng.random() < 0.35 else 0):
        c = hsv(rng.uniform(0, 360), rng.uniform(0.0, 0.06), rng.uniform(0.82, 1.0))
        w, h = rng.uniform(0.2 * n, 0.8 * n, 2)
        pts = rot([(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)],
                  rng.uniform(0, math.pi), rng.uniform(0, n), rng.uniform(0, n))
        scene.poly(pts, c)

    for _ in range(rng.integers(0, 5)):
        scene.blob(distractor_colour(rng))

    # One picture has one scale: the camera is at one height above the desk.
    unit = float(np.exp(rng.uniform(math.log(3.2), math.log(15.0)))) * SS

    floor = None
    if rng.random() < 0.45:
        w = rng.uniform(0.3, 0.85) * n
        h = w * rng.uniform(0.6, 1.0)
        _, floor = scene.bin(rng.uniform(0.05 * n, 0.95 * n), rng.uniform(0.05 * n, 0.95 * n),
                             w, h, rng.uniform(0, math.pi))

    count = 0 if rng.random() < 0.12 else rng.integers(1, 7)
    for _ in range(count):
        if rng.random() < 0.22:
            cls, colour = 0, distractor_colour(rng)
        else:
            cls = int(rng.integers(1, 4))
            colour = target_colour(rng, cls)
        nx, ny = BRICK_SIZES[rng.integers(0, len(BRICK_SIZES))]
        if floor is not None and rng.random() < 0.25:
            cx = floor[0] + rng.uniform(-0.3, 0.3) * floor[2]
            cy = floor[1] + rng.uniform(-0.3, 0.3) * floor[3]
        else:
            cx, cy = rng.uniform(0.02 * n, 0.98 * n, 2)
        scene.brick(colour, cls, cx, cy, unit * rng.uniform(0.9, 1.1), nx, ny,
                    rng.uniform(0, math.pi))

    for _ in range(rng.integers(0, 2)):
        scene.blob(distractor_colour(rng))
    if rng.random() < 0.3:
        scene.fingers()

    image = photometric(rng, scene.compose(), cast)
    inst = np.asarray(scene.inst)
    cells, objects = labels(inst, scene.inst_cls)
    return Sample(image, cells, objects, bg, cast)


def make_set(count: int, seed: int, bg: str | None = None, cast: str | None = None):
    rng = np.random.default_rng(seed)
    return [make_sample(rng, bg, cast) for _ in range(count)]


CLASS_RGB = ((0, 0, 0), (255, 220, 0), (255, 255, 255), (230, 30, 20), (40, 110, 255))


def annotate(image: np.ndarray, cells: np.ndarray, scale: int = 3) -> Image.Image:
    """The picture enlarged with the labelled cells outlined in the class colour."""
    im = Image.fromarray(image).resize((SIDE * scale, SIDE * scale), Image.NEAREST)
    d = ImageDraw.Draw(im)
    s = CELL * scale
    for gy in range(GRID):
        for gx in range(GRID):
            c = int(cells[gy, gx])
            if c == 0:
                continue
            col = (255, 0, 255) if c == IGNORE else CLASS_RGB[c]
            d.rectangle([gx * s + 1, gy * s + 1, gx * s + s - 2, gy * s + s - 2], outline=col)
            if c != IGNORE:
                d.rectangle([gx * s + 2, gy * s + 2, gx * s + s - 3, gy * s + s - 3],
                            outline=(0, 0, 0))
    return im


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preview", type=pathlib.Path, required=True,
                   help="write a sheet of annotated samples to this PNG")
    p.add_argument("--count", type=int, default=48)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--bg", choices=BG_FAMILIES)
    p.add_argument("--cast", choices=CASTS)
    args = p.parse_args()

    samples = make_set(args.count, args.seed, args.bg, args.cast)
    cols = 8
    rows = (len(samples) + cols - 1) // cols
    tile = SIDE * 3
    sheet = Image.new("RGB", (cols * tile, rows * tile))
    for i, s in enumerate(samples):
        sheet.paste(annotate(s.image, s.cells), ((i % cols) * tile, (i // cols) * tile))
    sheet.save(args.preview)
    print(f"wrote {args.preview} ({len(samples)} samples)")


if __name__ == "__main__":
    main()
