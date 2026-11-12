/* Copyright (c) 2026 Eoin Jordan
 * SPDX-License-Identifier: Apache-2.0
 *
 * The Seeed Studio Round Display for XIAO (GC9A01, 240 x 240) through
 * Zephyr's display API.
 *
 * There is no frame buffer: 240 x 240 RGB565 pixels would take 115 KiB. A
 * frame is drawn in horizontal bands instead. screen_render() calls the
 * draw function once per band; the drawing calls below clip to the band
 * being drawn, so the draw function simply paints the whole picture each
 * time it is called.
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SCREEN_W 240
#define SCREEN_H 240

#define RGB565(r, g, b) ((uint16_t)((((r) & 0xF8) << 8) | (((g) & 0xFC) << 3) | ((b) >> 3)))
#define COLOR_BLACK  RGB565(0, 0, 0)
#define COLOR_WHITE  RGB565(255, 255, 255)
#define COLOR_RED    RGB565(255, 0, 0)
#define COLOR_GREEN  RGB565(0, 255, 0)
#define COLOR_BLUE   RGB565(0, 0, 255)
#define COLOR_LIME   RGB565(120, 230, 120)
#define COLOR_AMBER  RGB565(250, 190, 60)
#define COLOR_GREY   RGB565(130, 130, 140)

/* Glyphs are 6 x 8 pixels times the scale given to screen_text(). */
#define SCREEN_FONT_W 6
#define SCREEN_FONT_H 8

typedef void (*screen_draw_fn)(void *ctx);

/* Clear the panel and switch it on. Returns 0 or a negative errno. The
 * backlight of the round display is on by default. */
int screen_init(void);
/* Draw one frame on a black background and send it to the panel. */
void screen_render(screen_draw_fn draw, void *ctx);

/* For use inside a draw function. */
void screen_fill(int x, int y, int w, int h, uint16_t color);
void screen_text(int x, int y, const char *s, int scale, uint16_t color);
/* 8-bit grayscale image with an integer scale (28 x 28 at scale 3 is 84 x 84). */
void screen_gray(int x, int y, const uint8_t *img, int w, int h, int scale);
/* RGB565 image in host byte order with an integer scale. */
void screen_rgb565(int x, int y, const uint16_t *img, int w, int h, int scale);
/* The outline of a rectangle, t pixels thick. */
void screen_frame(int x, int y, int w, int h, int t, uint16_t color);

/* Make the next screen_render() also print what it draws on the console:
 * "SCR 240 240 rgb565", one line of four hex digits a pixel for each row,
 * then "END SCR". tools/lego_capture.py turns that into a PNG. */
void screen_dump_next(void);

#ifdef __cplusplus
}
#endif
