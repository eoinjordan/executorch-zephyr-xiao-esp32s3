/* Copyright (c) 2026 Eoin Jordan
 * SPDX-License-Identifier: Apache-2.0
 */
#include "screen.h"

#include <errno.h>
#include <string.h>

#include <zephyr/device.h>
#include <zephyr/drivers/display.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

#include "font6x8.h"

#define BAND_ROWS CONFIG_APP_SCREEN_BAND_ROWS

static const struct device *const disp = DEVICE_DT_GET(DT_CHOSEN(zephyr_display));
static uint16_t band[SCREEN_W * BAND_ROWS];
static int band_y0;
static bool ready;
static bool dump_next;

/* RGB565 as the panel reads it. The GC9A01 takes the high byte first and
 * the driver sends the buffer unchanged, so on a little-endian CPU the two
 * bytes are swapped here. */
static inline uint16_t pack(uint16_t c)
{
	return IS_ENABLED(CONFIG_APP_SCREEN_SWAP_BYTES) ? (uint16_t)((c << 8) | (c >> 8)) : c;
}

/* The band as text: one line a row, four hex digits a pixel. */
static void dump_band(int rows)
{
	static const char hex[] = "0123456789abcdef";
	static char line[SCREEN_W * 4 + 1];

	for (int y = 0; y < rows; y++) {
		for (int x = 0; x < SCREEN_W; x++) {
			uint16_t c = pack(band[y * SCREEN_W + x]); /* pack() undoes itself */

			line[4 * x] = hex[c >> 12];
			line[4 * x + 1] = hex[(c >> 8) & 0xf];
			line[4 * x + 2] = hex[(c >> 4) & 0xf];
			line[4 * x + 3] = hex[c & 0xf];
		}
		line[SCREEN_W * 4] = '\0';
		printk("%s\n", line);
	}
}

static inline void put(int x, int y, uint16_t packed)
{
	y -= band_y0;
	if (x >= 0 && x < SCREEN_W && y >= 0 && y < BAND_ROWS) {
		band[y * SCREEN_W + x] = packed;
	}
}

int screen_init(void)
{
	struct display_capabilities caps;
	int ret;

	if (!device_is_ready(disp)) {
		printk("screen: display not ready\n");
		return -ENODEV;
	}
	display_get_capabilities(disp, &caps);
	printk("screen: %s %ux%u, pixel format 0x%x\n", disp->name, caps.x_resolution,
	       caps.y_resolution, (unsigned int)caps.current_pixel_format);
	if (caps.x_resolution != SCREEN_W || caps.y_resolution != SCREEN_H) {
		printk("screen: written for %dx%d\n", SCREEN_W, SCREEN_H);
		return -ENOTSUP;
	}
	if (caps.current_pixel_format != PIXEL_FORMAT_RGB_565) {
		ret = display_set_pixel_format(disp, PIXEL_FORMAT_RGB_565);
		if (ret != 0) {
			printk("screen: no 16-bit colour mode (%d)\n", ret);
			return ret;
		}
	}
	ready = true;
	screen_render(NULL, NULL);
	return display_blanking_off(disp);
}

void screen_render(screen_draw_fn draw, void *ctx)
{
	struct display_buffer_descriptor desc = {
		.width = SCREEN_W,
		.pitch = SCREEN_W,
	};

	if (!ready) {
		return;
	}
	if (dump_next) {
		printk("SCR %d %d rgb565\n", SCREEN_W, SCREEN_H);
	}
	for (band_y0 = 0; band_y0 < SCREEN_H; band_y0 += BAND_ROWS) {
		int rows = MIN(BAND_ROWS, SCREEN_H - band_y0);
		int ret;

		memset(band, 0, sizeof(band));
		if (draw != NULL) {
			draw(ctx);
		}
		if (dump_next) {
			dump_band(rows);
		}
		desc.height = rows;
		desc.buf_size = (uint32_t)rows * SCREEN_W * sizeof(uint16_t);
		ret = display_write(disp, 0, band_y0, &desc, band);
		if (ret != 0) {
			printk("screen: display_write %d\n", ret);
			break;
		}
	}
	if (dump_next) {
		printk("END SCR\n");
		dump_next = false;
	}
}

void screen_dump_next(void)
{
	dump_next = true;
}

void screen_fill(int x, int y, int w, int h, uint16_t color)
{
	uint16_t c = pack(color);

	for (int iy = MAX(y, band_y0); iy < MIN(y + h, band_y0 + BAND_ROWS); iy++) {
		for (int ix = x; ix < x + w; ix++) {
			put(ix, iy, c);
		}
	}
}

void screen_text(int x, int y, const char *s, int scale, uint16_t color)
{
	uint16_t c = pack(color);

	if (y + SCREEN_FONT_H * scale <= band_y0 || y >= band_y0 + BAND_ROWS) {
		return;
	}
	for (; *s != '\0'; s++, x += SCREEN_FONT_W * scale) {
		unsigned int ch = (unsigned char)*s;

		if (ch < FONT_FIRST || ch > FONT_LAST) {
			ch = '?';
		}
		const uint8_t *glyph = kFont6x8[ch - FONT_FIRST];

		for (int cx = 0; cx < FONT_W; cx++) {
			for (int cy = 0; cy < FONT_H; cy++) {
				if (((glyph[cx] >> cy) & 1) == 0) {
					continue;
				}
				for (int sy = 0; sy < scale; sy++) {
					for (int sx = 0; sx < scale; sx++) {
						put(x + cx * scale + sx, y + cy * scale + sy, c);
					}
				}
			}
		}
	}
}

void screen_rgb565(int x, int y, const uint16_t *img, int w, int h, int scale)
{
	if (y + h * scale <= band_y0 || y >= band_y0 + BAND_ROWS) {
		return;
	}
	for (int iy = 0; iy < h; iy++) {
		int py = y + iy * scale;

		if (py + scale <= band_y0 || py >= band_y0 + BAND_ROWS) {
			continue;
		}
		for (int ix = 0; ix < w; ix++) {
			uint16_t c = pack(img[iy * w + ix]);

			for (int sy = 0; sy < scale; sy++) {
				for (int sx = 0; sx < scale; sx++) {
					put(x + ix * scale + sx, py + sy, c);
				}
			}
		}
	}
}

void screen_frame(int x, int y, int w, int h, int t, uint16_t color)
{
	screen_fill(x, y, w, t, color);
	screen_fill(x, y + h - t, w, t, color);
	screen_fill(x, y, t, h, color);
	screen_fill(x + w - t, y, t, h, color);
}

void screen_gray(int x, int y, const uint8_t *img, int w, int h, int scale)
{
	if (y + h * scale <= band_y0 || y >= band_y0 + BAND_ROWS) {
		return;
	}
	for (int iy = 0; iy < h; iy++) {
		for (int ix = 0; ix < w; ix++) {
			uint8_t v = img[iy * w + ix];
			uint16_t c = pack(RGB565(v, v, v));

			for (int sy = 0; sy < scale; sy++) {
				for (int sx = 0; sx < scale; sx++) {
					put(x + ix * scale + sx, y + iy * scale + sy, c);
				}
			}
		}
	}
}
