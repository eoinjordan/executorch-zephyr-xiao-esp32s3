/* Copyright (c) 2026 Eoin Jordan
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define APP_IMG_SIDE 28

/* Bring up the camera chosen in devicetree (zephyr,camera) at 160x120 RGB565
 * and start streaming. Returns 0 or a negative errno.
 */
int app_camera_init(void);

/* Block for the next frame, centre-crop to a square, box-downsample to
 * 28x28 grayscale, invert (MNIST is light-on-dark) and normalise with
 * (x/255 - mean) / std into `out` (row-major, 784 floats). `preview`, if not
 * NULL, receives the 28x28 8-bit grayscale image before normalisation.
 */
int app_camera_capture(float *out, uint8_t *preview, float mean, float std);

#ifdef __cplusplus
}
#endif
