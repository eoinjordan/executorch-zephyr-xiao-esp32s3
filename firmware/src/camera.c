/* Copyright (c) 2026 Eoin Jordan
 * SPDX-License-Identifier: Apache-2.0
 *
 * OV2640 capture on the XIAO ESP32S3 Sense through Zephyr's video API
 * (LCD_CAM DVP receiver + ov2640 driver, both in the board devicetree).
 */

#include "camera.h"

#include <errno.h>
#include <string.h>
#include <zephyr/device.h>
#include <zephyr/drivers/video.h>
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(app_camera, CONFIG_LOG_DEFAULT_LEVEL);

#if !DT_HAS_CHOSEN(zephyr_camera)
#error "No zephyr,camera chosen in devicetree. Build for xiao_esp32s3/esp32s3/procpu/sense."
#endif

#define CAM_WIDTH  160
#define CAM_HEIGHT 120

static const struct device *const cam = DEVICE_DT_GET(DT_CHOSEN(zephyr_camera));
static struct video_format fmt = {.type = VIDEO_BUF_TYPE_OUTPUT};

static void grab_thread(void *a, void *b, void *c);
K_THREAD_STACK_DECLARE(grab_stack, 2048);
static struct k_thread grab_data;

int app_camera_init(void)
{
	struct video_caps caps = {.type = VIDEO_BUF_TYPE_OUTPUT};
	int ret;

	if (!device_is_ready(cam)) {
		LOG_ERR("%s not ready", cam->name);
		return -ENODEV;
	}

	ret = video_get_caps(cam, &caps);
	if (ret < 0) {
		LOG_ERR("video_get_caps: %d", ret);
		return ret;
	}

	ret = video_get_format(cam, &fmt);
	if (ret < 0) {
		LOG_ERR("video_get_format: %d", ret);
		return ret;
	}
	fmt.pixelformat = VIDEO_PIX_FMT_RGB565;
	fmt.width = CAM_WIDTH;
	fmt.height = CAM_HEIGHT;

	ret = video_set_compose_format(cam, &fmt);
	if (ret < 0) {
		LOG_ERR("video_set_compose_format: %d", ret);
		return ret;
	}
	/* Read back pitch/size as the driver computed them. */
	ret = video_get_format(cam, &fmt);
	if (ret < 0) {
		return ret;
	}
	LOG_INF("camera %s: %s %ux%u pitch=%u size=%u", cam->name,
		VIDEO_FOURCC_TO_STR(fmt.pixelformat), fmt.width, fmt.height, fmt.pitch,
		fmt.size);

	if (caps.min_vbuf_count > CONFIG_VIDEO_BUFFER_POOL_NUM_MAX) {
		LOG_ERR("driver needs %u buffers, pool has %u", caps.min_vbuf_count,
			CONFIG_VIDEO_BUFFER_POOL_NUM_MAX);
		return -EINVAL;
	}

	for (int i = 0; i < CONFIG_VIDEO_BUFFER_POOL_NUM_MAX; i++) {
		struct video_buffer *vbuf = video_buffer_aligned_alloc(
			fmt.size, CONFIG_VIDEO_BUFFER_POOL_ALIGN, K_NO_WAIT);

		if (vbuf == NULL) {
			LOG_ERR("video buffer alloc failed (raise CONFIG_VIDEO_BUFFER_POOL_HEAP_SIZE)");
			return -ENOMEM;
		}
		vbuf->type = VIDEO_BUF_TYPE_OUTPUT;
		ret = video_enqueue(cam, vbuf);
		if (ret < 0) {
			LOG_ERR("video_enqueue: %d", ret);
			return ret;
		}
	}

	ret = video_stream_start(cam, VIDEO_BUF_TYPE_OUTPUT);
	if (ret < 0) {
		LOG_ERR("video_stream_start: %d", ret);
		return ret;
	}
	LOG_INF("camera streaming");
	k_thread_create(&grab_data, grab_stack, K_THREAD_STACK_SIZEOF(grab_stack), grab_thread, NULL, NULL, NULL,
			K_PRIO_COOP(2), 0, K_NO_WAIT); /* above main: it must run during an inference */
	k_thread_name_set(&grab_data, "camera");
	return 0;
}

static inline uint8_t rgb565_to_gray(const uint8_t *p)
{
	uint16_t v;

	if (IS_ENABLED(CONFIG_APP_CAMERA_SWAP_BYTES)) {
		v = (uint16_t)((p[0] << 8) | p[1]);
	} else {
		v = (uint16_t)(p[0] | (p[1] << 8));
	}
	uint32_t r = ((v >> 11) & 0x1f) << 3;
	uint32_t g = ((v >> 5) & 0x3f) << 2;
	uint32_t b = (v & 0x1f) << 3;

	/* ITU-R BT.601 luma, integer weights summing to 256 */
	return (uint8_t)((77 * r + 150 * g + 29 * b) >> 8);
}

/* The newest frame, already reduced to 28x28. A thread keeps it fresh so the
 * driver always gets its buffers back at once: with two buffers it drops a
 * frame, and then stops delivering, whenever a buffer is held for longer than
 * a frame period, and one inference takes several.
 */
static uint8_t latest[APP_IMG_SIDE * APP_IMG_SIDE];
static K_MUTEX_DEFINE(latest_lock);
static K_SEM_DEFINE(latest_ready, 0, 1);

static int grab(void)
{
	struct video_buffer *vbuf = NULL;
	struct video_buffer tmp = {.type = VIDEO_BUF_TYPE_OUTPUT};
	int ret;

	vbuf = &tmp;
	ret = video_dequeue(cam, &vbuf, K_FOREVER);
	if (ret < 0) {
		LOG_ERR("video_dequeue: %d", ret);
		return ret;
	}

	const uint32_t w = fmt.width;
	const uint32_t h = fmt.height;
	const uint32_t side = (w < h) ? w : h;
	const uint32_t x0 = (w - side) / 2;
	const uint32_t y0 = (h - side) / 2;
	const uint8_t *frame = vbuf->buffer;

	static uint8_t gray[APP_IMG_SIDE * APP_IMG_SIDE];

	for (uint32_t oy = 0; oy < APP_IMG_SIDE; oy++) {
		uint32_t ya = y0 + (oy * side) / APP_IMG_SIDE;
		uint32_t yb = y0 + ((oy + 1) * side) / APP_IMG_SIDE;

		for (uint32_t ox = 0; ox < APP_IMG_SIDE; ox++) {
			uint32_t xa = x0 + (ox * side) / APP_IMG_SIDE;
			uint32_t xb = x0 + ((ox + 1) * side) / APP_IMG_SIDE;
			uint32_t sum = 0, n = 0;

			for (uint32_t y = ya; y < yb; y++) {
				const uint8_t *row = frame + y * fmt.pitch;

				for (uint32_t x = xa; x < xb; x++) {
					sum += rgb565_to_gray(row + 2 * x);
					n++;
				}
			}
			uint32_t v = n ? sum / n : 0;

			gray[oy * APP_IMG_SIDE + ox] = (uint8_t)v;
		}
	}

	ret = video_enqueue(cam, vbuf);
	if (ret < 0) {
		LOG_ERR("video_enqueue: %d", ret);
		return ret;
	}

	k_mutex_lock(&latest_lock, K_FOREVER);
	memcpy(latest, gray, sizeof(latest));
	k_mutex_unlock(&latest_lock);
	k_sem_give(&latest_ready);

	return 0;
}

static void grab_thread(void *a, void *b, void *c)
{
	ARG_UNUSED(a);
	ARG_UNUSED(b);
	ARG_UNUSED(c);

	while (grab() == 0) {
	}
}

K_THREAD_STACK_DEFINE(grab_stack, 2048);

int app_camera_capture(float *out, uint8_t *preview, float mean, float std)
{
	static uint8_t gray[APP_IMG_SIDE * APP_IMG_SIDE];
	uint32_t lo = 255, hi = 0;

	if (k_sem_take(&latest_ready, K_MSEC(2000)) != 0) {
		LOG_ERR("no frame from the camera");
		return -EIO;
	}
	k_mutex_lock(&latest_lock, K_FOREVER);
	memcpy(gray, latest, sizeof(gray));
	k_mutex_unlock(&latest_lock);

	for (uint32_t i = 0; i < APP_IMG_SIDE * APP_IMG_SIDE; i++) {
		lo = MIN(lo, (uint32_t)gray[i]);
		hi = MAX(hi, (uint32_t)gray[i]);
	}

	/* Stretch the contrast so that paper is 0 and the darkest ink 255 after
	 * inversion (MNIST is light strokes on dark), and push light paper noise
	 * to the background. A page with nothing on it stays dark.
	 */
	const uint32_t range = hi > lo + 16 ? hi - lo : 16;

	for (uint32_t i = 0; i < APP_IMG_SIDE * APP_IMG_SIDE; i++) {
		int32_t v = 255 - (int32_t)((gray[i] - lo) * 255u / range);

		if (v < 64) {
			v = 0;
		}
		if (preview) {
			preview[i] = (uint8_t)v;
		}
		out[i] = ((float)v / 255.0f - mean) / std;
	}

	return 0;
}
