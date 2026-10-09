/* Copyright (c) 2026 Eoin Jordan
 * SPDX-License-Identifier: Apache-2.0
 *
 * ExecuTorch on the Seeed XIAO ESP32S3 (Sense) under Zephyr RTOS.
 *
 * 1. Load the .pte embedded in flash (model_pte.h, generated at build time).
 * 2. Run it once on an MNIST test image baked into test_input.h and compare
 *    the logits with the PyTorch reference, so a fresh board can prove the
 *    whole train -> export -> embed -> execute path works.
 * 3. With CONFIG_APP_CAMERA, keep classifying frames from the OV2640.
 */

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/executor/memory_manager.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>

#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <zephyr/version.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "model_pte.h"  /* generated: model_pte[], model_pte_len */
#include "test_input.h" /* generated: kTestInput[], kTestExpectedLogits[] */

#ifdef CONFIG_APP_CAMERA
#include "camera.h"
#endif

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::extension::BufferDataLoader;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorch::runtime::TensorInfo;

/* Same constants as train/model.py */
static constexpr float kMean = 0.1307f;
static constexpr float kStd = 0.3081f;
static constexpr size_t kNumClasses = 10;

#ifndef APP_METHOD_ALLOCATOR_POOL_SIZE
#define APP_METHOD_ALLOCATOR_POOL_SIZE (128 * 1024)
#endif
#ifndef APP_TEMP_ALLOCATOR_POOL_SIZE
#define APP_TEMP_ALLOCATOR_POOL_SIZE (16 * 1024)
#endif

/* Static pools: everything the runtime needs lives here, nothing in flash
 * is copied (the .pte is read in place from memory-mapped flash). */
alignas(16) static uint8_t method_pool[APP_METHOD_ALLOCATOR_POOL_SIZE];
alignas(16) static uint8_t temp_pool[APP_TEMP_ALLOCATOR_POOL_SIZE];

namespace {

struct Runner {
	BufferDataLoader loader{model_pte, model_pte_len};
	MemoryAllocator method_allocator{APP_METHOD_ALLOCATOR_POOL_SIZE, method_pool};
	MemoryAllocator temp_allocator{APP_TEMP_ALLOCATOR_POOL_SIZE, temp_pool};
	std::vector<Span<uint8_t>> planned_spans;
	Program *program = nullptr;
	Method *method = nullptr;
	float *input = nullptr;
	size_t input_numel = 0;
	Tensor *input_tensor = nullptr;

	Error init();
	Error run(const float *pixels, float *logits, uint32_t *us);
};

Error Runner::init()
{
	Result<Program> program_res = Program::load(&loader);
	if (!program_res.ok()) {
		ET_LOG(Error, "Program::load failed: 0x%x", (unsigned)program_res.error());
		return program_res.error();
	}
	program = new Program(std::move(program_res.get()));

	const char *method_name = program->get_method_name(0).get();
	Result<MethodMeta> meta = program->method_meta(method_name);
	if (!meta.ok()) {
		return meta.error();
	}

	for (size_t id = 0; id < meta->num_memory_planned_buffers(); ++id) {
		size_t size = static_cast<size_t>(meta->memory_planned_buffer_size(id).get());
		uint8_t *buf = static_cast<uint8_t *>(method_allocator.allocate(size, 16));
		if (buf == nullptr) {
			ET_LOG(Error, "planned buffer %zu (%zu B) does not fit the method pool", id, size);
			return Error::MemoryAllocationFailed;
		}
		ET_LOG(Info, "planned buffer %zu: %zu bytes", id, size);
		planned_spans.push_back({buf, size});
	}

	static HierarchicalAllocator planned_memory({planned_spans.data(), planned_spans.size()});
	static MemoryManager memory_manager(&method_allocator, &planned_memory, &temp_allocator);

	Result<Method> method_res = program->load_method(method_name, &memory_manager);
	if (!method_res.ok()) {
		ET_LOG(Error, "load_method(%s) failed: 0x%x", method_name, (unsigned)method_res.error());
		return method_res.error();
	}
	method = new Method(std::move(method_res.get()));

	/* One input tensor, allocated once and reused for every inference. */
	Result<TensorInfo> in_meta = meta->input_tensor_meta(0);
	if (!in_meta.ok()) {
		return in_meta.error();
	}
	input_numel = in_meta->nbytes() / sizeof(float);
	input = static_cast<float *>(method_allocator.allocate(in_meta->nbytes(), 16));
	if (input == nullptr) {
		return Error::MemoryAllocationFailed;
	}
	static TensorImpl impl(
		in_meta->scalar_type(), in_meta->sizes().size(),
		const_cast<TensorImpl::SizesType *>(in_meta->sizes().data()), input,
		const_cast<TensorImpl::DimOrderType *>(in_meta->dim_order().data()));
	static Tensor tensor(&impl);
	input_tensor = &tensor;

	ET_LOG(Info, "method '%s' loaded; pool used %zu / %zu bytes", method_name,
	       method_allocator.used_size(), method_allocator.size());
	return Error::Ok;
}

Error Runner::run(const float *pixels, float *logits, uint32_t *us)
{
	std::memcpy(input, pixels, input_numel * sizeof(float));
	Error err = method->set_input(EValue(*input_tensor), 0);
	if (err != Error::Ok) {
		ET_LOG(Error, "set_input failed: 0x%x", (unsigned)err);
		return err;
	}

	uint32_t t0 = k_cycle_get_32();
	err = method->execute();
	uint32_t t1 = k_cycle_get_32();
	if (err != Error::Ok) {
		ET_LOG(Error, "execute failed: 0x%x", (unsigned)err);
		return err;
	}
	*us = k_cyc_to_us_floor32(t1 - t0);

	const EValue &out = method->get_output(0);
	if (!out.isTensor()) {
		return Error::InvalidState;
	}
	const Tensor &t = out.toTensor();
	const float *data = t.const_data_ptr<float>();
	for (size_t i = 0; i < kNumClasses && i < static_cast<size_t>(t.numel()); ++i) {
		logits[i] = data[i];
	}
	return Error::Ok;
}

size_t argmax(const float *v, size_t n)
{
	size_t best = 0;
	for (size_t i = 1; i < n; ++i) {
		if (v[i] > v[best]) {
			best = i;
		}
	}
	return best;
}

void print_logits(const float *logits)
{
	for (size_t i = 0; i < kNumClasses; ++i) {
		printk("  logit[%zu] = %9.4f%s\n", i, static_cast<double>(logits[i]),
		       i == argmax(logits, kNumClasses) ? "  <-- argmax" : "");
	}
}

#ifdef CONFIG_APP_CAMERA
void print_preview(const uint8_t *img)
{
	for (int y = 0; y < APP_IMG_SIDE; ++y) {
		char line[APP_IMG_SIDE + 1];
		for (int x = 0; x < APP_IMG_SIDE; ++x) {
			uint8_t v = img[y * APP_IMG_SIDE + x];
			line[x] = v > 160 ? '#' : (v > 90 ? '+' : '.');
		}
		line[APP_IMG_SIDE] = '\0';
		printk("  %s\n", line);
	}
}
#endif

} // namespace

int main(void)
{
	executorch::runtime::runtime_init();

	printk("\nExecuTorch on %s (Zephyr %s)\n", CONFIG_BOARD_TARGET, KERNEL_VERSION_STRING);
	printk("model: %u bytes in flash, method pool %u B, temp pool %u B\n",
	       (unsigned)model_pte_len, (unsigned)APP_METHOD_ALLOCATOR_POOL_SIZE,
	       (unsigned)APP_TEMP_ALLOCATOR_POOL_SIZE);

	static Runner runner;
	if (runner.init() != Error::Ok) {
		printk("FAIL: runtime init\n");
		return 1;
	}

	/* --- 1. Self-test on the embedded MNIST image --------------------- */
	float logits[kNumClasses];
	uint32_t us = 0;

	if (runner.run(kTestInput, logits, &us) != Error::Ok) {
		printk("FAIL: inference\n");
		return 1;
	}

	size_t pred = argmax(logits, kNumClasses);
	float max_diff = 0.f;
	for (size_t i = 0; i < kNumClasses; ++i) {
		max_diff = fmaxf(max_diff, fabsf(logits[i] - kTestExpectedLogits[i]));
	}

	printk("\nself-test: embedded MNIST image, label %d\n", TEST_INPUT_LABEL);
	print_logits(logits);
	printk("  predicted %zu, expected %d, max |esp32 - pytorch| = %.2e, %u us\n", pred,
	       TEST_INPUT_LABEL, static_cast<double>(max_diff), us);
	if (pred == TEST_INPUT_LABEL && max_diff < 1e-3f) {
		printk("SELF-TEST PASS\n");
	} else {
		printk("SELF-TEST FAIL\n");
	}

	/* Warm timing: run a few more times and report the average. */
	uint32_t total = 0;
	const int reps = 10;
	for (int i = 0; i < reps; ++i) {
		runner.run(kTestInput, logits, &us);
		total += us;
	}
	printk("inference: %u us average over %d runs\n", total / reps, reps);

	/* --- 2. Optional live camera loop ---------------------------------- */
#ifdef CONFIG_APP_CAMERA
	if (app_camera_init() != 0) {
		printk("camera init failed; stopping after self-test\n");
		return 0;
	}
	static float frame[APP_IMG_SIDE * APP_IMG_SIDE];
	static uint8_t preview[APP_IMG_SIDE * APP_IMG_SIDE];
	for (;;) {
		if (app_camera_capture(frame, preview, kMean, kStd) == 0 &&
		    runner.run(frame, logits, &us) == Error::Ok) {
			printk("\ncamera frame:\n");
			print_preview(preview);
			printk("  digit %zu (%u us)\n", argmax(logits, kNumClasses), us);
		}
		k_msleep(CONFIG_APP_CAMERA_INTERVAL_MS);
	}
#endif
	return 0;
}
