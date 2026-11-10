/* Copyright (c) 2026 Eoin Jordan
 * SPDX-License-Identifier: Apache-2.0
 *
 * ExecuTorch on the Seeed XIAO ESP32S3 (Sense) under Zephyr RTOS.
 *
 * 1. Load the programs embedded in flash (model_pte.h and, unless the
 *    firmware was built with -DAPP_CMSIS_NN=OFF, model_q_pte.h; both are
 *    generated at build time). They are the same network: one in fp32 on the
 *    portable kernels, one in int8 on the CMSIS-NN kernels.
 * 2. Run each once on an MNIST test image baked into test_input.h and
 *    compare the logits with the PyTorch reference, so a fresh board can
 *    prove the whole train -> export -> embed -> execute path works.
 * 3. With CONFIG_APP_CAMERA, keep classifying frames from the OV2640, and
 *    with CONFIG_APP_SCREEN show the frame, the digit and both times on the
 *    round display.
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

#ifdef APP_CMSIS_NN
#include "model_q_pte.h"  /* generated: model_q_pte[], model_q_pte_len */
#include "test_input_q.h" /* generated: kTestExpectedLogitsQ[], TEST_LOGIT_STEP_Q */
#endif

#ifdef CONFIG_APP_CAMERA
#include "camera.h"
#endif
#ifdef CONFIG_APP_SCREEN
#include "screen.h"
#endif

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
static constexpr int kSide = 28;

/* Static pools: everything the runtime needs lives here, nothing in flash
 * is copied (a .pte is read in place from memory-mapped flash). Each program
 * has its own pair. */
alignas(16) static uint8_t method_pool[APP_METHOD_ALLOCATOR_POOL_SIZE];
alignas(16) static uint8_t temp_pool[APP_TEMP_ALLOCATOR_POOL_SIZE];
#ifdef APP_CMSIS_NN
alignas(16) static uint8_t q_method_pool[APP_Q_METHOD_ALLOCATOR_POOL_SIZE];
alignas(16) static uint8_t q_temp_pool[APP_Q_TEMP_ALLOCATOR_POOL_SIZE];
#endif

namespace {

/* One loaded program with its memory, ready to run. */
class Runner {
      public:
	Runner(const char *name, const uint8_t *pte, size_t pte_len, uint8_t *method_mem,
	       size_t method_size, uint8_t *temp_mem, size_t temp_size)
		: name(name), pte_len(pte_len), loader_(pte, pte_len),
		  method_allocator_(method_size, method_mem), temp_allocator_(temp_size, temp_mem)
	{
	}

	Error init();
	/* 784 normalised pixels in, 10 logits and the time of execute() out. */
	Error run(const float *pixels, float *logits, uint32_t *us);

	const char *const name;
	const size_t pte_len;

      private:
	BufferDataLoader loader_;
	MemoryAllocator method_allocator_;
	MemoryAllocator temp_allocator_;
	std::vector<Span<uint8_t>> planned_spans_;
	Program *program_ = nullptr;
	Method *method_ = nullptr;
	Tensor *input_tensor_ = nullptr;
	float *input_ = nullptr;
	size_t input_numel_ = 0;
};

Error Runner::init()
{
	Result<Program> program_res = Program::load(&loader_);
	if (!program_res.ok()) {
		ET_LOG(Error, "%s: Program::load failed: 0x%x", name, (unsigned)program_res.error());
		return program_res.error();
	}
	program_ = new Program(std::move(program_res.get()));

	const char *method_name = program_->get_method_name(0).get();
	Result<MethodMeta> meta = program_->method_meta(method_name);
	if (!meta.ok()) {
		return meta.error();
	}

	size_t planned = 0;
	for (size_t id = 0; id < meta->num_memory_planned_buffers(); ++id) {
		size_t size = static_cast<size_t>(meta->memory_planned_buffer_size(id).get());
		uint8_t *buf = static_cast<uint8_t *>(method_allocator_.allocate(size, 16));
		if (buf == nullptr) {
			ET_LOG(Error, "%s: planned buffer %zu (%zu B) does not fit the method pool",
			       name, id, size);
			return Error::MemoryAllocationFailed;
		}
		planned += size;
		planned_spans_.push_back({buf, size});
	}

	auto *planned_memory =
		new HierarchicalAllocator({planned_spans_.data(), planned_spans_.size()});
	auto *memory_manager =
		new MemoryManager(&method_allocator_, planned_memory, &temp_allocator_);

	Result<Method> method_res = program_->load_method(method_name, memory_manager);
	if (!method_res.ok()) {
		ET_LOG(Error, "%s: load_method(%s) failed: 0x%x", name, method_name,
		       (unsigned)method_res.error());
		return method_res.error();
	}
	method_ = new Method(std::move(method_res.get()));

	/* One input tensor, allocated once and reused for every inference. Both
	 * programs take float pixels: the int8 one starts with a quantize
	 * operator. Its input is in NHWC order, which for one channel is the
	 * same bytes as NCHW. */
	Result<TensorInfo> in_meta = meta->input_tensor_meta(0);
	if (!in_meta.ok()) {
		return in_meta.error();
	}
	input_numel_ = in_meta->nbytes() / sizeof(float);
	input_ = static_cast<float *>(method_allocator_.allocate(in_meta->nbytes(), 16));
	if (input_ == nullptr || input_numel_ != kSide * kSide) {
		return Error::MemoryAllocationFailed;
	}
	auto *impl = new TensorImpl(
		in_meta->scalar_type(), in_meta->sizes().size(),
		const_cast<TensorImpl::SizesType *>(in_meta->sizes().data()), input_,
		const_cast<TensorImpl::DimOrderType *>(in_meta->dim_order().data()));
	input_tensor_ = new Tensor(impl);

	printk("%-9s %6u bytes in flash, %6u planned, pool %6u of %6u bytes used\n", name,
	       (unsigned)pte_len, (unsigned)planned, (unsigned)method_allocator_.used_size(),
	       (unsigned)method_allocator_.size());
	return Error::Ok;
}

Error Runner::run(const float *pixels, float *logits, uint32_t *us)
{
	std::memcpy(input_, pixels, input_numel_ * sizeof(float));
	Error err = method_->set_input(EValue(*input_tensor_), 0);
	if (err != Error::Ok) {
		ET_LOG(Error, "%s: set_input failed: 0x%x", name, (unsigned)err);
		return err;
	}

	uint32_t t0 = k_cycle_get_32();
	err = method_->execute();
	uint32_t t1 = k_cycle_get_32();
	if (err != Error::Ok) {
		ET_LOG(Error, "%s: execute failed: 0x%x", name, (unsigned)err);
		return err;
	}
	*us = k_cyc_to_us_floor32(t1 - t0);

	const EValue &out = method_->get_output(0);
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

/* Run the embedded test image, compare with the reference and time it.
 * `tolerance` is the largest difference from the reference that passes.
 * Returns the average time of an inference in microseconds, 0 on failure. */
uint32_t self_test(Runner &runner, const float *expected, float tolerance, size_t *digit)
{
	float logits[kNumClasses];
	uint32_t us = 0;

	if (runner.run(kTestInput, logits, &us) != Error::Ok) {
		printk("%s: FAIL: inference\n", runner.name);
		return 0;
	}
	size_t pred = argmax(logits, kNumClasses);
	float max_diff = 0.f;
	for (size_t i = 0; i < kNumClasses; ++i) {
		max_diff = fmaxf(max_diff, fabsf(logits[i] - expected[i]));
	}

	printk("\n%s self-test: embedded MNIST image, label %d\n", runner.name, TEST_INPUT_LABEL);
	print_logits(logits);
	printk("  predicted %zu, expected %d, max |esp32 - pytorch| = %.2e (limit %.2e)\n", pred,
	       TEST_INPUT_LABEL, static_cast<double>(max_diff), static_cast<double>(tolerance));
	bool pass = pred == TEST_INPUT_LABEL && max_diff <= tolerance;
	printk("%s SELF-TEST %s\n", runner.name, pass ? "PASS" : "FAIL");

	/* Warm timing: run a few more times and report the average. */
	uint32_t total = 0;
	const int reps = 10;
	for (int i = 0; i < reps; ++i) {
		runner.run(kTestInput, logits, &us);
		total += us;
	}
	printk("%s inference: %u us average over %d runs\n", runner.name, total / reps, reps);
	*digit = pred;
	return pass ? total / reps : 0;
}

#if defined(CONFIG_APP_CAMERA) && !defined(CONFIG_APP_SCREEN)
void print_preview(const uint8_t *img)
{
	for (int y = 0; y < kSide; ++y) {
		char line[kSide + 1];
		for (int x = 0; x < kSide; ++x) {
			uint8_t v = img[y * kSide + x];
			line[x] = v > 160 ? '#' : (v > 90 ? '+' : '.');
		}
		line[kSide] = '\0';
		printk("  %s\n", line);
	}
}
#endif

#ifdef CONFIG_APP_SCREEN
/* What the screen shows. The layout keeps inside the circle of the panel. */
struct View {
	const uint8_t *image; /* 28 x 28, what the model was given */
	const char *caption;
	int digit;
	uint32_t us;   /* fp32 on the portable kernels */
	int digit_q;   /* -1 without the int8 program */
	uint32_t us_q; /* int8 on CMSIS-NN */
};

void draw_view(void *ctx)
{
	const View *v = static_cast<const View *>(ctx);
	const int shown = v->digit_q >= 0 ? v->digit_q : v->digit;
	char text[24];

	screen_text(60, 24, "ExecuTorch", 2, COLOR_WHITE);
	screen_text(120 - 3 * static_cast<int>(strlen(v->caption)), 46, v->caption, 1, COLOR_GREY);

	screen_gray(30, 62, v->image, kSide, kSide, 3);
	text[0] = static_cast<char>('0' + shown);
	text[1] = '\0';
	screen_text(142, 72, text, 8, COLOR_AMBER);

	if (v->digit_q >= 0) {
		snprintk(text, sizeof(text), "CMSIS %d %u.%02ums", v->digit_q, v->us_q / 1000,
			 (v->us_q % 1000) / 10);
		screen_text(36, 156, text, 2, COLOR_LIME);
	}
	snprintk(text, sizeof(text), "FP32  %d %u.%02ums", v->digit, v->us / 1000,
		 (v->us % 1000) / 10);
	screen_text(36, 176, text, 2, COLOR_WHITE);

	/* Red, green, blue from left to right: a check of the colour order. */
	screen_fill(92, 202, 14, 14, COLOR_RED);
	screen_fill(113, 202, 14, 14, COLOR_GREEN);
	screen_fill(134, 202, 14, 14, COLOR_BLUE);
}
#endif

} // namespace

int main(void)
{
	executorch::runtime::runtime_init();

	printk("\nExecuTorch on %s (Zephyr %s)\n", CONFIG_BOARD_TARGET, KERNEL_VERSION_STRING);

	static Runner fp32("portable", model_pte, model_pte_len, method_pool, sizeof(method_pool),
			   temp_pool, sizeof(temp_pool));
	if (fp32.init() != Error::Ok) {
		printk("FAIL: runtime init (portable)\n");
		return 1;
	}
#ifdef APP_CMSIS_NN
	static Runner int8("cmsis-nn", model_q_pte, model_q_pte_len, q_method_pool,
			   sizeof(q_method_pool), q_temp_pool, sizeof(q_temp_pool));
	if (int8.init() != Error::Ok) {
		printk("FAIL: runtime init (cmsis-nn)\n");
		return 1;
	}
#endif

	/* --- 1. Self-test on the embedded MNIST image --------------------- */
	size_t digit = 0;
	uint32_t us = self_test(fp32, kTestExpectedLogits, 1e-3f, &digit);
#ifdef APP_CMSIS_NN
	/* Integer kernels may round a logit one step away from the PyTorch
	 * reference; anything beyond two steps is a real difference. */
	size_t digit_q = 0;
	uint32_t us_q = self_test(int8, kTestExpectedLogitsQ, 2.5f * TEST_LOGIT_STEP_Q, &digit_q);
	if (us != 0 && us_q != 0) {
		printk("\nint8 on CMSIS-NN is %u.%02ux the speed of fp32 on the portable kernels, "
		       "in %u%% of the flash\n",
		       us / us_q, (us % us_q) * 100 / us_q,
		       (unsigned)(model_q_pte_len * 100 / model_pte_len));
	}
#endif

#ifdef CONFIG_APP_SCREEN
	static uint8_t shown[kSide * kSide];
	static View view;

	for (int i = 0; i < kSide * kSide; ++i) {
		float v = (kTestInput[i] * kStd + kMean) * 255.f;
		shown[i] = static_cast<uint8_t>(fminf(fmaxf(v, 0.f), 255.f));
	}
	view.image = shown;
	view.caption = "self-test image";
	view.digit = static_cast<int>(digit);
	view.us = us;
	view.digit_q = -1;
#ifdef APP_CMSIS_NN
	view.digit_q = static_cast<int>(digit_q);
	view.us_q = us_q;
#endif
	const bool have_screen = screen_init() == 0;
	if (have_screen) {
		screen_render(draw_view, &view);
	}
#endif

	/* --- 2. Optional live camera loop ---------------------------------- */
#ifdef CONFIG_APP_CAMERA
	if (app_camera_init() != 0) {
		printk("camera init failed; stopping after self-test\n");
		return 0;
	}
	static float frame[kSide * kSide];
	static uint8_t preview[kSide * kSide];
	float logits[kNumClasses];

	for (;;) {
		if (app_camera_capture(frame, preview, kMean, kStd) != 0 ||
		    fp32.run(frame, logits, &us) != Error::Ok) {
			k_msleep(CONFIG_APP_CAMERA_INTERVAL_MS);
			continue;
		}
		digit = argmax(logits, kNumClasses);
#ifdef APP_CMSIS_NN
		if (int8.run(frame, logits, &us_q) != Error::Ok) {
			continue;
		}
		digit_q = argmax(logits, kNumClasses);
#endif

#ifdef CONFIG_APP_SCREEN
		if (have_screen) {
			view.image = preview;
			view.caption = "camera";
			view.digit = static_cast<int>(digit);
			view.us = us;
#ifdef APP_CMSIS_NN
			view.digit_q = static_cast<int>(digit_q);
			view.us_q = us_q;
#endif
			screen_render(draw_view, &view);
		}
#else
		printk("\ncamera frame:\n");
		print_preview(preview);
#endif
#ifdef APP_CMSIS_NN
		printk("digit %zu (cmsis-nn, %u us), %zu (portable, %u us)\n", digit_q, us_q, digit, us);
#else
		printk("digit %zu (%u us)\n", digit, us);
#endif
		k_msleep(CONFIG_APP_CAMERA_INTERVAL_MS);
	}
#endif
	return 0;
}
