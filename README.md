# ExecuTorch on the XIAO ESP32S3 Sense with Zephyr

Train a PyTorch model, export it with [ExecuTorch](https://github.com/pytorch/executorch),
and run it on a Seeed Studio **XIAO ESP32S3 (Sense)** under **Zephyr RTOS** using the
[ExecuTorch Zephyr module](https://docs.zephyrproject.org/latest/develop/manifest/external/executorch.html).

The upstream module and its `hello-executorch` sample are written around Arm Cortex-M and
Ethos-U. This repo shows the same module building for an **Xtensa LX7** part with no Arm
pieces at all: fp32 portable kernels, selective operator linking, the model embedded in
flash, and an optional live loop that classifies frames from the Sense board's OV2640
camera.

```
PyTorch (TinyDigitNet, MNIST)  ──torch.export──▶  .pte (portable ops, 63 KB)
                                                       │
            firmware/CMakeLists.txt: gen_oplist ◀──────┤  only the 5 ops in the
            + pte_to_header.py  ──▶ model_pte.h        │  model get linked
                                                       ▼
                 west build -b xiao_esp32s3/esp32s3/procpu/sense  ──▶  zephyr.bin
```

## What you get

| Piece | Where | Notes |
|---|---|---|
| Model + training | `train/model.py`, `train/train_mnist.py` | 14 410 params, ~97.5 % MNIST after 2 CPU epochs |
| Export to `.pte` | `train/train_mnist.py` (`export_pte`) | `to_edge_transform_and_lower` with no delegate |
| Host sanity check | `train/run_pte_host.py` | runs the `.pte` with the ExecuTorch Python runtime |
| Zephyr app | `firmware/` | self-test on an embedded MNIST image, then optional camera loop |
| Workspace manifest | `west.yml` | Zephyr v4.4.0 + `hal_espressif` + ExecuTorch v1.5.1 as a module |
| CI | `.github/workflows/build.yml` | trains a smoke model and builds all three firmware variants |

Pre-built outputs are committed so you can flash without training:
`firmware/model/mnist_tiny.pte` and `firmware/src/test_input.h`.

## Hardware

- Seeed Studio XIAO ESP32S3 Sense (ESP32-S3R8, 8 MB flash, 8 MB PSRAM, OV2640 camera, PDM mic).
  The plain XIAO ESP32S3 works for everything except the camera mode.
- USB-C cable. Flashing is over the native USB-serial, no extra adapter.

Zephyr board targets:

| Target | Use |
|---|---|
| `xiao_esp32s3/esp32s3/procpu` | plain board, self-test only |
| `xiao_esp32s3/esp32s3/procpu/sense` | Sense board, self-test; add `camera.conf` for the live loop |

## Quick start

### 1. Host setup

Python 3.12 or 3.13 on Linux or macOS (WSL2 is fine). One venv serves training, export and
`west`, because the firmware build calls ExecuTorch's `codegen.tools.gen_oplist` at
configure time to work out which kernels to link.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

### 2. Train and export

```bash
python train/train_mnist.py            # ~1-2 min on CPU, downloads MNIST (~10 MB)
python train/run_pte_host.py           # optional: run the .pte on the host
```

`train_mnist.py` writes:

- `firmware/model/mnist_tiny.pte`, the ExecuTorch program. The operator list it prints
  is exactly what gets linked into the firmware:
  `aten.convolution.out`, `aten.max_pool2d_with_indices.out`, `aten.relu.out`,
  `aten.permute_copy.out`, `aten.addmm.out`.
- `firmware/src/test_input.h`, one MNIST test image (label 7) and the PyTorch logits for
  it, so the board can prove end-to-end numerical agreement on first boot.

To use your own model: keep the input as a single fp32 tensor, export the same way, and point
the build at it with `-DET_PTE_FILE_PATH=/path/to/model.pte`. Anything the portable kernel
library implements will link; there is no Xtensa backend, so quantised graphs would need the
quantized kernel library added to the selective build (see "Going further").

### 3. Zephyr workspace

The manifest in this repo creates a minimal workspace: Zephyr, the Espressif HAL and
ExecuTorch (as `modules/lib/executorch`). Do this in an empty directory, with the venv
active.

```bash
west init -m https://github.com/eoinjordan/executorch-zephyr-xiao-esp32s3 ws
cd ws
west update                          # ExecuTorch has large submodules; a few minutes
west blobs fetch hal_espressif       # Espressif binary blobs (Wi-Fi/BT libs, needed to link)
west sdk install --toolchains xtensa-espressif_esp32s3_zephyr-elf
```

If you already have a Zephyr workspace, copy `modules/lib/executorch` registration from
`west.yml` into `zephyr/submanifests/executorch.yaml` instead, as the Zephyr docs describe,
and add `+executorch` to your `manifest.project-filter`.

### 4. Build, flash, watch

```bash
cd ws
west build -p always -b xiao_esp32s3/esp32s3/procpu/sense executorch-zephyr-xiao-esp32s3/firmware
west flash
west espressif monitor
```

Expected serial output (abridged):

```
ExecuTorch on xiao_esp32s3/esp32s3/procpu/sense (Zephyr 4.4.0)
model: 62888 bytes in flash, method pool 131072 B, temp pool 16384 B
I [executorch:main.cpp:...] planned buffer 0: ... bytes
I [executorch:main.cpp:...] method 'forward' loaded; pool used ... / 131072 bytes

self-test: embedded MNIST image, label 7
  logit[0] =   -1.2525
  ...
  logit[7] =   10.5990  <-- argmax
  ...
  predicted 7, expected 7, max |esp32 - pytorch| = 3.3e-06, ... us
SELF-TEST PASS
inference: ... us average over 10 runs
```

If the chip stays in download mode after flashing, use `west flash --reset-type watchdog-reset`.

### 5. Live camera mode (Sense board)

```bash
west build -p always -b xiao_esp32s3/esp32s3/procpu/sense executorch-zephyr-xiao-esp32s3/firmware -- -DEXTRA_CONF_FILE=camera.conf
west flash && west espressif monitor
```

The firmware streams 160x120 RGB565 from the OV2640 through Zephyr's video API, centre-crops
to 120x120, box-downsamples to 28x28, inverts (MNIST digits are light on dark, so write a dark
digit on white paper), normalises with the training mean and std, and classifies once a
second. It prints a 28x28 ASCII preview of what the model saw next to each prediction, which
is the quickest way to tune framing and lighting. If the preview looks like noise, set
`CONFIG_APP_CAMERA_SWAP_BYTES=y`.

This is a demonstration of the sensor-to-tensor path, not a robust digit reader: the model was
trained on MNIST, so expect it to be picky about stroke width and contrast.

## Verify

Three checks, each proving one stage of the pipeline:

1. **Export is faithful.** `python train/run_pte_host.py` runs the committed `.pte` with the
   ExecuTorch Python runtime and asserts the logits match the PyTorch reference in
   `firmware/src/test_input.h` to within 1e-3 (measured: 3.3e-06).
2. **Firmware builds.** `west build` for the three variants in `.github/workflows/build.yml`
   finishes with the Zephyr memory report and no warnings from the ExecuTorch tree.
3. **Runs without hardware.** The app also builds for `native_sim/native/64`
   (`west build -b native_sim/native/64 executorch-zephyr-xiao-esp32s3/firmware`, then run
   `build/zephyr/zephyr.exe`) to try the same self-test on your PC.
4. **Device agrees with the host.** The serial log shows `SELF-TEST PASS`, which requires the
   predicted class to equal the embedded label and the ESP32's logits to match PyTorch to
   within 1e-3.

## How the firmware is put together

`firmware/CMakeLists.txt` mirrors the upstream `zephyr/samples/hello-executorch` layout but
drops everything Arm-specific:

- `CONFIG_EXECUTORCH=y` in `prj.conf` turns on the module (`modules/lib/executorch/zephyr`),
  which builds `executorch_core` with the Zephyr PAL and links it as `libexecutorch`.
- `CONFIG_EXECUTORCH_BUILD_PORTABLE_OPS=n` stops the module linking every portable operator.
  The app instead calls `arm_runner_create_selected_ops_lib` (despite its name a target-agnostic
  wrapper over `gen_selected_ops` / `gen_operators_lib`) with `OPS_FROM_MODEL` pointing at the
  `.pte`, so only the kernels the model needs end up in the image.
- `scripts/pte_to_header.py` turns the `.pte` into a 16-byte aligned `const uint8_t[]` in
  `.rodata`. On the ESP32-S3 that is memory-mapped flash, so the model costs no SRAM: the
  runtime reads it in place through a `BufferDataLoader`.
- `src/main.cpp` allocates two static pools (method and temp), loads the `Program`, sizes the
  memory-planned buffers from `MethodMeta`, loads the `Method` once and reuses it for every
  inference. Timing uses `k_cycle_get_32`.
- `src/camera.c` is plain C against `zephyr/drivers/video.h` and is only compiled with
  `CONFIG_APP_CAMERA` (set by `camera.conf`).

Differences from the Arm sample worth knowing about if you compare the two:

| Arm `hello-executorch` | This sample | Why |
|---|---|---|
| `CONFIG_FP_HARDABI=y` | omitted | Arm-only Kconfig; `CONFIG_FPU=y` is enough for the LX7 FPU |
| `cortex_m_ops_lib`, CMSIS-NN | omitted | no such backend for Xtensa |
| `arm_memory_allocator.cpp` | `MemoryAllocator` from the runtime | the Arm helper only adds usage statistics |
| `pte_to_header.py --section network_model_sec` | own script, plain `.rodata` | the custom section name is only placed by the Arm linker scripts |
| copies the model into SRAM for the NPU DMA | reads it from flash | no DMA engine needs it |

## Memory and performance

Measured with the committed model, Zephyr v4.4.0, ExecuTorch v1.5.1, Zephyr SDK 1.0.1
(xtensa-espressif gcc 14.3), `-Os`. From the linker's memory report:

| Region | self-test build | with `camera.conf` | Notes |
|---|---|---|---|
| FLASH image (`zephyr.bin`) | 398 716 B | 404 124 B | runtime + 5 kernels + 63 KB model |
| `dram0_0_seg` (SRAM) | 251 208 B / 399 108 B | 363 888 B / 399 108 B | includes the 128 KB method pool, 16 KB temp pool, 32 KB heap, 16 KB main stack; camera adds 80 KB of frame buffers |
| `iram0_0_seg` | 35 396 B | 47 088 B | |
| `irom0_0_seg` (code in flash) | 140 151 B | 155 427 B | |
| `drom0_0_seg` (rodata in flash) | 333 180 B | 338 588 B | the `.pte` lives here |

Only the five operator kernels used by the model are linked; `zephyr.map` lists
`op_addmm`, `op_convolution`, `op_max_pool2d_with_indices`, `op_permute_copy` and
`op_relu` from `libportable_kernels.a` and nothing else.

The method pool is generous for this model. The boot log prints `planned buffer 0: N bytes`
and `pool used X / Y`; set `CONFIG_EXECUTORCH_METHOD_ALLOCATOR_POOL_SIZE` a little above X
if you need the SRAM back.

The `inference:` line in the boot log is the number to compare against other runtimes on the
same board.

## Going further

- **Quantise.** Export an int8 graph with `XNNPACKQuantizer`-style PT2E quantisation and add
  `kernels/quantized/quantized.yaml` as a second `arm_runner_create_selected_ops_lib` call
  (the module already builds `quantized_kernels`). Expect roughly 4x smaller weights and a
  solid speed-up from the LX7's SIMD-ish integer paths.
- **Bigger inputs.** Raise `CONFIG_EXECUTORCH_METHOD_ALLOCATOR_POOL_SIZE` (the planned buffer
  size is printed at boot) and, for anything beyond a few hundred KB, put the pool in PSRAM
  with the `espressif-psram-8M` snippet.
- **Audio.** The Sense board's PDM mic is on GPIO41/42; Zephyr's `dmic` API plus a KWS model
  exported the same way is the natural next sample.
- **Upstream.** `firmware/` is laid out to drop into `zephyr/samples/` in the ExecuTorch repo.
  See `docs/upstream-pr.md`.

## Troubleshooting

- `ExecuTorch module not found`: `west list executorch` must show
  `modules/lib/executorch`. Re-run `west update`.
- `No module named codegen` / `gen_oplist` errors at configure time: the `executorch` pip
  package is not in the Python that `west` is using. Activate the venv from step 1.
- Link errors about Espressif Wi-Fi/BT symbols: run `west blobs fetch hal_espressif`.
- `Could not allocate memory for memory planned buffer`: raise
  `CONFIG_EXECUTORCH_METHOD_ALLOCATOR_POOL_SIZE` in `prj.conf`.
- `Missing operator` at runtime: the `.pte` used at build time differs from the one you
  flashed, or the model uses an op the portable library lacks. Rebuild with `-p always` and
  check the operator list printed by `train_mnist.py`.

## License

Apache-2.0. See [LICENSE](LICENSE). ExecuTorch is BSD-3-Clause and Zephyr is Apache-2.0.
