# Upstreaming to pytorch/executorch `zephyr/samples`

Goal: contribute `firmware/` as a second Zephyr sample in
[pytorch/executorch/zephyr](https://github.com/pytorch/executorch/tree/main/zephyr), proving
the module on a non-Arm (Xtensa) target, alongside `hello-executorch` and `mv2-ethosu`.

## Proposed layout in the ExecuTorch repo

```
zephyr/
  samples/
    hello-executorch/            (existing, Arm / Ethos-U)
    mv2-ethosu/                  (existing)
    xiao-esp32s3-mnist/          (this repo's firmware/)
      CMakeLists.txt
      Kconfig
      prj.conf
      camera.conf
      README.md                  (trimmed from this repo's README, sections 2-4 and "How it is put together")
      boards/                    (empty today; placeholder for esp32s3_devkitc tuning)
      models/
        tiny_digit_net.py        (train/model.py + export_pte, no torchvision dependency
                                  needed for export itself; training stays optional)
        mnist_tiny.pte           (committed, 63 KB)
      scripts/pte_to_header.py
      src/main.cpp
      src/camera.c, camera.h
      src/test_input.h
```

## Changes that belong in the module itself, not the sample

1. `zephyr/CMakeLists.txt`: the backend selection block only knows `CONFIG_CPU_CORTEX_M`,
   `CONFIG_CPU_CORTEX_A` and Ethos-U. Nothing is needed for Xtensa today (portable only), but
   a comment noting that non-Arm targets fall through to portable kernels would help the next
   reader.
2. `zephyr/samples/hello-executorch/prj.conf` sets `CONFIG_FP_HARDABI=y`, which is Arm-only
   and warns on other arches. Guard it in a `boards/*.conf` instead.
3. `backends/arm/cmake/ArmRunnerUtils.cmake` is the only public home of
   `arm_runner_create_selected_ops_lib`. A non-Arm sample using it reads oddly; suggest an
   alias in `tools/cmake/Utils.cmake` (e.g. `executorch_create_selected_ops_lib`) with the
   Arm name kept as a wrapper.
4. Docs: add ESP32-S3 to the tested-boards list in
   `docs/source/...zephyr` and the Zephyr-side page
   (`doc/develop/manifest/external/executorch.rst` in zephyrproject-rtos/zephyr).

## Checklist before opening the PR

- [ ] Rebase the sample on `main` (module CMake is identical to v1.5.1 as of this writing).
- [ ] Switch headers to the ExecuTorch convention: Meta copyright line plus BSD-style
      licence text in `src/`, Apache-2.0 SPDX in CMake/Kconfig, matching `hello-executorch`.
- [ ] `lintrunner` passes (clang-format, cmake-format, ufmt for the Python).
- [ ] CI: add `xiao_esp32s3/esp32s3/procpu` to the Zephyr job in `.github/workflows` of
      ExecuTorch if one exists for Zephyr; otherwise document it as hardware-only and keep
      the FVP jobs unchanged.
- [ ] Attach serial log from real hardware showing `SELF-TEST PASS` and the timing line.
- [ ] Sign the Meta CLA.

## PR description draft

> Add a Zephyr sample running ExecuTorch on the Seeed XIAO ESP32S3 (Xtensa LX7).
>
> The existing Zephyr sample targets Cortex-M and Ethos-U. This adds a second sample that
> exercises the module on a non-Arm MCU using only the portable kernels with selective
> operator linking. It embeds a 63 KB MNIST CNN, verifies the output against PyTorch logits
> at boot, and optionally classifies frames from the board's OV2640 camera through Zephyr's
> video API. Tested on Zephyr v4.4.0 with the xtensa-espressif toolchain from Zephyr SDK
> 1.0.1.
