// SPDX-Licence-Identifier: EUPL-1.2

#pragma once

#include "mlx/c/mlx.h"

#ifdef __cplusplus
extern "C" {
#endif

// Decode record/replay driver (lthn #perf) — wraps the MLX-fork primitives in
// mlx/backend/metal/device.cpp so the Go decode loop can record one decode step
// and replay it for subsequent tokens, skipping the ~12 ms/token host encode.
void go_lthn_decode_step_begin(void);
int go_lthn_decode_step_end(void);
void go_lthn_decode_pin_begin(void);
void go_lthn_decode_pin_release(void);
void go_lthn_decode_replay_step(const mlx_stream stream);

#ifdef __cplusplus
}
#endif
