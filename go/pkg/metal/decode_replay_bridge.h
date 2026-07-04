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

// Overwrite an array's buffer contents in place (shared-mode Metal buffer). Used
// to update the per-token input (token id) + cache offset before replay, so each
// replayed step advances instead of repeating.
void go_lthn_array_write_bytes(mlx_array arr, const void* src, int n);

// Perf-probe gate (env MLX_DECODE_REPLAY_PROBE) + a guaranteed-visible stderr log
// so the in-generate replay-cost measurement shows regardless of Go log level.
int go_lthn_replay_probe_enabled(void);
void go_lthn_probe_log_ms(int phase, double ms, int extra); // phase 0=recorded,1=replayed

#ifdef __cplusplus
}
#endif
