// SPDX-Licence-Identifier: EUPL-1.2

#include "decode_replay_bridge.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "mlx/backend/metal/device.h"
#include "mlx/c/private/mlx.h"

extern "C" void go_lthn_decode_step_begin(void) {
  mlx::core::metal::lthn_decode_step_begin();
}

extern "C" int go_lthn_decode_step_end(void) {
  return static_cast<int>(mlx::core::metal::lthn_decode_step_end());
}

extern "C" void go_lthn_decode_pin_begin(void) {
  mlx::core::metal::lthn_decode_pin_begin();
}

extern "C" void go_lthn_decode_pin_release(void) {
  mlx::core::metal::lthn_decode_pin_release();
}

extern "C" void go_lthn_decode_replay_step(const mlx_stream stream) {
  mlx::core::metal::lthn_decode_replay_step(mlx_stream_get_(stream));
}

extern "C" void go_lthn_array_write_bytes(mlx_array arr, const void* src, int n) {
  auto& a = mlx_array_get_(arr);
  void* dst = a.buffer().raw_ptr();
  if (dst != nullptr && src != nullptr && n > 0) {
    std::memcpy(dst, src, static_cast<size_t>(n));
  }
}

extern "C" int go_lthn_replay_probe_enabled(void) {
  const char* v = std::getenv("MLX_DECODE_REPLAY_PROBE");
  return (v && v[0] != '\0' && v[0] != '0') ? 1 : 0;
}

extern "C" void go_lthn_probe_log_ms(int phase, double ms, int extra) {
  std::fprintf(stderr, "[replay-probe] %s: %.3f ms (n=%d)\n",
               phase == 0 ? "recorded" : "replayed", ms, extra);
}
