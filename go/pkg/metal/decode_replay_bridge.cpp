// SPDX-Licence-Identifier: EUPL-1.2

#include "decode_replay_bridge.h"

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
