// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

/*
#include "decode_replay_bridge.h"
*/
import "C"

// Decode record/replay (lthn #perf). The recorded decode step's command stream
// is re-issued verbatim on replay, skipping the MLX tape-walk + per-primitive
// eval_gpu that costs ~12 ms/token (GPU idle). See mlx/backend/metal/device.cpp.

// lthnDecodeStepBegin arms recording of a full decode step (its command buffers).
func lthnDecodeStepBegin() { C.go_lthn_decode_step_begin() }

// lthnDecodeStepEnd ends step capture and returns the number of command buffers
// recorded for the step (12B decode = 2).
func lthnDecodeStepEnd() int { return int(C.go_lthn_decode_step_end()) }

// lthnDecodePinBegin defers buffer frees so the recorded step's buffers keep a
// stable address for replay. Pair with lthnDecodePinRelease.
func lthnDecodePinBegin() { C.go_lthn_decode_pin_begin() }

// lthnDecodePinRelease frees the buffers pinned since lthnDecodePinBegin.
func lthnDecodePinRelease() { C.go_lthn_decode_pin_release() }

// lthnDecodeReplayStep re-issues the captured step on the given stream and waits.
func lthnDecodeReplayStep(s *Stream) { C.go_lthn_decode_replay_step(s.ctx) }
