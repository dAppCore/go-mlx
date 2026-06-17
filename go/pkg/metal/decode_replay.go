// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

/*
#include "decode_replay_bridge.h"
*/
import "C"

import "unsafe"

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

// lthnArrayWriteFloats overwrites an array's buffer contents in place (the
// per-token input/offset update before replay).
func lthnArrayWriteFloats(a *Array, vals []float32) {
	if len(vals) == 0 {
		return
	}
	C.go_lthn_array_write_bytes(a.ctx, unsafe.Pointer(&vals[0]), C.int(len(vals)*4))
}

// lthnReplayProbeEnabled reports the MLX_DECODE_REPLAY_PROBE env gate (read in C++
// since os is a banned import here).
func lthnReplayProbeEnabled() bool { return C.go_lthn_replay_probe_enabled() != 0 }

// lthnProbeLogMs writes a guaranteed-visible stderr line for the probe.
func lthnProbeLogMs(phase int, ms float64, extra int) {
	C.go_lthn_probe_log_ms(C.int(phase), C.double(ms), C.int(extra))
}

// lthnReplayOffsetArr, when non-nil, is the single shared int32 offset Array the
// gemma4 decode path binds for RoPE + attention instead of minting a fresh
// per-call FromValue. The record/replay driver installs it so ONE rewritable
// buffer advances the whole recorded step's cache position. Nil in normal decode
// (each call owns its offset — no aliasing across the pipeline). The driver owns
// its lifetime; the decode path never frees it while it is installed.
var lthnReplayOffsetArr *Array

// SetLthnReplayOffset installs the shared decode offset Array (nil clears it).
func SetLthnReplayOffset(a *Array) { lthnReplayOffsetArr = a }

// LthnReplayOffset returns the shared decode offset Array, or nil in normal decode.
func LthnReplayOffset() *Array { return lthnReplayOffsetArr }

// lthnWriteOffset overwrites the shared offset Array's int32 contents — advance
// the recorded step to a new cache position before re-issuing it.
func lthnWriteOffset(a *Array, off int32) {
	C.go_lthn_array_write_bytes(a.ctx, unsafe.Pointer(&off), C.int(4))
}
