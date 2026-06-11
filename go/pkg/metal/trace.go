// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"crypto/sha256"
	"math"
	"sync"
	"sync/atomic"
	"time"

	"dappco.re/go"
)

var nativePhaseTraceState struct {
	sync.Mutex
	armed  atomic.Bool
	events []NativePhaseTrace
}

// nativePhaseMaterializeTrace forces phase materialisation during forward so the
// native-phase tracer can record eval points. It STEERS execution (extra
// materialisation), so it is an in-code diagnostic only — off by default, NEVER
// ambient env (an env-readable execution toggle is external control). Set it in
// code / a test to trace.
var nativePhaseMaterializeTrace = false

func NativePhaseMaterializeTraceEnabled() bool {
	return nativePhaseMaterializeTrace
}

func NativePhaseTraceArmed() bool {
	return nativePhaseTraceState.armed.Load()
}

func resetNativePhaseTraceEvents() {
	nativePhaseTraceState.Lock()
	nativePhaseTraceState.events = nativePhaseTraceState.events[:0]
	nativePhaseTraceState.armed.Store(true)
	nativePhaseTraceState.Unlock()
}

func AppendNativePhaseTraceEvent(event NativePhaseTrace) {
	if !NativePhaseTraceArmed() {
		return
	}
	nativePhaseTraceState.Lock()
	if !NativePhaseTraceArmed() {
		nativePhaseTraceState.Unlock()
		return
	}
	nativePhaseTraceState.events = append(nativePhaseTraceState.events, event)
	nativePhaseTraceState.Unlock()
}

func takeNativePhaseTraceEvents() []NativePhaseTrace {
	if !NativePhaseTraceArmed() {
		return nil
	}
	nativePhaseTraceState.Lock()
	defer nativePhaseTraceState.Unlock()
	if !NativePhaseTraceArmed() {
		return nil
	}
	if len(nativePhaseTraceState.events) == 0 {
		nativePhaseTraceState.armed.Store(false)
		return nil
	}
	events := append([]NativePhaseTrace(nil), nativePhaseTraceState.events...)
	nativePhaseTraceState.events = nativePhaseTraceState.events[:0]
	nativePhaseTraceState.armed.Store(false)
	return events
}

func TraceNativeMaterialize(name string, arrays ...*Array) {
	hashing := nativePhaseValueHash.Load()
	timing := NativePhaseMaterializeTraceEnabled() && NativePhaseTraceArmed()
	if !hashing && !timing {
		return
	}
	start := time.Now()
	err := Eval(arrays...)
	if hashing {
		appendNativePhaseValueHash(name, err, arrays...)
	}
	if !timing {
		if err == nil {
			Detach(arrays...)
		}
		return
	}
	event := NativePhaseTrace{Name: name, Duration: time.Since(start)}
	if err != nil {
		event.Error = err.Error()
		core.Error("mlx: native phase trace materialize", "phase", name, "error", err)
	} else {
		Detach(arrays...)
	}
	AppendNativePhaseTraceEvent(event)
}

// Phase value hashing — the determinism bisect instrument. When enabled, every
// TraceNativeMaterialize point also records a sha256 of the phase tensors'
// float32-converted contents, in execution order, into its own log. It STEERS
// execution exactly like the timing trace (per-phase materialisation), so it
// is an in-code diagnostic only — never ambient env.
var nativePhaseValueHash atomic.Bool

// NativePhaseValueHash is one hashed phase tensor observation.
type NativePhaseValueHash struct {
	Name string
	Hash string
}

var nativePhaseValueHashState struct {
	sync.Mutex
	log []NativePhaseValueHash
}

// SetNativePhaseValueHashCapture toggles phase value hashing (diagnostic).
func SetNativePhaseValueHashCapture(enabled bool) {
	nativePhaseValueHash.Store(enabled)
}

// NativePhaseValueHashEnabled reports whether phase value hashing is on.
func NativePhaseValueHashEnabled() bool {
	return nativePhaseValueHash.Load()
}

// TakeNativePhaseValueHashes returns and clears the hash log.
func TakeNativePhaseValueHashes() []NativePhaseValueHash {
	nativePhaseValueHashState.Lock()
	defer nativePhaseValueHashState.Unlock()
	log := append([]NativePhaseValueHash(nil), nativePhaseValueHashState.log...)
	nativePhaseValueHashState.log = nativePhaseValueHashState.log[:0]
	return log
}

func appendNativePhaseValueHash(name string, evalErr error, arrays ...*Array) {
	entry := NativePhaseValueHash{Name: name}
	if evalErr != nil {
		entry.Hash = "eval-error: " + evalErr.Error()
	} else {
		digest := sha256.New()
		for _, arr := range arrays {
			if arr == nil || !arr.Valid() {
				digest.Write([]byte("|nil"))
				continue
			}
			f32 := AsType(arr, DTypeFloat32)
			if err := Eval(f32); err != nil {
				digest.Write([]byte("|eval-error:" + err.Error()))
				Free(f32)
				continue
			}
			floats := f32.Floats()
			var quad [4]byte
			for _, f := range floats {
				bits := math.Float32bits(f)
				quad[0], quad[1], quad[2], quad[3] = byte(bits), byte(bits>>8), byte(bits>>16), byte(bits>>24)
				digest.Write(quad[:])
			}
			Free(f32)
		}
		entry.Hash = core.Sprintf("%x", digest.Sum(nil)[:8])
	}
	nativePhaseValueHashState.Lock()
	nativePhaseValueHashState.log = append(nativePhaseValueHashState.log, entry)
	nativePhaseValueHashState.Unlock()
}

func TraceNativeSkip(name, reason string) {
	if !NativePhaseTraceArmed() || name == "" || reason == "" {
		return
	}
	AppendNativePhaseTraceEvent(NativePhaseTrace{Name: name, Error: reason})
}
