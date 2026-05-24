// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
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

func nativePhaseMaterializeTraceEnabled() bool {
	return core.Env("GO_MLX_TRACE_FORWARD_EVAL") == "1"
}

func nativePhaseTraceArmed() bool {
	return nativePhaseTraceState.armed.Load()
}

func resetNativePhaseTraceEvents() {
	nativePhaseTraceState.Lock()
	nativePhaseTraceState.events = nativePhaseTraceState.events[:0]
	nativePhaseTraceState.armed.Store(true)
	nativePhaseTraceState.Unlock()
}

func appendNativePhaseTraceEvent(event NativePhaseTrace) {
	if !nativePhaseTraceArmed() {
		return
	}
	nativePhaseTraceState.Lock()
	if !nativePhaseTraceArmed() {
		nativePhaseTraceState.Unlock()
		return
	}
	nativePhaseTraceState.events = append(nativePhaseTraceState.events, event)
	nativePhaseTraceState.Unlock()
}

func takeNativePhaseTraceEvents() []NativePhaseTrace {
	if !nativePhaseTraceArmed() {
		return nil
	}
	nativePhaseTraceState.Lock()
	defer nativePhaseTraceState.Unlock()
	if !nativePhaseTraceArmed() {
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

func traceNativeMaterialize(name string, arrays ...*Array) {
	if !nativePhaseMaterializeTraceEnabled() || !nativePhaseTraceArmed() {
		return
	}
	start := time.Now()
	err := Eval(arrays...)
	event := NativePhaseTrace{Name: name, Duration: time.Since(start)}
	if err != nil {
		event.Error = err.Error()
		core.Error("mlx: native phase trace materialize", "phase", name, "error", err)
	} else {
		Detach(arrays...)
	}
	appendNativePhaseTraceEvent(event)
}

func traceNativeSkip(name, reason string) {
	if !nativePhaseTraceArmed() || name == "" || reason == "" {
		return
	}
	appendNativePhaseTraceEvent(NativePhaseTrace{Name: name, Error: reason})
}
