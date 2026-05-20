// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"sync"
	"time"

	"dappco.re/go"
)

var nativePhaseTraceState struct {
	sync.Mutex
	armed  bool
	events []NativePhaseTrace
}

func nativePhaseTraceEnabled() bool {
	return core.Env("GO_MLX_TRACE_FORWARD_EVAL") == "1"
}

func resetNativePhaseTraceEvents() {
	if !nativePhaseTraceEnabled() {
		return
	}
	nativePhaseTraceState.Lock()
	nativePhaseTraceState.events = nativePhaseTraceState.events[:0]
	nativePhaseTraceState.armed = true
	nativePhaseTraceState.Unlock()
}

func appendNativePhaseTraceEvent(event NativePhaseTrace) {
	if !nativePhaseTraceEnabled() {
		return
	}
	nativePhaseTraceState.Lock()
	if !nativePhaseTraceState.armed {
		nativePhaseTraceState.Unlock()
		return
	}
	nativePhaseTraceState.events = append(nativePhaseTraceState.events, event)
	nativePhaseTraceState.Unlock()
}

func takeNativePhaseTraceEvents() []NativePhaseTrace {
	if !nativePhaseTraceEnabled() {
		return nil
	}
	nativePhaseTraceState.Lock()
	defer nativePhaseTraceState.Unlock()
	if len(nativePhaseTraceState.events) == 0 {
		return nil
	}
	events := append([]NativePhaseTrace(nil), nativePhaseTraceState.events...)
	nativePhaseTraceState.events = nativePhaseTraceState.events[:0]
	nativePhaseTraceState.armed = false
	return events
}

func traceNativeMaterialize(name string, arrays ...*Array) {
	if !nativePhaseTraceEnabled() {
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
	if !nativePhaseTraceEnabled() || name == "" || reason == "" {
		return
	}
	appendNativePhaseTraceEvent(NativePhaseTrace{Name: name, Error: reason})
}
