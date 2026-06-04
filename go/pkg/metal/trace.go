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

func NativePhaseMaterializeTraceEnabled() bool {
	return core.Env("GO_MLX_TRACE_FORWARD_EVAL") == "1"
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
	if !NativePhaseMaterializeTraceEnabled() || !NativePhaseTraceArmed() {
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
	AppendNativePhaseTraceEvent(event)
}

func TraceNativeSkip(name, reason string) {
	if !NativePhaseTraceArmed() || name == "" || reason == "" {
		return
	}
	AppendNativePhaseTraceEvent(NativePhaseTrace{Name: name, Error: reason})
}
