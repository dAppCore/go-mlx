// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"testing"
)

func TestStreams_DefaultStreamsAreUsable_Good(t *testing.T) {
	Init()

	_ = LastError()

	cpu := DefaultCPUStream()
	if cpu == nil || cpu.ctx.ctx == nil {
		host := HostDeviceInfo()
		if err := LastError(); err != nil {
			t.Fatalf("DefaultCPUStream() returned nil stream: %v; host=%+v", err, host)
		}
		t.Fatalf("DefaultCPUStream() returned nil stream; host=%+v", host)
	}

	gpu := DefaultGPUStream()
	if gpu == nil || gpu.ctx.ctx == nil {
		host := HostDeviceInfo()
		if err := LastError(); err != nil {
			t.Fatalf("DefaultGPUStream() returned nil stream: %v; host=%+v", err, host)
		}
		t.Fatalf("DefaultGPUStream() returned nil stream; host=%+v", host)
	}
}
