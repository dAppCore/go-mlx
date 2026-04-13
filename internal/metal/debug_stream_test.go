//go:build darwin && arm64

package metal

import (
	"testing"
)

func TestDebugStream(t *testing.T) {
	Init()
	
	// Clear any previous errors
	_ = lastError()
	
	s := DefaultCPUStream()
	t.Logf("CPU stream ctx nil: %v", s.ctx.ctx == nil)
	
	if err := lastError(); err != nil {
		t.Logf("error after CPU stream: %v", err)
	}
	
	gs := DefaultStream()
	t.Logf("GPU stream ctx nil: %v", gs.ctx.ctx == nil)
	
	if err := lastError(); err != nil {
		t.Logf("error after GPU stream: %v", err)
	}
}
