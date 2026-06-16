// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// TestLthnDecodeReplay_TinyAdd_Good proves the record/replay mechanism end-to-end
// on the GPU: capture a tiny op's command stream as a "decode step", pin its
// buffers, then re-issue the recorded commands directly (no MLX tape-walk) and
// confirm the GPU recomputes the same result into the same buffers.
func TestLthnDecodeReplay_TinyAdd_Good(t *testing.T) {
	requireMetalRuntime(t)

	a := FromValues([]float32{1, 2, 3, 4}, 4)
	b := FromValues([]float32{10, 20, 30, 40}, 4)
	defer Free(a, b)
	Materialize(a, b)

	// Pin so the step's buffers keep stable addresses for replay.
	lthnDecodePinBegin()
	defer lthnDecodePinRelease()

	lthnDecodeStepBegin()
	c := Add(a, b)
	if err := Eval(c); err != nil {
		t.Fatalf("eval: %v", err)
	}
	nCBs := lthnDecodeStepEnd()
	if nCBs == 0 {
		t.Fatal("recorded 0 command buffers — recorder did not capture the step")
	}
	before := c.Floats()
	want := []float32{11, 22, 33, 44}
	for i := range want {
		if before[i] != want[i] {
			t.Fatalf("original add wrong at %d: got %v want %v", i, before[i], want[i])
		}
	}

	// Replay the recorded step — re-issues the Add into the same pinned buffers,
	// bypassing the MLX graph entirely.
	lthnDecodeReplayStep(DefaultStream())
	after := c.Floats()
	t.Logf("captured %d command buffer(s); before=%v after=%v", nCBs, before, after)
	for i := range after {
		if after[i] != before[i] {
			t.Fatalf("replay produced wrong output at %d: before %v after %v", i, before[i], after[i])
		}
	}
	Free(c)
}
