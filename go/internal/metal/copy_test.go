// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

func TestCopy_Copy_BreaksGraph_Good(t *testing.T) {
	// Create a chain: a -> b -> c
	a := FromValue(float32(1.0))
	b := Add(a, FromValue(float32(2.0)))
	Eval(b)

	// Copy should break the graph
	c := Copy(b)
	Eval(c)

	// Free b — if Copy truly detaches, c should still be valid
	Free(b)

	val := c.Float()
	if val != 3.0 {
		t.Fatalf("expected 3.0, got %f", val)
	}
	Free(a, c)
}
