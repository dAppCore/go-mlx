// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// affine dispatches through the registered loader; the resulting Linear is the
// same as the built-in assembly (behaviour-identical for the only shipped
// format).
func TestQuantLoaderDispatch_Affine_Good(t *testing.T) {
	w := Zeros([]int32{4, 8}, DTypeFloat32)
	defer Free(w)
	lin := NewQuantizedLinearWithMode(w, nil, nil, nil, 64, 4, "affine")
	if lin == nil || lin.Weight != w || lin.GroupSize != 64 || lin.Bits != 4 || lin.QuantizationMode != "affine" {
		t.Fatalf("affine dispatch = %+v, want weight=w group=64 bits=4 mode=affine", lin)
	}
	// Empty mode normalises to affine and dispatches the same.
	if got := NewQuantizedLinearWithMode(w, nil, nil, nil, 64, 4, ""); got.QuantizationMode != "affine" {
		t.Errorf("empty mode = %q, want affine", got.QuantizationMode)
	}
}

// A mode with no registered loader falls back to the built-in assembly,
// preserving the mode — so unbinding affine into the registry changed nothing
// for any other mode.
func TestQuantLoaderDispatch_UnregisteredFallsBack_Ugly(t *testing.T) {
	w := Zeros([]int32{4, 8}, DTypeFloat32)
	defer Free(w)
	lin := NewQuantizedLinearWithMode(w, nil, nil, nil, 32, 8, "no-such-quant-format")
	if lin == nil || lin.GroupSize != 32 || lin.Bits != 8 || lin.QuantizationMode != "no-such-quant-format" {
		t.Fatalf("unregistered mode = %+v, want built-in assembly with mode preserved", lin)
	}
	if _, ok := quantLoaders["no-such-quant-format"]; ok {
		t.Error("unexpected loader registered for the fake mode")
	}
}
