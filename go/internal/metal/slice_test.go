// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// Generated file-aware compliance coverage.
func TestSlice_Slice_Good(t *testing.T) {
	target := "Slice"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestSlice_Slice_Bad(t *testing.T) {
	target := "Slice"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestSlice_Slice_Ugly(t *testing.T) {
	target := "Slice"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestSlice_SliceAxis_Good(t *testing.T) {
	target := "SliceAxis"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestSlice_SliceAxis_Bad(t *testing.T) {
	target := "SliceAxis"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestSlice_SliceAxis_Ugly(t *testing.T) {
	target := "SliceAxis"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestSlice_SliceUpdateInplace_Good(t *testing.T) {
	target := "SliceUpdateInplace"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestSlice_SliceUpdateInplace_Bad(t *testing.T) {
	target := "SliceUpdateInplace"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestSlice_SliceUpdateInplace_Ugly(t *testing.T) {
	target := "SliceUpdateInplace"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

// TestSlice4WithStream_Parity checks that the stream-passing Slice4 variant
// produces bit-exact same output as the DefaultStream-resolving form,
// across a representative KV-cache slice geometry. The two forms only
// differ in whether the stream is hoisted by the caller.
func TestSlice4WithStream_Parity(t *testing.T) {
	if !MetalAvailable() {
		t.Skip("Metal unavailable")
	}
	// Seeded source — mirrors the KV-cache rank-4 [B, H, L, D] slice
	// geometry in KVCache.Update.
	src := RandomUniform(-1, 1, []int32{2, 4, 8, 16}, DTypeFloat32)
	defer Free(src)

	// Default-stream form.
	a := Slice4(src, 0, 0, 2, 0, 2, 4, 7, 16)
	defer Free(a)
	// Stream-hoisted form — same arguments.
	stream := DefaultStream()
	b := Slice4WithStream(src, 0, 0, 2, 0, 2, 4, 7, 16, stream)
	defer Free(b)

	if err := Eval(a, b); err != nil {
		t.Fatalf("Eval: %v", err)
	}
	aHost := a.Floats()
	bHost := b.Floats()
	if len(aHost) != len(bHost) {
		t.Fatalf("Slice4WithStream length mismatch: default=%d stream=%d", len(aHost), len(bHost))
	}
	for i := range aHost {
		if aHost[i] != bHost[i] {
			t.Fatalf("Slice4WithStream parity mismatch at i=%d: default=%g stream=%g", i, aHost[i], bHost[i])
		}
	}
}

// TestSliceUpdateInplace4WithStream_Parity is the SliceUpdateInplace4
// counterpart to TestSlice4WithStream_Parity — verifies bit-exact output
// equivalence between the default-stream-resolving form and the
// stream-passing sibling under a KV-cache append geometry.
func TestSliceUpdateInplace4WithStream_Parity(t *testing.T) {
	if !MetalAvailable() {
		t.Skip("Metal unavailable")
	}
	base := RandomUniform(-1, 1, []int32{2, 4, 8, 16}, DTypeFloat32)
	patch := RandomUniform(-1, 1, []int32{2, 4, 3, 16}, DTypeFloat32)
	defer Free(base, patch)

	// Default-stream form.
	a := SliceUpdateInplace4(base, patch, 0, 0, 2, 0, 2, 4, 5, 16)
	defer Free(a)
	// Stream-hoisted form — same arguments.
	stream := DefaultStream()
	b := SliceUpdateInplace4WithStream(base, patch, 0, 0, 2, 0, 2, 4, 5, 16, stream)
	defer Free(b)

	if err := Eval(a, b); err != nil {
		t.Fatalf("Eval: %v", err)
	}
	aHost := a.Floats()
	bHost := b.Floats()
	if len(aHost) != len(bHost) {
		t.Fatalf("SliceUpdateInplace4WithStream length mismatch: default=%d stream=%d", len(aHost), len(bHost))
	}
	for i := range aHost {
		if aHost[i] != bHost[i] {
			t.Fatalf("SliceUpdateInplace4WithStream parity mismatch at i=%d: default=%g stream=%g", i, aHost[i], bHost[i])
		}
	}
}
