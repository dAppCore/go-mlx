// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

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

// TestSlice_Slice1_Parity locks the W11-AC rank-1 scalar-pass slice
// primitive to bit-exact equality with the variadic Slice path so a
// regression in the rank-1 inline-C wrapper surfaces immediately rather
// than as a silent kernel divergence in unpackQ4's tail-trim boundary.
// Mirrors the W10-A Slice4 parity discipline at the rank-1 frontier.
func TestSlice_Slice1_Parity(t *testing.T) {
	cases := []struct {
		name  string
		data  []float32
		start int32
		end   int32
	}{
		{"prefix", []float32{1, 2, 3, 4, 5, 6}, 0, 3},
		{"suffix", []float32{1, 2, 3, 4, 5, 6}, 3, 6},
		{"middle", []float32{1, 2, 3, 4, 5, 6}, 2, 5},
		{"single", []float32{1, 2, 3, 4, 5, 6}, 4, 5},
		{"full", []float32{10, 20, 30}, 0, 3},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			a := FromValues(tc.data, len(tc.data))
			defer Free(a)

			scalar := Slice1(a, tc.start, tc.end)
			defer Free(scalar)
			Materialize(scalar)

			variadic := Slice(a, []int32{tc.start}, []int32{tc.end})
			defer Free(variadic)
			Materialize(variadic)

			sf, vf := scalar.Floats(), variadic.Floats()
			if len(sf) != len(vf) {
				t.Fatalf("length mismatch: scalar=%d variadic=%d", len(sf), len(vf))
			}
			ss := scalar.Shape()
			if len(ss) != 1 || ss[0] != tc.end-tc.start {
				t.Fatalf("scalar shape = %v, want [%d]", ss, tc.end-tc.start)
			}
			for i := range sf {
				if sf[i] != vf[i] {
					t.Fatalf("bit divergence at i=%d: scalar=%v variadic=%v", i, sf[i], vf[i])
				}
			}
		})
	}
}

// TestSlice_Slice2_Parity locks Slice2 to bit-exact equality with the
// variadic Slice and SliceAxis paths for rank-2 — covers the
// packQ4Cached low/high nibble extraction (`SliceAxis(paired, 1, 0, 1)`
// and `SliceAxis(paired, 1, 1, 2)`).
func TestSlice_Slice2_Parity(t *testing.T) {
	cases := []struct {
		name           string
		data           []float32
		h, w           int32
		s0, s1, e0, e1 int32
	}{
		{"full", []float32{1, 2, 3, 4, 5, 6}, 2, 3, 0, 0, 2, 3},
		{"col0", []float32{1, 2, 3, 4, 5, 6}, 3, 2, 0, 0, 3, 1}, // first column
		{"col1", []float32{1, 2, 3, 4, 5, 6}, 3, 2, 0, 1, 3, 2}, // second column
		{"row0", []float32{1, 2, 3, 4, 5, 6}, 2, 3, 0, 0, 1, 3}, // first row
		{"submat", []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3, 3, 1, 1, 3, 3},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			a := FromValues(tc.data, int(tc.h), int(tc.w))
			defer Free(a)

			scalar := Slice2(a, tc.s0, tc.s1, tc.e0, tc.e1)
			defer Free(scalar)
			Materialize(scalar)

			variadic := Slice(a, []int32{tc.s0, tc.s1}, []int32{tc.e0, tc.e1})
			defer Free(variadic)
			Materialize(variadic)

			sf, vf := scalar.Floats(), variadic.Floats()
			if len(sf) != len(vf) {
				t.Fatalf("length mismatch: scalar=%d variadic=%d", len(sf), len(vf))
			}
			ss := scalar.Shape()
			if len(ss) != 2 || ss[0] != tc.e0-tc.s0 || ss[1] != tc.e1-tc.s1 {
				t.Fatalf("scalar shape = %v, want [%d %d]", ss, tc.e0-tc.s0, tc.e1-tc.s1)
			}
			for i := range sf {
				if sf[i] != vf[i] {
					t.Fatalf("bit divergence at i=%d: scalar=%v variadic=%v", i, sf[i], vf[i])
				}
			}
		})
	}
}

// TestSlice_SliceUpdateInplace2_Parity locks SliceUpdateInplace2 to
// bit-exact equality with the variadic SliceUpdateInplace path on rank-2
// inputs so the pair-symmetry with Slice2 holds at the byte level.
func TestSlice_SliceUpdateInplace2_Parity(t *testing.T) {
	cases := []struct {
		name           string
		data, upd      []float32
		h, w           int32
		uh, uw         int32
		s0, s1, e0, e1 int32
	}{
		{"row0_replace", []float32{1, 2, 3, 4, 5, 6}, []float32{10, 20, 30}, 2, 3, 1, 3, 0, 0, 1, 3},
		{"col1_replace", []float32{1, 2, 3, 4, 5, 6}, []float32{99, 88}, 2, 3, 2, 1, 0, 1, 2, 2},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			a1 := FromValues(tc.data, int(tc.h), int(tc.w))
			defer Free(a1)
			upd1 := FromValues(tc.upd, int(tc.uh), int(tc.uw))
			defer Free(upd1)

			a2 := FromValues(tc.data, int(tc.h), int(tc.w))
			defer Free(a2)
			upd2 := FromValues(tc.upd, int(tc.uh), int(tc.uw))
			defer Free(upd2)

			scalar := SliceUpdateInplace2(a1, upd1, tc.s0, tc.s1, tc.e0, tc.e1)
			defer Free(scalar)
			Materialize(scalar)

			variadic := SliceUpdateInplace(a2, upd2, []int32{tc.s0, tc.s1}, []int32{tc.e0, tc.e1})
			defer Free(variadic)
			Materialize(variadic)

			sf, vf := scalar.Floats(), variadic.Floats()
			if len(sf) != len(vf) {
				t.Fatalf("length mismatch: scalar=%d variadic=%d", len(sf), len(vf))
			}
			for i := range sf {
				if sf[i] != vf[i] {
					t.Fatalf("bit divergence at i=%d: scalar=%v variadic=%v", i, sf[i], vf[i])
				}
			}
		})
	}
}
