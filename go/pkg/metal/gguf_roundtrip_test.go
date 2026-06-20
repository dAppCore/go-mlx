// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"math"
	"testing"

	core "dappco.re/go"
)

// TestGGUF_SaveLoadRoundtrip_Good drives the GGUF tensor I/O on real Metal
// arrays: SaveGGUF writes a named-tensor map to a .gguf file, LoadAllGGUF reads
// it back as a map, and LoadGGUF streams the same file as a (name, array)
// iterator. The synthetic suite never touches a .gguf file (the hub cache is
// all safetensors), so all three were dark. No model load — this is a pure
// array round-trip, gated only on a usable Metal device. Run:
//
//	go test -tags metal_runtime -run TestGGUF_SaveLoadRoundtrip_Good ./pkg/metal/
func TestGGUF_SaveLoadRoundtrip_Good(t *testing.T) {
	requireMetalRuntime(t)

	dir := t.TempDir()
	path := core.PathJoin(dir, "weights.gguf")

	// Two named 2-D float32 tensors of different shapes.
	wA := FromValues([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	defer Free(wA)
	wB := FromValues([]float32{-1, -2, -3, -4}, 4, 1)
	defer Free(wB)

	weights := map[string]*Array{"a": wA, "b": wB}
	if err := SaveGGUF(path, weights); err != nil {
		t.Fatalf("SaveGGUF: %v", err)
	}
	if result := core.Stat(path); !result.OK {
		t.Fatalf("SaveGGUF produced no file at %s", path)
	}

	// LoadAllGGUF: full-map read-back, values must round-trip.
	t.Run("LoadAllGGUF", func(t *testing.T) {
		loaded, err := LoadAllGGUF(path)
		if err != nil {
			t.Fatalf("LoadAllGGUF: %v", err)
		}
		defer func() {
			for _, arr := range loaded {
				Free(arr)
			}
		}()
		if len(loaded) != 2 {
			t.Fatalf("LoadAllGGUF returned %d tensors, want 2", len(loaded))
		}
		got, ok := loaded["a"]
		if !ok {
			t.Fatal("tensor \"a\" missing from round-trip")
		}
		vals := got.Floats()
		want := []float32{1, 2, 3, 4, 5, 6}
		if len(vals) != len(want) {
			t.Fatalf("tensor \"a\" len = %d, want %d", len(vals), len(want))
		}
		for i := range want {
			if math.Abs(float64(vals[i]-want[i])) > 1e-5 {
				t.Fatalf("tensor \"a\"[%d] = %v, want %v", i, vals[i], want[i])
			}
		}
	})

	// LoadGGUF: streaming iterator over the same file. Cover both the
	// run-to-completion path and the early-break (yield == false) path.
	t.Run("LoadGGUF_Iterate", func(t *testing.T) {
		seen := 0
		for name, arr := range LoadGGUF(path) {
			if name == "" {
				t.Fatal("LoadGGUF yielded an empty tensor name")
			}
			if arr == nil {
				t.Fatalf("LoadGGUF yielded a nil array for %q", name)
			}
			Free(arr)
			seen++
		}
		if seen != 2 {
			t.Fatalf("LoadGGUF iterated %d tensors, want 2", seen)
		}
	})

	t.Run("LoadGGUF_EarlyBreak", func(t *testing.T) {
		count := 0
		for _, arr := range LoadGGUF(path) {
			count++
			Free(arr)
			break // exercise the yield==false / Free(arr) branch
		}
		if count != 1 {
			t.Fatalf("early-break iterated %d times, want 1", count)
		}
	})
}

// TestLoadAllGGUF_Missing_Bad: loading a non-existent .gguf returns an error,
// not an empty map silently — the LastError / no-tensors guard branch.
func TestLoadAllGGUF_Missing_Bad(t *testing.T) {
	requireMetalRuntime(t)

	dir := t.TempDir()
	missing := core.PathJoin(dir, "does-not-exist.gguf")
	if _, err := LoadAllGGUF(missing); err == nil {
		t.Fatal("LoadAllGGUF on a missing file returned nil error, want a load failure")
	}
}
