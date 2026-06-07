// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"testing"

	core "dappco.re/go"
)

// TestDeriveAffineBits_Good derives the bit-width of an MLX affine-quantised
// linear from the only thing that can't lie: the packed-weight and scales
// last-dims plus the group size. The identity is
//
//	packed weight last-dim = logical_in * bits / 32
//	scales      last-dim   = logical_in / group_size
//	⇒ bits = 32 * weightLast / (scalesLast * group_size)
//
// The cases are real shapes pulled from gemma-4 packs (q4 g64 is E2B's q_proj),
// so a green test means the engine reads bits from the bytes, not the filename.
func TestDeriveAffineBits_Good(t *testing.T) {
	cases := []struct {
		name                       string
		weightLast, scalesLast, gs int
		want                       int
	}{
		{"q4 g64 (E2B q_proj)", 192, 24, 64, 4},
		{"q4 g64 (256-wide)", 256, 32, 64, 4},
		{"q8 g64", 512, 32, 64, 8},
		{"q6 g64", 384, 32, 64, 6},
		{"q4 g32", 96, 24, 32, 4},
	}
	for _, c := range cases {
		got, ok := deriveAffineBits(c.weightLast, c.scalesLast, c.gs)
		if !ok || got != c.want {
			t.Errorf("%s: deriveAffineBits(%d,%d,%d) = (%d,%v), want (%d,true)",
				c.name, c.weightLast, c.scalesLast, c.gs, got, ok, c.want)
		}
	}
}

// TestDeriveAffineBits_Bad rejects shapes that don't describe a clean affine
// pack — zero/negative dims, and ratios that don't yield an integer 1..8 bit
// width. Garbage in must fail loud (ok=false), never return a plausible-looking
// wrong bit-width.
func TestDeriveAffineBits_Bad(t *testing.T) {
	cases := []struct {
		name                       string
		weightLast, scalesLast, gs int
	}{
		{"zero group", 192, 24, 0},
		{"zero scales", 192, 0, 64},
		{"zero weight", 0, 24, 64},
		{"negative", -192, 24, 64},
		{"non-integer bits", 100, 24, 64},
		{"bits over 8", 4096, 24, 64},
	}
	for _, c := range cases {
		if got, ok := deriveAffineBits(c.weightLast, c.scalesLast, c.gs); ok {
			t.Errorf("%s: deriveAffineBits(%d,%d,%d) = (%d,true), want ok=false",
				c.name, c.weightLast, c.scalesLast, c.gs, got)
		}
	}
}

// TestResolveQuant_Good resolves real gemma-4 packs from the HF cache: q4 and q6
// E2B. It proves the read comes from the model's own bytes — bits derived from
// the tensor geometry, cross-checked against the declared group — across two
// bit-widths. Skips when the packs aren't cached so CI stays green without them.
func TestResolveQuant_Good(t *testing.T) {
	cases := []struct {
		repo string
		bits int
	}{
		{"models--mlx-community--gemma-4-e2b-it-4bit", 4},
		{"models--mlx-community--gemma-4-e2b-it-6bit", 6},
	}
	for _, c := range cases {
		dir := hfSnapshotOrSkip(t, c.repo)
		spec, err := ResolveQuant(dir)
		if err != nil {
			t.Fatalf("%s: ResolveQuant: %v", c.repo, err)
		}
		if spec.Format != QuantAffine || spec.Bits != c.bits || spec.GroupSize != 64 {
			t.Fatalf("%s: got %+v, want {affine %d 64}", c.repo, spec, c.bits)
		}
	}
}

// TestResolveQuant_FullPrecision resolves a bf16 pack to QuantNone — a model
// with no quantization block carries no bits.
func TestResolveQuant_FullPrecision(t *testing.T) {
	dir := hfSnapshotOrSkip(t, "models--mlx-community--gemma-4-E2B-it-bf16")
	spec, err := ResolveQuant(dir)
	if err != nil {
		t.Fatalf("ResolveQuant: %v", err)
	}
	if spec.Format != QuantNone || spec.Bits != 0 {
		t.Fatalf("got %+v, want {none 0}", spec)
	}
}

// hfSnapshotOrSkip resolves a HuggingFace cache snapshot directory
// (~/.cache/huggingface/hub/<repo>/snapshots/<hash>), skipping the test when the
// pack isn't present so a machine without the cache stays green.
func hfSnapshotOrSkip(t *testing.T, repo string) string {
	t.Helper()
	home := core.UserHomeDir()
	if !home.OK {
		t.Skip("no home dir")
	}
	snapshots := core.PathJoin(home.Value.(string), ".cache", "huggingface", "hub", repo, "snapshots")
	listed := core.ReadDir(core.DirFS(snapshots), ".")
	if !listed.OK {
		t.Skipf("not cached: %s", repo)
	}
	for _, entry := range listed.Value.([]core.FsDirEntry) {
		return core.PathJoin(snapshots, entry.Name())
	}
	t.Skipf("no snapshot under %s", snapshots)
	return ""
}
