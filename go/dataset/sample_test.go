// SPDX-Licence-Identifier: EUPL-1.2

package dataset

import "testing"

// --- Func adapter ---

func TestSample_FuncNext_Good(t *testing.T) {
	want := Sample{Prompt: "p", Response: "r", Format: "prompt_response"}
	calls := 0
	fn := Func(func() (Sample, bool, error) {
		calls++
		return want, true, nil
	})

	got, ok, err := fn.Next()
	if err != nil {
		t.Fatalf("Func.Next() error = %v", err)
	}
	if !ok {
		t.Fatal("Func.Next() ok = false, want true")
	}
	if got.Prompt != want.Prompt || got.Response != want.Response || got.Format != want.Format {
		t.Fatalf("Func.Next() = %+v, want %+v", got, want)
	}
	if calls != 1 {
		t.Fatalf("wrapped fn called %d times, want 1", calls)
	}
}

func TestSample_FuncNext_Bad(t *testing.T) {
	var fn Func // nil function value
	if _, _, err := fn.Next(); err == nil {
		t.Fatal("Func.Next() on nil func: expected error, got nil")
	}
}

// Ugly: the wrapped function reports exhaustion. Next must surface
// (zero, false, nil) without inventing an error.
func TestSample_FuncNext_Ugly(t *testing.T) {
	fn := Func(func() (Sample, bool, error) {
		return Sample{}, false, nil
	})
	got, ok, err := fn.Next()
	if err != nil {
		t.Fatalf("Func.Next() error = %v", err)
	}
	if ok {
		t.Fatalf("Func.Next() ok = true on exhausted func, want false (got %+v)", got)
	}
}

// --- NewSliceDataset + SliceDataset.Next/Reset ---

// Good: NewSliceDataset clones the slice header, so reordering or
// reassigning entries in the source slice after construction cannot reach
// into the dataset's iteration.
func TestSample_NewSliceDataset_Good(t *testing.T) {
	source := []Sample{
		{Text: "a"},
		{Prompt: "p", Response: "r"},
	}
	ds := NewSliceDataset(source)

	// Reassign the source entries (slice-header level mutation) — the
	// dataset holds its own backing array and is unaffected.
	source[0] = Sample{Text: "mutated"}
	source[1] = Sample{Response: "mutated"}

	first, ok, err := ds.Next()
	if err != nil {
		t.Fatalf("Next() error = %v", err)
	}
	if !ok || first.Text != "a" {
		t.Fatalf("Next()[0] = %+v ok=%v, want original Text 'a'", first, ok)
	}
	second, ok, err := ds.Next()
	if err != nil {
		t.Fatalf("Next() error = %v", err)
	}
	if !ok || second.Response != "r" {
		t.Fatalf("Next()[1] = %+v ok=%v, want original Response 'r'", second, ok)
	}
	// Exhaustion: third Next returns (zero, false, nil).
	if _, ok, err := ds.Next(); ok || err != nil {
		t.Fatalf("Next() after end = ok %v err %v, want false,nil", ok, err)
	}
}

// Reset rewinds so a second full pass yields the same records — the
// multi-epoch training contract.
func TestSample_SliceDatasetReset_Good(t *testing.T) {
	ds := NewSliceDataset([]Sample{{Text: "row0"}, {Text: "row1"}})
	drain := func() []Sample {
		var out []Sample
		for {
			s, ok, err := ds.Next()
			if err != nil {
				t.Fatalf("Next() error = %v", err)
			}
			if !ok {
				return out
			}
			out = append(out, s)
		}
	}
	first := drain()
	if len(first) != 2 {
		t.Fatalf("first pass len = %d, want 2", len(first))
	}
	if err := ds.Reset(); err != nil {
		t.Fatalf("Reset() error = %v", err)
	}
	second := drain()
	if len(second) != 2 || second[0].Text != "row0" || second[1].Text != "row1" {
		t.Fatalf("second pass after Reset = %+v, want identical replay", second)
	}
}

// Bad: nil-receiver guards on both Next and Reset return the sentinel.
func TestSample_SliceDatasetNilReceiver_Bad(t *testing.T) {
	var ds *SliceDataset
	if _, ok, err := ds.Next(); err == nil || ok {
		t.Fatalf("nil SliceDataset.Next() = ok %v err %v, want false + error", ok, err)
	}
	if err := ds.Reset(); err == nil {
		t.Fatal("nil SliceDataset.Reset(): expected error, got nil")
	}
}

// --- CloneSamples (empty/nil path — the 83.3% gap) ---

func TestSample_CloneSamples_Good(t *testing.T) {
	source := []Sample{{Text: "a", Meta: map[string]string{"k": "v"}}}
	out := CloneSamples(source)
	if len(out) != 1 {
		t.Fatalf("CloneSamples len = %d, want 1", len(out))
	}
	// Mutating the clone's Meta must not touch the source map.
	out[0].Meta["k"] = "changed"
	if source[0].Meta["k"] != "v" {
		t.Fatalf("CloneSamples aliased Meta: source k = %q, want 'v'", source[0].Meta["k"])
	}
}

// Ugly: empty and nil inputs both return nil (not a zero-length non-nil
// slice) — exercises the len==0 fast path.
func TestSample_CloneSamples_Ugly(t *testing.T) {
	if got := CloneSamples(nil); got != nil {
		t.Fatalf("CloneSamples(nil) = %v, want nil", got)
	}
	if got := CloneSamples([]Sample{}); got != nil {
		t.Fatalf("CloneSamples(empty) = %v, want nil", got)
	}
}

// --- CloneSample Meta deep-copy + nil-Meta path ---

func TestSample_CloneSample_Good(t *testing.T) {
	src := Sample{Prompt: "p", Response: "r", Meta: map[string]string{"split": "train"}}
	clone := CloneSample(src)
	clone.Meta["split"] = "test"
	if src.Meta["split"] != "train" {
		t.Fatalf("CloneSample aliased Meta: src split = %q, want 'train'", src.Meta["split"])
	}
	if clone.Prompt != "p" || clone.Response != "r" {
		t.Fatalf("CloneSample dropped scalar fields: %+v", clone)
	}
}

// Ugly: a Sample with no Meta clones to a nil Meta (cloneStringMap nil-fast
// path) rather than an empty allocated map.
func TestSample_CloneSample_Ugly(t *testing.T) {
	clone := CloneSample(Sample{Text: "no meta"})
	if clone.Meta != nil {
		t.Fatalf("CloneSample(no meta).Meta = %v, want nil", clone.Meta)
	}
	if clone.Text != "no meta" {
		t.Fatalf("CloneSample dropped Text: %+v", clone)
	}
}

// An empty/nil source yields a dataset that is immediately exhausted —
// NewSliceDataset(nil).Next() must not panic on the cloned-nil backing.
func TestSample_NewSliceDataset_EmptyInput(t *testing.T) {
	ds := NewSliceDataset(nil)
	if _, ok, err := ds.Next(); ok || err != nil {
		t.Fatalf("empty NewSliceDataset.Next() = ok %v err %v, want false,nil", ok, err)
	}
	if err := ds.Reset(); err != nil {
		t.Fatalf("empty NewSliceDataset.Reset() error = %v", err)
	}
}
