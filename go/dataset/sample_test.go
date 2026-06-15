// SPDX-Licence-Identifier: EUPL-1.2

package dataset

import "testing"

// --- Func.Next ---

func TestSample_Func_Next_Good(t *testing.T) {
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

func TestSample_Func_Next_Bad(t *testing.T) {
	var fn Func // nil function value
	if _, _, err := fn.Next(); err == nil {
		t.Fatal("Func.Next() on nil func: expected error, got nil")
	}
}

// Ugly: the wrapped function reports exhaustion. Next must surface
// (zero, false, nil) without inventing an error. A second Next on the same
// exhausted closure must stay (zero, false, nil) — Func is a pure pass-through
// with no state of its own.
func TestSample_Func_Next_Ugly(t *testing.T) {
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
	// Repeated calls on an exhausted generator keep returning exhaustion.
	if _, ok, err := fn.Next(); ok || err != nil {
		t.Fatalf("Func.Next() second call = ok %v err %v, want false,nil", ok, err)
	}
}

// --- NewSliceDataset ---

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
		t.Fatalf("NewSliceDataset clone failed: first = %+v ok=%v, want original Text 'a'", first, ok)
	}
}

// Bad: NewSliceDataset is total — there is no input that makes it error or
// return a nil pointer. The degenerate input is a nil slice; the constructed
// dataset must be a usable, immediately-exhausted dataset (never nil, never a
// panic on first Next). This pins the "no bad input produces a broken dataset"
// contract.
func TestSample_NewSliceDataset_Bad(t *testing.T) {
	ds := NewSliceDataset(nil)
	if ds == nil {
		t.Fatal("NewSliceDataset(nil) = nil, want a usable empty dataset")
	}
	if _, ok, err := ds.Next(); ok || err != nil {
		t.Fatalf("NewSliceDataset(nil).Next() = ok %v err %v, want false,nil", ok, err)
	}
}

// Ugly: an empty (zero-length, non-nil) source yields a dataset that is
// immediately exhausted — NewSliceDataset([]Sample{}).Next() must not panic on
// the cloned backing, and Reset on it must stay a no-op error-free.
func TestSample_NewSliceDataset_Ugly(t *testing.T) {
	ds := NewSliceDataset([]Sample{})
	if _, ok, err := ds.Next(); ok || err != nil {
		t.Fatalf("empty NewSliceDataset.Next() = ok %v err %v, want false,nil", ok, err)
	}
	if err := ds.Reset(); err != nil {
		t.Fatalf("empty NewSliceDataset.Reset() error = %v", err)
	}
}

// --- SliceDataset.Next ---

// Good: sequential Next calls yield each record in order, then exhaust with
// (zero, false, nil). The clone-isolation guarantee is asserted alongside —
// reassigning source entries after construction cannot reach the iteration.
func TestSample_SliceDataset_Next_Good(t *testing.T) {
	source := []Sample{
		{Text: "a"},
		{Prompt: "p", Response: "r"},
	}
	ds := NewSliceDataset(source)
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

// Bad: Next on a nil *SliceDataset returns the sentinel error and ok=false
// rather than panicking.
func TestSample_SliceDataset_Next_Bad(t *testing.T) {
	var ds *SliceDataset
	if _, ok, err := ds.Next(); err == nil || ok {
		t.Fatalf("nil SliceDataset.Next() = ok %v err %v, want false + error", ok, err)
	}
}

// Ugly: Next past the end is idempotent — once exhausted, every further Next
// keeps returning (zero, false, nil) without advancing or erroring, and the
// zero value carries no leaked fields from the last real record.
func TestSample_SliceDataset_Next_Ugly(t *testing.T) {
	ds := NewSliceDataset([]Sample{{Text: "only"}})
	if _, ok, _ := ds.Next(); !ok {
		t.Fatal("first Next() ok = false, want the single record")
	}
	for i := 0; i < 3; i++ {
		got, ok, err := ds.Next()
		if ok || err != nil {
			t.Fatalf("Next() past end #%d = ok %v err %v, want false,nil", i, ok, err)
		}
		if got.Text != "" || got.Prompt != "" || got.Response != "" || got.Format != "" || got.Meta != nil {
			t.Fatalf("Next() past end #%d returned non-zero sample %+v", i, got)
		}
	}
}

// --- SliceDataset.Reset ---

// Good: Reset rewinds so a second full pass yields the same records — the
// multi-epoch training contract.
func TestSample_SliceDataset_Reset_Good(t *testing.T) {
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

// Bad: Reset on a nil *SliceDataset returns the sentinel error rather than
// panicking.
func TestSample_SliceDataset_Reset_Bad(t *testing.T) {
	var ds *SliceDataset
	if err := ds.Reset(); err == nil {
		t.Fatal("nil SliceDataset.Reset(): expected error, got nil")
	}
}

// Ugly: Reset is safe at the boundaries — calling it before any Next (no-op),
// and calling it twice in a row, both leave the cursor at the start with no
// error and a faithful first record.
func TestSample_SliceDataset_Reset_Ugly(t *testing.T) {
	ds := NewSliceDataset([]Sample{{Text: "head"}, {Text: "tail"}})
	// Reset before consuming anything.
	if err := ds.Reset(); err != nil {
		t.Fatalf("Reset() before first Next error = %v", err)
	}
	// Double Reset.
	if err := ds.Reset(); err != nil {
		t.Fatalf("second consecutive Reset() error = %v", err)
	}
	got, ok, err := ds.Next()
	if err != nil || !ok || got.Text != "head" {
		t.Fatalf("after boundary Resets, Next() = %+v ok=%v err=%v, want head", got, ok, err)
	}
}

// --- CloneSample ---

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

// Bad: CloneSample is total (no error channel), so the adversarial input is the
// fully zero Sample. The clone must equal the zero value exactly — no map
// materialised, no scalar invented — proving the no-Meta fast path doesn't
// fabricate state for an empty record.
func TestSample_CloneSample_Bad(t *testing.T) {
	clone := CloneSample(Sample{})
	if clone.Text != "" || clone.Prompt != "" || clone.Response != "" || clone.Format != "" {
		t.Fatalf("CloneSample(zero) = %+v, want the zero Sample", clone)
	}
	if clone.Meta != nil {
		t.Fatalf("CloneSample(zero).Meta = %v, want nil (no map materialised)", clone.Meta)
	}
}

// Ugly: a Sample with no Meta clones to a nil Meta (cloneStringMap nil-fast
// path) rather than an empty allocated map, while non-Meta fields survive. An
// explicitly empty (non-nil) Meta also collapses to nil — the len==0 guard
// does not preserve emptiness.
func TestSample_CloneSample_Ugly(t *testing.T) {
	clone := CloneSample(Sample{Text: "no meta"})
	if clone.Meta != nil {
		t.Fatalf("CloneSample(no meta).Meta = %v, want nil", clone.Meta)
	}
	if clone.Text != "no meta" {
		t.Fatalf("CloneSample dropped Text: %+v", clone)
	}
	// An empty-but-non-nil Meta is treated identically to nil.
	emptied := CloneSample(Sample{Text: "empty meta", Meta: map[string]string{}})
	if emptied.Meta != nil {
		t.Fatalf("CloneSample(empty meta).Meta = %v, want nil", emptied.Meta)
	}
}

// --- CloneSamples ---

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

// Bad: CloneSamples has no error channel, so the degenerate input is the empty
// slice — it must return nil (not a zero-length non-nil slice). A slice of
// zero-value Samples must clone element-for-element without inventing Meta maps.
func TestSample_CloneSamples_Bad(t *testing.T) {
	if got := CloneSamples([]Sample{}); got != nil {
		t.Fatalf("CloneSamples(empty) = %v, want nil", got)
	}
	// Non-empty slice of empties: length preserved, no maps fabricated.
	out := CloneSamples([]Sample{{}, {}})
	if len(out) != 2 {
		t.Fatalf("CloneSamples([2 empties]) len = %d, want 2", len(out))
	}
	for i, s := range out {
		if s.Text != "" || s.Prompt != "" || s.Response != "" || s.Format != "" || s.Meta != nil {
			t.Fatalf("CloneSamples element %d = %+v, want zero Sample", i, s)
		}
	}
}

// Ugly: nil input returns nil (not a zero-length non-nil slice) — exercises the
// len==0 fast path identically to the empty-slice case.
func TestSample_CloneSamples_Ugly(t *testing.T) {
	if got := CloneSamples(nil); got != nil {
		t.Fatalf("CloneSamples(nil) = %v, want nil", got)
	}
}
