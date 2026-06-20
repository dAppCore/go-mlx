// SPDX-Licence-Identifier: EUPL-1.2

package artifact

import (
	"context"
	"math"
	"testing"

	core "dappco.re/go"
	state "dappco.re/go/inference/state"
	"dappco.re/go/mlx/kv"
)

// failingStore is a state.Writer whose Put always errors — exercises the
// Store.Put failure return inside Export.
type failingStore struct{ err error }

func (f failingStore) Put(_ context.Context, _ string, _ state.PutOptions) (state.ChunkRef, error) {
	return state.ChunkRef{}, f.err
}

// TestExport_NilContext drives the ctx == nil branch: Export must default
// to context.Background() and succeed exactly as if a live context were
// supplied. The existing TestExport_Good always passes a non-nil context,
// so the nil-substitution path is otherwise unreached.
func TestExport_NilContext(t *testing.T) {
	//nolint:staticcheck // SA1012: passing a nil context is the behaviour under test.
	record, err := Export(nil, testSnapshot(), Options{Model: "lem-gemma"})
	if err != nil {
		t.Fatalf("Export(nil ctx) error = %v", err)
	}
	if record == nil {
		t.Fatal("Export(nil ctx) record = nil")
	}
	if record.SAMI.Model != "lem-gemma" {
		t.Fatalf("record.SAMI.Model = %q, want %q", record.SAMI.Model, "lem-gemma")
	}
}

// TestExport_SaveError drives the snapshot.Save failure return: a KVPath
// rooted under a path component that does not exist makes core.WriteFile
// fail (os.WriteFile does not create parent directories), so Export must
// propagate that error before building the record.
func TestExport_SaveError(t *testing.T) {
	badPath := core.PathJoin(t.TempDir(), "no-such-dir", "state.kvbin")

	record, err := Export(context.Background(), testSnapshot(), Options{KVPath: badPath})
	if err == nil {
		t.Fatal("Export() with unwritable KVPath: expected error, got nil")
	}
	if record != nil {
		t.Fatalf("Export() record = %+v, want nil on save error", record)
	}
}

// TestExport_MarshalError drives the !data.OK marshal-failure return (and,
// through it, the error-bearing branch of resultError). A NaN in the
// supplied analysis propagates into the Record's float64 fields, which
// encoding/json refuses to marshal ("unsupported value: NaN"). A Store is
// set so the marshal path is reached.
func TestExport_MarshalError(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	analysis := &kv.Analysis{MeanKeyCoherence: math.NaN()}

	record, err := Export(context.Background(), testSnapshot(), Options{
		Model:    "lem-gemma",
		Analysis: analysis,
		Store:    store,
		URI:      "mlx://session/marshal-fail",
	})
	if err == nil {
		t.Fatal("Export() with NaN analysis: expected marshal error, got nil")
	}
	if record != nil {
		t.Fatalf("Export() record = %+v, want nil on marshal error", record)
	}
	if !core.Contains(err.Error(), "marshal record") {
		t.Fatalf("Export() error = %v, want marshal record wrap", err)
	}
}

// TestExport_StorePutError drives the Store.Put failure return: a Writer
// whose Put always errors makes Export propagate that error verbatim
// (no wrap) after a successful marshal.
func TestExport_StorePutError(t *testing.T) {
	sentinel := core.NewError("artifact-test: put rejected")
	store := failingStore{err: sentinel}

	record, err := Export(context.Background(), testSnapshot(), Options{
		Model: "lem-gemma",
		Store: store,
		URI:   "mlx://session/put-fail",
	})
	if !core.Is(err, sentinel) {
		t.Fatalf("Export() error = %v, want %v", err, sentinel)
	}
	if record != nil {
		t.Fatalf("Export() record = %+v, want nil on Put error", record)
	}
}

// TestResultError exercises resultError directly across all three branches
// — they are not all reachable through Export (Export only calls it on a
// marshal failure, whose Result.Value is always an error).
func TestResultError(t *testing.T) {
	// OK result → nil.
	if got := resultError(core.Ok([]byte("payload"))); got != nil {
		t.Fatalf("resultError(Ok) = %v, want nil", got)
	}

	// !OK with an error Value → that error returned verbatim.
	sentinel := core.NewError("artifact-test: boom")
	if got := resultError(core.Result{Value: sentinel, OK: false}); !core.Is(got, sentinel) {
		t.Fatalf("resultError(error value) = %v, want %v", got, sentinel)
	}

	// !OK with a non-error Value → the errResultFailed fallback sentinel.
	got := resultError(core.Result{Value: "not-an-error", OK: false})
	if !core.Is(got, errResultFailed) {
		t.Fatalf("resultError(non-error value) = %v, want errResultFailed", got)
	}
}
