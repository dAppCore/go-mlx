// SPDX-Licence-Identifier: EUPL-1.2

package memorypretrain

import (
	"context"
	"errors"
	"strings"
	"testing"
)

func runtimeMemoryBank(t *testing.T) *FFNMemoryBank {
	t.Helper()
	bank, err := NewFFNMemoryBank(FFNMemoryConfig{
		HiddenSize:       2,
		Layers:           1,
		MemoryLevels:     []string{"1"},
		FFNMemoryTokens:  []int{1},
		NumClusters:      []int{2},
		AddedGenericSize: 1,
	})
	if err != nil {
		t.Fatalf("NewFFNMemoryBank() error = %v", err)
	}
	return bank
}

// TestFfnMemoryRuntime_NewFFNMemoryRuntime_Good builds a runtime facade in both
// supported modes: a generic-only runtime needs no router or embedder, and a
// routed runtime binds a router with its anchor embedder.
func TestFfnMemoryRuntime_NewFFNMemoryRuntime_Good(t *testing.T) {
	mem := runtimeMemoryBank(t)
	generic, err := NewFFNMemoryRuntime(mem, nil, nil)
	if err != nil {
		t.Fatalf("NewFFNMemoryRuntime(generic) error = %v", err)
	}
	if generic.Memory != mem || generic.Router != nil || generic.Embedder != nil {
		t.Fatalf("NewFFNMemoryRuntime(generic) = %+v, want memory-only facade", generic)
	}
	router, err := BuildBank([]Block{
		{ID: "a", Embedding: []float32{1, 0}},
		{ID: "b", Embedding: []float32{0, 1}},
	}, BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 1, KMeansIters: 8})
	if err != nil {
		t.Fatalf("BuildBank() error = %v", err)
	}
	embedder := EmbedFunc(func(context.Context, string) ([]float32, error) { return []float32{1, 0}, nil })
	routed, err := NewFFNMemoryRuntime(mem, router, embedder)
	if err != nil {
		t.Fatalf("NewFFNMemoryRuntime(routed) error = %v", err)
	}
	if routed.Router != router || routed.Embedder == nil {
		t.Fatalf("NewFFNMemoryRuntime(routed) = %+v, want router and embedder bound", routed)
	}
}

// TestFfnMemoryRuntime_NewFFNMemoryRuntime_Bad rejects a nil memory bank.
func TestFfnMemoryRuntime_NewFFNMemoryRuntime_Bad(t *testing.T) {
	if _, err := NewFFNMemoryRuntime(nil, nil, nil); err == nil {
		t.Fatal("NewFFNMemoryRuntime(nil memory) error = nil")
	}
}

// TestFfnMemoryRuntime_NewFFNMemoryRuntime_Ugly drives the router-without-embedder
// guard: a router is configured but no embedder is supplied, which the
// constructor rejects rather than building a runtime that cannot route.
func TestFfnMemoryRuntime_NewFFNMemoryRuntime_Ugly(t *testing.T) {
	mem := runtimeMemoryBank(t)
	router, err := BuildBank([]Block{{ID: "a", Embedding: []float32{1, 0}}}, BuildConfig{})
	if err != nil {
		t.Fatalf("BuildBank() error = %v", err)
	}
	if _, err := NewFFNMemoryRuntime(mem, router, nil); err == nil {
		t.Fatal("NewFFNMemoryRuntime(router without embedder) error = nil")
	}
}

// TestFfnMemoryRuntime_FFNMemoryRuntime_AddTextToFFNOutput_Good embeds the query
// text and applies the selected memory in both modes: routed through the learned
// clustering bank, and through the generic fallback when no router is configured.
func TestFfnMemoryRuntime_FFNMemoryRuntime_AddTextToFFNOutput_Good(t *testing.T) {
	t.Run("routes through embedder", func(t *testing.T) {
		router, err := BuildBank([]Block{
			{ID: "go-1", Embedding: []float32{1, 0}},
			{ID: "go-2", Embedding: []float32{0.9, 0.1}},
			{ID: "poem-1", Embedding: []float32{0, 1}},
			{ID: "poem-2", Embedding: []float32{0.1, 0.9}},
		}, BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 1, KMeansIters: 8})
		if err != nil {
			t.Fatalf("BuildBank() error = %v", err)
		}
		clusterIDs, err := router.ClusterIDs([]float32{1, 0})
		if err != nil {
			t.Fatalf("ClusterIDs() error = %v", err)
		}
		mem, err := NewFFNMemoryBank(FFNMemoryConfig{
			HiddenSize:       2,
			Layers:           1,
			MemoryLevels:     []string{"1"},
			FFNMemoryTokens:  []int{1},
			NumClusters:      []int{2},
			AddedGenericSize: 1,
		})
		if err != nil {
			t.Fatalf("NewFFNMemoryBank() error = %v", err)
		}
		level := &mem.Layers[0].Levels[0]
		level.W1 = []float32{0, 0, 0, 0, 0, 0}
		level.W2 = []float32{0, 0, 0, 0, 0, 0}
		level.W3 = []float32{0, 0, 0, 0, 0, 0}
		cluster := clusterIDs[0]
		level.W1[cluster*2] = 1
		level.W2[cluster*2+1] = 1
		level.W3[cluster*2] = 2
		level.W3[cluster*2+1] = 3
		embedCalls := 0
		runtime, err := NewFFNMemoryRuntime(mem, router, EmbedFunc(func(_ context.Context, text string) ([]float32, error) {
			embedCalls++
			if text != "Go memory planning" {
				t.Fatalf("embedded text = %q, want model-side query text", text)
			}
			return []float32{1, 0}, nil
		}))
		if err != nil {
			t.Fatalf("NewFFNMemoryRuntime() error = %v", err)
		}

		out, ids, stats, err := runtime.AddTextToFFNOutput(context.Background(), nil, []float32{1, 2}, []float32{2, 4}, "Go memory planning", 0)
		if err != nil {
			t.Fatalf("AddTextToFFNOutput() error = %v", err)
		}
		wantContribution := siluTest(2) * 4
		want := []float32{1 + 2*wantContribution, 2 + 3*wantContribution}
		if embedCalls != 1 || len(ids) != 1 || ids[0] != cluster || len(out) != 2 || !approx32(out[0], want[0]) || !approx32(out[1], want[1]) {
			t.Fatalf("AddTextToFFNOutput() calls=%d out=%+v ids=%+v, want out=%+v ids=%+v", embedCalls, out, ids, want, clusterIDs)
		}
		if !stats.Applied || stats.LevelsApplied != 1 {
			t.Fatalf("stats = %+v, want routed runtime memory applied", stats)
		}
	})

	t.Run("uses generic fallback", func(t *testing.T) {
		mem, err := NewFFNMemoryBank(FFNMemoryConfig{
			HiddenSize:       2,
			Layers:           1,
			MemoryLevels:     []string{"1"},
			FFNMemoryTokens:  []int{1},
			NumClusters:      []int{2},
			AddedGenericSize: 1,
		})
		if err != nil {
			t.Fatalf("NewFFNMemoryBank() error = %v", err)
		}
		level := &mem.Layers[0].Levels[0]
		level.W1 = []float32{0, 0, 0, 0, 1, 0}
		level.W2 = []float32{0, 0, 0, 0, 0, 1}
		level.W3 = []float32{0, 0, 0, 0, 2, 3}
		runtime, err := NewFFNMemoryRuntime(mem, nil, nil)
		if err != nil {
			t.Fatalf("NewFFNMemoryRuntime(generic) error = %v", err)
		}

		out, ids, stats, err := runtime.AddTextToFFNOutput(context.Background(), nil, []float32{5, 7}, []float32{2, 4}, "", 0)
		if err != nil {
			t.Fatalf("AddTextToFFNOutput(generic) error = %v", err)
		}
		wantContribution := siluTest(2) * 4
		want := []float32{5 + 2*wantContribution, 7 + 3*wantContribution}
		if len(ids) != 1 || ids[0] != 2 || len(out) != 2 || !approx32(out[0], want[0]) || !approx32(out[1], want[1]) {
			t.Fatalf("AddTextToFFNOutput(generic) out=%+v ids=%+v, want out=%+v ids=[2]", out, ids, want)
		}
		if !stats.Applied || stats.LevelsApplied != 1 {
			t.Fatalf("stats = %+v, want generic runtime memory applied", stats)
		}
	})
}

// TestFfnMemoryRuntime_FFNMemoryRuntime_AddTextToFFNOutput_Bad covers the
// method-side guards: a nil receiver, a nil-memory runtime, and a runtime whose
// router is set but whose embedder was cleared after construction.
func TestFfnMemoryRuntime_FFNMemoryRuntime_AddTextToFFNOutput_Bad(t *testing.T) {
	mem := runtimeMemoryBank(t)
	router, err := BuildBank([]Block{{ID: "a", Embedding: []float32{1, 0}}}, BuildConfig{})
	if err != nil {
		t.Fatalf("BuildBank() error = %v", err)
	}
	// Nil receiver and a nil-memory runtime both error from the method.
	if _, _, _, err := (*FFNMemoryRuntime)(nil).AddTextToFFNOutput(context.Background(), nil, []float32{1, 2}, []float32{3, 4}, "", 0); err == nil {
		t.Fatal("AddTextToFFNOutput(nil receiver) error = nil")
	}
	if _, _, _, err := (&FFNMemoryRuntime{}).AddTextToFFNOutput(context.Background(), nil, []float32{1, 2}, []float32{3, 4}, "", 0); err == nil {
		t.Fatal("AddTextToFFNOutput(nil memory) error = nil")
	}
	// A runtime whose router is set but embedder was cleared after construction
	// rejects the call rather than routing without an embedder.
	if _, _, _, err := (&FFNMemoryRuntime{Memory: mem, Router: router}).AddTextToFFNOutput(context.Background(), nil, []float32{1, 2}, []float32{3, 4}, "x", 0); err == nil {
		t.Fatal("AddTextToFFNOutput(router without embedder) error = nil")
	}
}

// TestFfnMemoryRuntime_FFNMemoryRuntime_AddTextToFFNOutput_Ugly drives the
// context-cancellation and embedder-error paths: cancellation before the call
// skips embedding, an embedder error is wrapped with call context, and an
// embedder that cancels mid-call trips the post-embed cancellation guard.
func TestFfnMemoryRuntime_FFNMemoryRuntime_AddTextToFFNOutput_Ugly(t *testing.T) {
	mem := runtimeMemoryBank(t)
	router, err := BuildBank([]Block{
		{ID: "a", Embedding: []float32{1, 0}},
		{ID: "b", Embedding: []float32{0, 1}},
	}, BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 1, KMeansIters: 8})
	if err != nil {
		t.Fatalf("BuildBank() error = %v", err)
	}

	// A context cancelled before the call returns early without embedding.
	embedCalls := 0
	embedder := EmbedFunc(func(_ context.Context, _ string) ([]float32, error) {
		embedCalls++
		return []float32{1, 0}, nil
	})
	runtime, err := NewFFNMemoryRuntime(mem, router, embedder)
	if err != nil {
		t.Fatalf("NewFFNMemoryRuntime() error = %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, _, err := runtime.AddTextToFFNOutput(ctx, nil, []float32{1, 2}, []float32{3, 4}, "x", 0); !errors.Is(err, context.Canceled) {
		t.Fatalf("AddTextToFFNOutput(cancelled) error = %v, want context.Canceled", err)
	}
	if embedCalls != 0 {
		t.Fatalf("embed calls = %d, want cancellation before embedding", embedCalls)
	}

	// An embedder error is wrapped with call context.
	failing, err := NewFFNMemoryRuntime(mem, router, EmbedFunc(func(context.Context, string) ([]float32, error) {
		return nil, errors.New("anchor offline")
	}))
	if err != nil {
		t.Fatalf("NewFFNMemoryRuntime(failing) error = %v", err)
	}
	if _, _, _, err := failing.AddTextToFFNOutput(context.Background(), nil, []float32{1, 2}, []float32{3, 4}, "x", 0); err == nil || !strings.Contains(err.Error(), "embed query text") {
		t.Fatalf("AddTextToFFNOutput(embed error) error = %v, want embed context", err)
	}

	// An embedder that cancels the context mid-call passes the pre-embed check
	// but trips the post-embed cancellation guard before routing.
	midCtx, midCancel := context.WithCancel(context.Background())
	midRuntime, err := NewFFNMemoryRuntime(mem, router, EmbedFunc(func(context.Context, string) ([]float32, error) {
		midCancel()
		return []float32{1, 0}, nil
	}))
	if err != nil {
		t.Fatalf("NewFFNMemoryRuntime(mid-cancel) error = %v", err)
	}
	if _, _, _, err := midRuntime.AddTextToFFNOutput(midCtx, nil, []float32{1, 2}, []float32{3, 4}, "x", 0); !errors.Is(err, context.Canceled) {
		t.Fatalf("AddTextToFFNOutput(cancel during embed) error = %v, want context.Canceled", err)
	}
}
