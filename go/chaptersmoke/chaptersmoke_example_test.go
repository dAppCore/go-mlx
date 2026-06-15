// SPDX-Licence-Identifier: EUPL-1.2

package chaptersmoke_test

import (
	"context"
	"fmt"
	"time"

	state "dappco.re/go/inference/state"
	"dappco.re/go/mlx/chaptersmoke"
	"dappco.re/go/mlx/kv"
)

// exampleSnapshot is a tiny single-layer KV snapshot — enough to write real
// State blocks to the file-log store without loading a model.
func exampleSnapshot() *kv.Snapshot {
	return &kv.Snapshot{
		Version:       kv.SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1, 2, 3},
		TokenOffset:   3,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        3,
		HeadDim:       2,
		NumQueryHeads: 1,
		Layers: []kv.LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []kv.HeadSnapshot{{
				Key:   []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
				Value: []float32{0.6, 0.5, 0.4, 0.3, 0.2, 0.1},
			}},
		}},
	}
}

// ExampleRun shows the chapter-smoke harness driven by a fake Runner: Capture
// writes a snapshot's KV state to the store, Generate returns a fixed answer.
// No model is loaded — the callbacks supply all model-specific behaviour, so
// the example is deterministic. Only the plausibility verdict and answer text
// are printed (durations and paths are runtime-dependent).
func ExampleRun() {
	runner := chaptersmoke.Runner{
		Capture: func(ctx context.Context, prompt string, store state.Writer, opts kv.StateBlockOptions) (*kv.StateBlockBundle, error) {
			return exampleSnapshot().SaveStateBlocks(ctx, store, opts)
		},
		Generate: func(ctx context.Context, store state.Store, bundle *kv.StateBlockBundle, prefixTokens int, suffix string) (chaptersmoke.Generation, error) {
			return chaptersmoke.Generation{
				Text:           "Marcus opens the sealed letter.",
				DecodeDuration: time.Millisecond,
			}, nil
		},
	}

	report, err := chaptersmoke.Run(context.Background(), runner, chaptersmoke.Config{
		StoreDir:        "", // empty → a temp dir is created for the run
		BlockSize:       2,
		AnswerMaxTokens: 8,
		Chapters: []chaptersmoke.Input{{
			Name:          "Chapter 1",
			Text:          "Chapter 1. Marcus opens the sealed letter and names the risk.",
			Question:      "who opens the sealed letter?",
			ExpectedTerms: []string{"Marcus"},
		}},
	})
	if err != nil {
		fmt.Println("run error:", err)
		return
	}
	chapter := report.Chapters[0]
	fmt.Println("answer:", chapter.Answer)
	fmt.Println("plausible:", chapter.Plausible)
	// Output:
	// answer: Marcus opens the sealed letter.
	// plausible: true
}

// ExampleRun_unsupportedStoreKind shows the store-kind validation contract: an
// unrecognised StoreKind is rejected up front — before any store is opened or
// either callback runs — with a stable sentinel error. This is the guard a
// caller hits when a typo'd or future-only backend name reaches Run.
func ExampleRun_unsupportedStoreKind() {
	runner := chaptersmoke.Runner{
		Capture: func(context.Context, string, state.Writer, kv.StateBlockOptions) (*kv.StateBlockBundle, error) {
			return nil, nil // never invoked — validation fails first
		},
		Generate: func(context.Context, state.Store, *kv.StateBlockBundle, int, string) (chaptersmoke.Generation, error) {
			return chaptersmoke.Generation{}, nil // never invoked
		},
	}

	_, err := chaptersmoke.Run(context.Background(), runner, chaptersmoke.Config{
		StoreKind: "rot13-video", // not file-log, and not a known alias
		Chapters:  []chaptersmoke.Input{{Text: "Chapter 1. Marcus.", Question: "who?"}},
	})
	fmt.Println(err)
	// Output:
	// chaptersmoke: unsupported store kind
}
