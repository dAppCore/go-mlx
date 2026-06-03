// SPDX-Licence-Identifier: EUPL-1.2

// Package artifact exports compact session-state records — KV provenance,
// optional binary KV snapshots, and SAMI visualisation data — that can be
// archived to State stores or local files.
//
//	record, err := artifact.Export(ctx, snapshot, artifact.Options{
//	    Model: "gemma3-1b",
//	    Store: store,
//	    URI:   "mlx://session/trace-1",
//	})
package artifact

import (
	"context"

	core "dappco.re/go"
	state "dappco.re/go/inference/state"
	"dappco.re/go/mlx/bundle"
	"dappco.re/go/mlx/kv"
)

// Kind labels session-state artifacts written by this package.
const Kind = "go-mlx/session-state"

// errSnapshotNil is the sentinel returned when Export is invoked without
// a KV snapshot. Hoisted to a package var so the nil-guard at the top
// of Export does not allocate a fresh *Err on every call.
var errSnapshotNil = core.NewError("artifact: KV snapshot is nil")

// errResultFailed is the fallback sentinel returned by resultError when
// a core.Result reports !OK but its Value is not an error. Hoisted to a
// package var to avoid allocating on this rare-but-hot helper path.
var errResultFailed = core.NewError("core result failed")

// cachedFeatureLabels is the package-once-cached result of kv.FeatureLabels.
// kv.FeatureLabels allocates a fresh slice every call (currently 7 strings);
// Export embeds the slice once per Record so the labels alloc fires on
// every Export call. The label list is invariant — kv exposes it as the
// stable order matching Features — so it is safe to compute once at
// package init and share across all Exports. Callers must NOT mutate the
// slice (none currently do; Records that travel to JSON only ever read).
var cachedFeatureLabels = kv.FeatureLabels()

// Options controls local model-state artifact export.
type Options struct {
	Model    string
	Prompt   string
	Analysis *kv.Analysis
	KVPath   string
	Store    state.Writer
	URI      string
	Title    string
	Kind     string
	Track    string
	Tags     map[string]string
	Labels   []string
}

// Record is the compact JSON payload written into a State chunk.
type Record struct {
	Version       int               `json:"version"`
	Kind          string            `json:"kind"`
	Model         string            `json:"model"`
	Prompt        string            `json:"prompt"`
	Snapshot      Snapshot          `json:"snapshot"`
	Analysis      *kv.Analysis      `json:"analysis"`
	Features      []float64         `json:"features"`
	FeatureLabels []string          `json:"feature_labels"`
	SAMI          bundle.SAMIResult `json:"sami"`
	KVPath        string            `json:"kv_path,omitempty"`
	ChunkRef      state.ChunkRef    `json:"chunk_ref"`
}

// Snapshot is the lightweight tensor provenance stored in text chunks.
type Snapshot struct {
	Architecture  string `json:"architecture"`
	TokenCount    int    `json:"token_count"`
	NumLayers     int    `json:"num_layers"`
	NumHeads      int    `json:"num_heads"`
	SeqLen        int    `json:"seq_len"`
	HeadDim       int    `json:"head_dim"`
	NumQueryHeads int    `json:"num_query_heads"`
}

// Export writes optional KV binary data and optional State JSON for the
// supplied KV snapshot.
//
//	record, err := artifact.Export(ctx, snapshot, artifact.Options{KVPath: "/tmp/state.kv"})
func Export(ctx context.Context, snapshot *kv.Snapshot, opts Options) (*Record, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}
	if snapshot == nil {
		return nil, errSnapshotNil
	}
	if opts.KVPath != "" {
		if err := snapshot.Save(opts.KVPath); err != nil {
			return nil, err
		}
	}
	analysis := opts.Analysis
	if analysis == nil {
		analysis = kv.Analyze(snapshot)
	}
	record := &Record{
		Version: 1,
		Kind:    Kind,
		Model:   opts.Model,
		Prompt:  opts.Prompt,
		Snapshot: Snapshot{
			Architecture:  snapshot.Architecture,
			TokenCount:    len(snapshot.Tokens),
			NumLayers:     snapshot.NumLayers,
			NumHeads:      snapshot.NumHeads,
			SeqLen:        snapshot.SeqLen,
			HeadDim:       snapshot.HeadDim,
			NumQueryHeads: snapshot.NumQueryHeads,
		},
		Analysis:      analysis,
		Features:      kv.Features(analysis),
		FeatureLabels: cachedFeatureLabels,
		SAMI:          bundle.SAMIFromKV(snapshot, analysis, bundle.SAMIOptions{Model: opts.Model, Prompt: opts.Prompt}),
		KVPath:        opts.KVPath,
	}
	if opts.Store != nil {
		data := core.JSONMarshalIndent(record, "", "  ")
		if !data.OK {
			return nil, core.E("artifact.Export", "marshal record", resultError(data))
		}
		// JSONMarshalIndent returns a fresh buffer that nothing else
		// references; AsString aliases it into the string Put requires
		// without the extra copy a `string(...)` cast emits. The buffer
		// stays alive via the alias because Put retains the string.
		marshalled := data.Value.([]byte)
		ref, err := opts.Store.Put(ctx, core.AsString(marshalled), state.PutOptions{
			URI:    opts.URI,
			Title:  opts.Title,
			Kind:   opts.Kind,
			Track:  opts.Track,
			Tags:   opts.Tags,
			Labels: opts.Labels,
		})
		if err != nil {
			return nil, err
		}
		record.ChunkRef = ref
	}
	return record, nil
}

func resultError(result core.Result) error {
	if result.OK {
		return nil
	}
	if err, ok := result.Value.(error); ok {
		return err
	}
	return errResultFailed
}
