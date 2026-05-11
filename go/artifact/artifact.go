// SPDX-Licence-Identifier: EUPL-1.2

// Package artifact exports compact session-state records — KV provenance,
// optional binary KV snapshots, and SAMI visualisation data — that can be
// archived to memvid stores or local files.
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
	memvid "dappco.re/go/inference/state"
	"dappco.re/go/mlx/bundle"
	"dappco.re/go/mlx/kv"
)

// Kind labels session-state artifacts written by this package.
const Kind = "go-mlx/session-state"

// Options controls local model-state artifact export.
type Options struct {
	Model    string
	Prompt   string
	Analysis *kv.Analysis
	KVPath   string
	Store    memvid.Writer
	URI      string
	Title    string
	Kind     string
	Track    string
	Tags     map[string]string
	Labels   []string
}

// Record is the compact JSON payload written into a memvid chunk.
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
	ChunkRef      memvid.ChunkRef   `json:"chunk_ref,omitempty"`
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

// Export writes optional KV binary data and optional memvid JSON for the
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
		return nil, core.NewError("artifact: KV snapshot is nil")
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
		FeatureLabels: kv.FeatureLabels(),
		SAMI:          bundle.SAMIFromKV(snapshot, analysis, bundle.SAMIOptions{Model: opts.Model, Prompt: opts.Prompt}),
		KVPath:        opts.KVPath,
	}
	if opts.Store != nil {
		data := core.JSONMarshalIndent(record, "", "  ")
		if !data.OK {
			return nil, core.E("artifact.Export", "marshal record", resultError(data))
		}
		ref, err := opts.Store.Put(ctx, string(data.Value.([]byte)), memvid.PutOptions{
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
	return core.NewError("core result failed")
}
