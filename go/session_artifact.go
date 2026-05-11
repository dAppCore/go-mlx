// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"

	core "dappco.re/go"
	memvid "dappco.re/go/inference/state"
	"dappco.re/go/mlx/bundle"
	"dappco.re/go/mlx/kv"
)

const sessionArtifactKind = "go-mlx/session-state"

// SAMIResult is the SAMI BOResult-compatible model-state visualization
// schema. Aliased from dappco.re/go/mlx/bundle/.
type SAMIResult = bundle.SAMIResult

// SAMIOptions labels a SAMI export with caller-owned provenance.
// Aliased from dappco.re/go/mlx/bundle/.
type SAMIOptions = bundle.SAMIOptions

// SessionArtifactOptions controls local model-state artifact export.
type SessionArtifactOptions struct {
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

// SessionArtifact is the compact JSON payload written into a memvid chunk.
type SessionArtifact struct {
	Version       int                     `json:"version"`
	Kind          string                  `json:"kind"`
	Model         string                  `json:"model"`
	Prompt        string                  `json:"prompt"`
	Snapshot      SessionArtifactSnapshot `json:"snapshot"`
	Analysis      *kv.Analysis             `json:"analysis"`
	Features      []float64               `json:"features"`
	FeatureLabels []string                `json:"feature_labels"`
	SAMI          SAMIResult              `json:"sami"`
	KVPath        string                  `json:"kv_path,omitempty"`
	ChunkRef      memvid.ChunkRef         `json:"chunk_ref,omitempty"`
}

// SessionArtifactSnapshot is the lightweight tensor provenance stored in text chunks.
type SessionArtifactSnapshot struct {
	Architecture  string `json:"architecture"`
	TokenCount    int    `json:"token_count"`
	NumLayers     int    `json:"num_layers"`
	NumHeads      int    `json:"num_heads"`
	SeqLen        int    `json:"seq_len"`
	HeadDim       int    `json:"head_dim"`
	NumQueryHeads int    `json:"num_query_heads"`
}

// SAMIFromKV converts K/V analysis into SAMI's visualization schema.
//
//	sami := mlx.SAMIFromKV(snapshot, analysis, mlx.SAMIOptions{Model: name})
func SAMIFromKV(snapshot *kv.Snapshot, analysis *kv.Analysis, opts SAMIOptions) SAMIResult {
	return bundle.SAMIFromKV(snapshot, analysis, opts)
}

// ExportSessionArtifacts writes optional KV binary data and optional memvid JSON.
func ExportSessionArtifacts(ctx context.Context, snapshot *kv.Snapshot, opts SessionArtifactOptions) (*SessionArtifact, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}
	if snapshot == nil {
		return nil, core.NewError("mlx: KV snapshot is nil")
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
	artifact := &SessionArtifact{
		Version: 1,
		Kind:    sessionArtifactKind,
		Model:   opts.Model,
		Prompt:  opts.Prompt,
		Snapshot: SessionArtifactSnapshot{
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
		SAMI:          SAMIFromKV(snapshot, analysis, SAMIOptions{Model: opts.Model, Prompt: opts.Prompt}),
		KVPath:        opts.KVPath,
	}
	if opts.Store != nil {
		data := core.JSONMarshalIndent(artifact, "", "  ")
		if !data.OK {
			return nil, core.E("ExportSessionArtifacts", "marshal artifact", sessionArtifactResultError(data))
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
		artifact.ChunkRef = ref
	}
	return artifact, nil
}

// ExportArtifacts captures the session state and exports it as local artifacts.
func (s *ModelSession) ExportArtifacts(opts SessionArtifactOptions) (*SessionArtifact, error) {
	snapshot, err := s.CaptureKV()
	if err != nil {
		return nil, err
	}
	return ExportSessionArtifacts(context.Background(), snapshot, opts)
}

func sessionArtifactResultError(result core.Result) error {
	if result.OK {
		return nil
	}
	if err, ok := result.Value.(error); ok {
		return err
	}
	return core.NewError("core result failed")
}

