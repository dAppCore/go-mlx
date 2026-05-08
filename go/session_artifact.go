// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"math"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/memvid"
)

const sessionArtifactKind = "go-mlx/session-state"

// SAMIResult is the SAMI BOResult-compatible model-state visualization schema.
type SAMIResult struct {
	Model               string    `json:"model"`
	Prompt              string    `json:"prompt"`
	Architecture        string    `json:"architecture"`
	NumLayers           int       `json:"num_layers"`
	NumHeads            int       `json:"num_heads"`
	SeqLen              int       `json:"seq_len"`
	HeadDim             int       `json:"head_dim"`
	MeanCoherence       float64   `json:"mean_coherence"`
	MeanCrossAlignment  float64   `json:"mean_cross_alignment"`
	MeanHeadEntropy     float64   `json:"mean_head_entropy"`
	PhaseLockScore      float64   `json:"phase_lock_score"`
	JointCollapseCount  int       `json:"joint_collapse_count"`
	LayerCoherence      []float64 `json:"layer_coherence"`
	LayerCrossAlignment []float64 `json:"layer_cross_alignment"`
	Composite           float64   `json:"composite"`
}

// SAMIOptions labels a SAMI export with caller-owned provenance.
type SAMIOptions struct {
	Model  string
	Prompt string
}

// SessionArtifactOptions controls local model-state artifact export.
type SessionArtifactOptions struct {
	Model    string
	Prompt   string
	Analysis *KVAnalysis
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
	Analysis      *KVAnalysis             `json:"analysis"`
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
func SAMIFromKV(snapshot *KVSnapshot, analysis *KVAnalysis, opts SAMIOptions) SAMIResult {
	if snapshot == nil {
		return SAMIResult{}
	}
	if analysis == nil {
		analysis = AnalyzeKV(snapshot)
	}
	numLayers := snapshot.NumLayers
	if numLayers <= 0 {
		numLayers = len(snapshot.Layers)
	}
	meanCoherence := meanUnit(analysis.MeanKeyCoherence, analysis.MeanValueCoherence)
	meanCross := clampUnit(analysis.MeanCrossAlignment)
	layerCoherence := make([]float64, numLayers)
	layerCross := make([]float64, numLayers)
	for layer := range numLayers {
		layerCoherence[layer] = meanUnit(
			layerMetric(analysis.LayerKeyCoherence, layer, analysis.MeanKeyCoherence),
			layerMetric(analysis.LayerValueCoherence, layer, analysis.MeanValueCoherence),
		)
		layerCross[layer] = layerMetric(analysis.LayerCrossAlignment, layer, analysis.MeanCrossAlignment)
	}
	jointCollapseCount := analysis.JointCollapseCount
	if jointCollapseCount < 0 {
		jointCollapseCount = 0
	}
	if numLayers > 0 && jointCollapseCount > numLayers {
		jointCollapseCount = numLayers
	}
	return SAMIResult{
		Model:               opts.Model,
		Prompt:              opts.Prompt,
		Architecture:        snapshot.Architecture,
		NumLayers:           numLayers,
		NumHeads:            snapshot.NumHeads,
		SeqLen:              snapshot.SeqLen,
		HeadDim:             snapshot.HeadDim,
		MeanCoherence:       meanCoherence,
		MeanCrossAlignment:  meanCross,
		MeanHeadEntropy:     clampUnit(analysis.MeanHeadEntropy),
		PhaseLockScore:      clampUnit(analysis.PhaseLockScore),
		JointCollapseCount:  jointCollapseCount,
		LayerCoherence:      layerCoherence,
		LayerCrossAlignment: layerCross,
		Composite:           clampRange(float64(analysis.Composite())/100.0, 0, 100),
	}
}

// ExportSessionArtifacts writes optional KV binary data and optional memvid JSON.
func ExportSessionArtifacts(ctx context.Context, snapshot *KVSnapshot, opts SessionArtifactOptions) (*SessionArtifact, error) {
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
		analysis = AnalyzeKV(snapshot)
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
		Features:      KVFeatures(analysis),
		FeatureLabels: KVFeatureLabels(),
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

func layerMetric(values []float64, index int, fallback float64) float64 {
	if index >= 0 && index < len(values) {
		return clampUnit(values[index])
	}
	return clampUnit(fallback)
}

func meanUnit(a, b float64) float64 {
	return clampUnit((clampUnit(a) + clampUnit(b)) / 2.0)
}

func clampUnit(value float64) float64 {
	return clampRange(value, 0, 1)
}

func clampRange(value, minValue, maxValue float64) float64 {
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return minValue
	}
	if value < minValue {
		return minValue
	}
	if value > maxValue {
		return maxValue
	}
	return value
}
