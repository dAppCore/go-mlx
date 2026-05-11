// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	memvid "dappco.re/go/inference/state"
	"dappco.re/go/mlx/bundle"
	"dappco.re/go/mlx/kv"
)

// Legacy aliases — the canonical state-bundle package lives at
// dappco.re/go/mlx/bundle/. mlx-root callers keep their existing
// StateBundle* surface via these aliases plus the wrapper constructors
// below.
type (
	StateBundle          = bundle.Bundle
	StateBundleModel     = bundle.Model
	StateBundlePrompt    = bundle.Prompt
	StateBundleTokenizer = bundle.Tokenizer
	StateBundleRuntime   = bundle.Runtime
	StateBundleAdapter   = bundle.Adapter
	StateBundleSampler   = bundle.Sampler
	StateBundleRef       = bundle.Ref
)

// Schema constants forwarded from the bundle package.
const (
	StateBundleVersion   = bundle.Version
	StateBundleKind      = bundle.Kind
	StateBundleRefMemvid = bundle.RefMemvid
)

// StateBundleOptions labels a state bundle with caller-owned provenance.
// Carries mlx-shaped ModelInfo + GenerateConfig at the boundary; the
// wrapper NewStateBundle converts to bundle.Options before delegating.
type StateBundleOptions struct {
	Model       string
	ModelPath   string
	ModelInfo   ModelInfo
	Prompt      string
	Tokenizer   StateBundleTokenizer
	Runtime     StateBundleRuntime
	Adapter     StateBundleAdapter
	AdapterPath string
	KVPath      string
	Sampler     GenerateConfig
	Analysis    *kv.Analysis
	SAMI        *SAMIResult
	Refs        []StateBundleRef
	MemvidRefs  []memvid.ChunkRef
	Meta        map[string]string
}

// NewStateBundle builds a portable state bundle around a restorable KV snapshot.
//
//	bundle, err := mlx.NewStateBundle(snapshot, opts)
func NewStateBundle(snapshot *kv.Snapshot, opts StateBundleOptions) (*StateBundle, error) {
	return bundle.New(snapshot, bundle.Options{
		Model:       opts.Model,
		ModelPath:   opts.ModelPath,
		Source:      modelInfoToBundle(opts.ModelInfo),
		Prompt:      opts.Prompt,
		Tokenizer:   opts.Tokenizer,
		Runtime:     opts.Runtime,
		Adapter:     opts.Adapter,
		AdapterPath: opts.AdapterPath,
		KVPath:      opts.KVPath,
		Sampler:     stateSamplerFromGenerateConfig(opts.Sampler),
		Analysis:    opts.Analysis,
		SAMI:        opts.SAMI,
		Refs:        opts.Refs,
		MemvidRefs:  opts.MemvidRefs,
		Meta:        opts.Meta,
	})
}

// ExportBundle captures a live session and returns a portable state bundle.
//
//	bundle, err := session.ExportBundle(opts)
func (s *ModelSession) ExportBundle(opts StateBundleOptions) (*StateBundle, error) {
	snapshot, err := s.CaptureKV()
	if err != nil {
		return nil, err
	}
	return NewStateBundle(snapshot, opts)
}

// LoadStateBundle reads a bundle saved by (*StateBundle).Save.
//
//	bundle, err := mlx.LoadStateBundle(path)
func LoadStateBundle(path string) (*StateBundle, error) {
	return bundle.Load(path)
}

// CheckStateBundleCompatibility verifies that a loaded model can safely restore a bundle.
//
//	if err := mlx.CheckStateBundleCompatibility(model.Info(), bundle); err != nil { … }
func CheckStateBundleCompatibility(info ModelInfo, b *StateBundle) error {
	return bundle.CheckCompatibility(modelInfoToBundle(info), b)
}

// StateBundleFileHash hashes an external file for strict bundle metadata.
//
//	hash, err := mlx.StateBundleFileHash(path)
func StateBundleFileHash(path string) (string, error) {
	return bundle.FileHash(path)
}

func stateSamplerFromGenerateConfig(cfg GenerateConfig) StateBundleSampler {
	return StateBundleSampler{
		MaxTokens:     cfg.MaxTokens,
		Temperature:   cfg.Temperature,
		TopK:          cfg.TopK,
		TopP:          cfg.TopP,
		MinP:          cfg.MinP,
		StopTokens:    append([]int32(nil), cfg.StopTokens...),
		RepeatPenalty: cfg.RepeatPenalty,
	}
}

func modelInfoToBundle(info ModelInfo) bundle.ModelInfo {
	return bundle.ModelInfo{
		Architecture:  info.Architecture,
		VocabSize:     info.VocabSize,
		NumLayers:     info.NumLayers,
		HiddenSize:    info.HiddenSize,
		QuantBits:     info.QuantBits,
		QuantGroup:    info.QuantGroup,
		ContextLength: info.ContextLength,
		Adapter:       info.Adapter,
	}
}

// stateBundleTokenizer fills missing Tokenizer hash fields. Retained as
// a mlx-root private helper for callers (session_agent_darwin,
// kv_snapshot_index) that use the old in-package name.
func stateBundleTokenizer(t StateBundleTokenizer) StateBundleTokenizer {
	return bundle.NormaliseTokenizer(t)
}

// stateHash returns the SHA-256 hex of a string. Retained as a
// mlx-root private helper for callers (kv_snapshot_index) that use the
// old in-package name.
func stateHash(s string) string {
	return bundle.HashString(s)
}

// stateMemvidURI renders a memvid chunk reference as a memvid:// URI.
// Retained as a mlx-root private helper for state_bundle_test.go.
func stateMemvidURI(ref memvid.ChunkRef) string {
	return bundle.MemvidURI(ref)
}

