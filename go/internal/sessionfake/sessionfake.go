// SPDX-Licence-Identifier: EUPL-1.2

// Package sessionfake provides the shared in-memory metal.SessionHandle
// fixture used by the root mlx tests (Model.NewSession / agent-memory
// entry points) and the session package tests. It records every call so
// assertions can inspect what reached the native layer, and implements
// the optional capability interfaces (chunk/token prefill+append, KV
// block capture/restore) the session machinery probes for.
package sessionfake

import (
	"context"
	"iter"

	"dappco.re/go/mlx/pkg/metal"
)

// Handle is a recording fake metal.SessionHandle. Zero value is usable;
// seed the exported fields to steer behaviour (KV for capture results,
// Tokens for generation output, *Err to force failures).
type Handle struct {
	PrefillPrompt     string
	AppendPromptSeen  string
	PrefillChunksSeen []string
	AppendChunksSeen  []string
	PrefillTokensSeen []int32
	AppendTokensSeen  []int32
	PrefillErr        error
	AppendErr         error
	Tokens            []metal.Token
	Cfg               metal.GenerateConfig
	GenerateCalls     int
	ProbeEvents       []metal.ProbeEvent
	AfterGenerate     func(*Handle)
	KV                *metal.KVSnapshot
	KVBlocks          []metal.KVSnapshotBlock
	CaptureErr        error
	RestoredKV        *metal.KVSnapshot
	RestoredBlocks    []metal.KVSnapshotBlock
	RestoreErr        error
	RestoreBlocksErr  error
	Forked            metal.SessionHandle
	ForkErr           error
	ErrValue          error
	ResetCalls        int
	CloseCalls        int
	CloseErr          error
}

// Prefill records the prompt.
func (s *Handle) Prefill(_ context.Context, prompt string) error {
	s.PrefillPrompt = prompt
	return s.PrefillErr
}

// PrefillChunks records the chunk sequence.
func (s *Handle) PrefillChunks(_ context.Context, chunks iter.Seq[string]) error {
	s.PrefillChunksSeen = collectChunks(chunks)
	return s.PrefillErr
}

// PrefillTokens records the token IDs.
func (s *Handle) PrefillTokens(_ context.Context, tokens []int32) error {
	s.PrefillTokensSeen = append([]int32(nil), tokens...)
	return s.PrefillErr
}

// AppendPrompt records the appended prompt.
func (s *Handle) AppendPrompt(_ context.Context, prompt string) error {
	s.AppendPromptSeen = prompt
	return s.AppendErr
}

// AppendPromptChunks records the appended chunk sequence.
func (s *Handle) AppendPromptChunks(_ context.Context, chunks iter.Seq[string]) error {
	s.AppendChunksSeen = collectChunks(chunks)
	return s.AppendErr
}

// AppendTokens records the appended token IDs.
func (s *Handle) AppendTokens(_ context.Context, tokens []int32) error {
	s.AppendTokensSeen = append([]int32(nil), tokens...)
	return s.AppendErr
}

func collectChunks(chunks iter.Seq[string]) []string {
	out := []string{}
	if chunks == nil {
		return out
	}
	for chunk := range chunks {
		out = append(out, chunk)
	}
	return out
}

// Generate replays the seeded ProbeEvents then yields the seeded Tokens.
func (s *Handle) Generate(_ context.Context, cfg metal.GenerateConfig) iter.Seq[metal.Token] {
	s.Cfg = cfg
	s.GenerateCalls++
	return func(yield func(metal.Token) bool) {
		defer func() {
			if s.AfterGenerate != nil {
				s.AfterGenerate(s)
			}
		}()
		for _, event := range s.ProbeEvents {
			if cfg.ProbeSink != nil {
				cfg.ProbeSink.EmitProbe(event)
			}
		}
		for _, tok := range s.Tokens {
			if !yield(tok) {
				return
			}
		}
	}
}

// CaptureKV returns the seeded snapshot.
func (s *Handle) CaptureKV(_ context.Context) (*metal.KVSnapshot, error) {
	return s.KV, s.CaptureErr
}

// RangeKVBlocks yields the seeded blocks, or the whole KV as one block.
func (s *Handle) RangeKVBlocks(_ context.Context, _ int, _ metal.KVSnapshotCaptureOptions, yield func(metal.KVSnapshotBlock) (bool, error)) error {
	if len(s.KVBlocks) == 0 && s.KV != nil {
		_, err := yield(metal.KVSnapshotBlock{Index: 0, TokenStart: 0, TokenCount: len(s.KV.Tokens), Snapshot: s.KV})
		return err
	}
	for _, block := range s.KVBlocks {
		ok, err := yield(block)
		if err != nil || !ok {
			return err
		}
	}
	return nil
}

// RestoreKV records the restored snapshot.
func (s *Handle) RestoreKV(_ context.Context, snapshot *metal.KVSnapshot) error {
	s.RestoredKV = snapshot
	return s.RestoreErr
}

// RestoreKVBlocks loads blocks from source up to the prefix boundary.
func (s *Handle) RestoreKVBlocks(ctx context.Context, source metal.KVSnapshotBlockSource) error {
	if s.RestoreBlocksErr != nil {
		return s.RestoreBlocksErr
	}
	for i := 0; i < source.BlockCount; i++ {
		block, err := source.Load(ctx, i)
		if err != nil {
			return err
		}
		s.RestoredBlocks = append(s.RestoredBlocks, block)
		if block.TokenStart+block.TokenCount >= source.PrefixTokens {
			break
		}
	}
	if len(s.RestoredBlocks) == 1 {
		s.RestoredKV = s.RestoredBlocks[0].Snapshot
	}
	return nil
}

// Fork returns the seeded fork handle.
func (s *Handle) Fork(_ context.Context) (metal.SessionHandle, error) {
	return s.Forked, s.ForkErr
}

// Reset counts the call.
func (s *Handle) Reset() {
	s.ResetCalls++
}

// Close counts the call.
func (s *Handle) Close() error {
	s.CloseCalls++
	return s.CloseErr
}

// Err returns the seeded error.
func (s *Handle) Err() error {
	return s.ErrValue
}

// TestKVSnapshot builds the canonical two-token gemma4 KV snapshot the
// session and root agent-memory tests sleep/wake against.
func TestKVSnapshot() *metal.KVSnapshot {
	return &metal.KVSnapshot{
		Version:       metal.KVSnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1, 2},
		Generated:     []int32{2},
		TokenOffset:   2,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        2,
		HeadDim:       2,
		NumQueryHeads: 8,
		LogitShape:    []int32{1, 1, 3},
		Logits:        []float32{0.1, 0.2, 0.7},
		Layers: []metal.KVLayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []metal.KVHeadSnapshot{{
				Key:        []float32{1, 0, 0, 1},
				KeyDType:   metal.DTypeFloat32,
				KeyBytes:   []byte{0, 0, 128, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 63},
				Value:      []float32{0, 1, 1, 0},
				ValueDType: metal.DTypeFloat32,
				ValueBytes: []byte{0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 0, 0},
			}},
		}},
	}
}
