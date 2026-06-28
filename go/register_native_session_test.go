// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mlx

import (
	"context"
	"encoding/binary"
	"math"
	"reflect"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/pkg/model"
	"dappco.re/go/mlx/pkg/native"
)

type nativeSessionTextTokenModel struct {
	sessions []*nativeSessionTextSession
	opens    int
}

func (m *nativeSessionTextTokenModel) Embed(id int32) ([]byte, error) { return []byte{byte(id)}, nil }

func (m *nativeSessionTextTokenModel) DecodeForward([][]byte) ([][]byte, error) {
	return [][]byte{{0}}, nil
}

func (m *nativeSessionTextTokenModel) Head([]byte) ([]byte, error) {
	return nativeTextF32ToBF16([]float32{0.1, 0.2, 0.3, 0.4}), nil
}

func (m *nativeSessionTextTokenModel) Vocab() int { return 4 }

func (m *nativeSessionTextTokenModel) OpenSession() (model.DecodeStepper, error) {
	if m.opens >= len(m.sessions) {
		s := newNativeSessionTextSession()
		m.sessions = append(m.sessions, s)
	}
	sess := m.sessions[m.opens]
	m.opens++
	return sess, nil
}

type nativeSessionTextSession struct {
	tokens                    []int32
	logits                    []byte
	pos                       int
	restored                  native.SessionStateBlockSource
	restoredBlocks            []native.SessionStateBlock
	generateFromLogitsCalls   int
	generateSampledCalls      int
	generateSampledCacheCalls int
	generateFromCacheCalls    int
	closeCalls                int
}

func newNativeSessionTextSession() *nativeSessionTextSession {
	return &nativeSessionTextSession{logits: nativeTextF32ToBF16([]float32{0.1, 0.2, 0.3, 0.4})}
}

type retainedBoundaryOnlyTokenModel struct {
	sessions []*retainedBoundaryOnlySession
	opens    int
}

func (m *retainedBoundaryOnlyTokenModel) Embed(id int32) ([]byte, error) {
	return []byte{byte(id)}, nil
}

func (m *retainedBoundaryOnlyTokenModel) DecodeForward([][]byte) ([][]byte, error) {
	return [][]byte{{0}}, nil
}

func (m *retainedBoundaryOnlyTokenModel) Head([]byte) ([]byte, error) {
	return nativeTextF32ToBF16([]float32{0.1, 0.2, 0.3, 0.4}), nil
}

func (m *retainedBoundaryOnlyTokenModel) Vocab() int { return 4 }

func (m *retainedBoundaryOnlyTokenModel) OpenSession() (model.DecodeStepper, error) {
	if m.opens >= len(m.sessions) {
		s := newRetainedBoundaryOnlySession()
		m.sessions = append(m.sessions, s)
	}
	sess := m.sessions[m.opens]
	m.opens++
	return sess, nil
}

type retainedBoundaryOnlySession struct {
	tokens                    []int32
	logits                    []byte
	pos                       int
	restored                  native.SessionStateBlockSource
	generateSampledCacheCalls int
	generateFromCacheCalls    int
	closeCalls                int
}

func newRetainedBoundaryOnlySession() *retainedBoundaryOnlySession {
	return &retainedBoundaryOnlySession{logits: nativeTextF32ToBF16([]float32{0.1, 0.2, 0.3, 0.4})}
}

func (s *retainedBoundaryOnlySession) Step([]byte) ([]byte, error) { return []byte{0}, nil }

func (s *retainedBoundaryOnlySession) PrefillTokens(tokens []int32) error {
	s.tokens = append(s.tokens[:0], tokens...)
	s.pos = len(tokens)
	return nil
}

func (s *retainedBoundaryOnlySession) AppendTokens(tokens []int32) error {
	s.tokens = append(s.tokens, tokens...)
	s.pos = len(s.tokens)
	return nil
}

func (s *retainedBoundaryOnlySession) BoundaryLogits() ([]byte, error) {
	return append([]byte(nil), s.logits...), nil
}

func (s *retainedBoundaryOnlySession) GenerateFromCacheEach(maxNew, _ int, yield func(int32) bool) ([]int32, error) {
	s.generateFromCacheCalls++
	return s.generate(maxNew, yield), nil
}

func (s *retainedBoundaryOnlySession) GenerateSampledFromCacheEach(maxNew int, _ []int32, _ *model.Sampler, _ model.SampleParams, _ model.TokenTransform, yield func(int32) bool) ([]int32, error) {
	s.generateSampledCacheCalls++
	return s.generate(maxNew, yield), nil
}

func (s *retainedBoundaryOnlySession) generate(maxNew int, yield func(int32) bool) []int32 {
	if maxNew <= 0 {
		return nil
	}
	out := []int32{3}
	kept := out[:0]
	for _, id := range out {
		kept = append(kept, id)
		if yield != nil && !yield(id) {
			break
		}
	}
	s.tokens = append(s.tokens, kept...)
	s.pos = len(s.tokens)
	return append([]int32(nil), kept...)
}

func (s *retainedBoundaryOnlySession) StateBlockSource(blockSize int) (native.SessionStateBlockSource, error) {
	return s.StateBlockSourceFrom(0, blockSize)
}

func (s *retainedBoundaryOnlySession) StateBlockSourceFrom(startToken, blockSize int) (native.SessionStateBlockSource, error) {
	totalBlocks := (s.pos + blockSize - 1) / blockSize
	firstBlock := 0
	for firstBlock < totalBlocks {
		end := (firstBlock + 1) * blockSize
		if end > s.pos {
			end = s.pos
		}
		if end > startToken {
			break
		}
		firstBlock++
	}
	source := native.SessionStateBlockSource{
		Position:           s.pos,
		CachedIDs:          append([]int32(nil), s.tokens...),
		CachedPromptIDs:    append([]int32(nil), s.tokens...),
		CachedPromptLogits: append([]byte(nil), s.logits...),
		RetainedLogits:     append([]byte(nil), s.logits...),
		BlockCount:         totalBlocks - firstBlock,
	}
	source.Load = func(index int) (native.SessionStateBlock, error) {
		blockIndex := firstBlock + index
		start := blockIndex * blockSize
		end := start + blockSize
		if end > s.pos {
			end = s.pos
		}
		keyBytes := nativeSessionTextKVBytes(s.tokens[start:end], 0x10)
		valueBytes := nativeSessionTextKVBytes(s.tokens[start:end], 0x20)
		return native.SessionStateBlock{
			Index:      blockIndex,
			TokenStart: start,
			TokenCount: end - start,
			Layers: []native.SessionStateLayerBlock{{
				Layer:      0,
				KVHeads:    1,
				HeadDim:    2,
				RowBytes:   4,
				KeyBytes:   keyBytes,
				ValueBytes: valueBytes,
			}},
		}, nil
	}
	return source, nil
}

func (s *retainedBoundaryOnlySession) RestoreStateBlocks(source native.SessionStateBlockSource) error {
	s.restored = source
	s.tokens = append(s.tokens[:0], source.CachedIDs...)
	if len(source.RetainedLogits) > 0 {
		s.logits = append(s.logits[:0], source.RetainedLogits...)
	} else {
		s.logits = append(s.logits[:0], source.CachedPromptLogits...)
	}
	s.pos = source.Position
	for i := 0; i < source.BlockCount; i++ {
		if _, err := source.Load(i); err != nil {
			return err
		}
	}
	return nil
}

func (s *retainedBoundaryOnlySession) Pos() int { return s.pos }

func (s *retainedBoundaryOnlySession) Close() error {
	s.closeCalls++
	return nil
}

func (s *nativeSessionTextSession) Step([]byte) ([]byte, error) { return []byte{0}, nil }

func (s *nativeSessionTextSession) PrefillTokens(tokens []int32) error {
	s.tokens = append(s.tokens[:0], tokens...)
	s.pos = len(tokens)
	return nil
}

func (s *nativeSessionTextSession) AppendTokens(tokens []int32) error {
	s.tokens = append(s.tokens, tokens...)
	s.pos = len(s.tokens)
	return nil
}

func (s *nativeSessionTextSession) BoundaryLogits() ([]byte, error) {
	return append([]byte(nil), s.logits...), nil
}

func (s *nativeSessionTextSession) GenerateFromCacheEach(maxNew, _ int, yield func(int32) bool) ([]int32, error) {
	s.generateFromCacheCalls++
	return s.generate(maxNew, yield), nil
}

func (s *nativeSessionTextSession) GenerateFromCacheLogitsEach(_ []byte, maxNew, _ int, yield func(int32) bool) ([]int32, error) {
	s.generateFromLogitsCalls++
	return s.generate(maxNew, yield), nil
}

func (s *nativeSessionTextSession) GenerateSampledFromCacheLogitsEach(_ []byte, maxNew int, _ []int32, _ *model.Sampler, _ model.SampleParams, _ model.TokenTransform, yield func(int32) bool) ([]int32, error) {
	s.generateSampledCalls++
	return s.generate(maxNew, yield), nil
}

func (s *nativeSessionTextSession) GenerateSampledFromCacheEach(maxNew int, _ []int32, _ *model.Sampler, _ model.SampleParams, _ model.TokenTransform, yield func(int32) bool) ([]int32, error) {
	s.generateSampledCacheCalls++
	return s.generate(maxNew, yield), nil
}

func (s *nativeSessionTextSession) generate(maxNew int, yield func(int32) bool) []int32 {
	if maxNew <= 0 {
		return nil
	}
	out := []int32{3}
	if maxNew > 1 {
		out = append(out, 2)
	}
	kept := out[:0]
	for _, id := range out {
		kept = append(kept, id)
		if yield != nil && !yield(id) {
			break
		}
	}
	s.tokens = append(s.tokens, kept...)
	s.pos = len(s.tokens)
	return append([]int32(nil), kept...)
}

func (s *nativeSessionTextSession) StateBlockSource(blockSize int) (native.SessionStateBlockSource, error) {
	return s.StateBlockSourceFrom(0, blockSize)
}

func (s *nativeSessionTextSession) StateBlockSourceFrom(startToken, blockSize int) (native.SessionStateBlockSource, error) {
	totalBlocks := (s.pos + blockSize - 1) / blockSize
	firstBlock := 0
	for firstBlock < totalBlocks {
		end := (firstBlock + 1) * blockSize
		if end > s.pos {
			end = s.pos
		}
		if end > startToken {
			break
		}
		firstBlock++
	}
	source := native.SessionStateBlockSource{
		Position:           s.pos,
		CachedIDs:          append([]int32(nil), s.tokens...),
		CachedPromptIDs:    append([]int32(nil), s.tokens...),
		CachedPromptLogits: append([]byte(nil), s.logits...),
		RetainedLogits:     append([]byte(nil), s.logits...),
		BlockCount:         totalBlocks - firstBlock,
	}
	source.Load = func(index int) (native.SessionStateBlock, error) {
		blockIndex := firstBlock + index
		start := blockIndex * blockSize
		end := start + blockSize
		if end > s.pos {
			end = s.pos
		}
		keyBytes := nativeSessionTextKVBytes(s.tokens[start:end], 0x10)
		valueBytes := nativeSessionTextKVBytes(s.tokens[start:end], 0x20)
		return native.SessionStateBlock{
			Index:      blockIndex,
			TokenStart: start,
			TokenCount: end - start,
			Layers: []native.SessionStateLayerBlock{{
				Layer:      0,
				KVHeads:    1,
				HeadDim:    2,
				RowBytes:   4,
				KeyBytes:   keyBytes,
				ValueBytes: valueBytes,
			}},
		}, nil
	}
	return source, nil
}

func (s *nativeSessionTextSession) RestoreStateBlocks(source native.SessionStateBlockSource) error {
	s.restored = source
	s.tokens = append(s.tokens[:0], source.CachedIDs...)
	s.logits = append(s.logits[:0], source.CachedPromptLogits...)
	s.pos = source.Position
	s.restoredBlocks = s.restoredBlocks[:0]
	for i := 0; i < source.BlockCount; i++ {
		block, err := source.Load(i)
		if err != nil {
			return err
		}
		s.restoredBlocks = append(s.restoredBlocks, block)
	}
	return nil
}

func (s *nativeSessionTextSession) Pos() int { return s.pos }

func (s *nativeSessionTextSession) Close() error {
	s.closeCalls++
	return nil
}

func nativeSessionTextKVBytes(tokens []int32, salt byte) []byte {
	out := make([]byte, len(tokens)*4)
	for i, id := range tokens {
		out[i*4] = byte(id)
		out[i*4+1] = salt
		out[i*4+2] = byte(i)
		out[i*4+3] = salt + 1
	}
	return out
}

func nativeTextF32RawBytes(values []float32) []byte {
	out := make([]byte, len(values)*4)
	for i, v := range values {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}

func testNativeTextSessionModel(sessions ...*nativeSessionTextSession) *nativeTextModel {
	return &nativeTextModel{
		tm:        &nativeSessionTextTokenModel{sessions: sessions},
		modelType: "gemma4",
		info:      inference.ModelInfo{Architecture: "gemma4", VocabSize: 4, NumLayers: 1},
		maxLen:    16,
	}
}

func testRetainedBoundaryOnlyTextModel(sessions ...*retainedBoundaryOnlySession) *nativeTextModel {
	return &nativeTextModel{
		tm:        &retainedBoundaryOnlyTokenModel{sessions: sessions},
		modelType: "gemma4",
		info:      inference.ModelInfo{Architecture: "gemma4", VocabSize: 4, NumLayers: 1},
		maxLen:    16,
	}
}

func TestNativeTextModelNewSessionAcceptsRetainedBoundaryOnlySession_Good(t *testing.T) {
	ctx := context.Background()
	session := newRetainedBoundaryOnlySession()
	nativeModel := testRetainedBoundaryOnlyTextModel(session)
	handle := nativeModel.NewSession()
	if handle == nil {
		t.Fatal("NewSession() = nil, want retained-boundary-only native session handle")
	}
	if err := handle.(interface {
		RestoreKVBlocks(context.Context, metal.KVSnapshotBlockSource) error
	}).RestoreKVBlocks(ctx, nativeSessionTextBlockSource()); err != nil {
		t.Fatalf("RestoreKVBlocks: %v", err)
	}
	wantLogits := nativeTextF32ToBF16([]float32{0.1, 0.2, 0.3, 0.4})
	if !reflect.DeepEqual(session.restored.RetainedLogits, wantLogits) {
		t.Fatalf("restored retained logits = %v, want %v", session.restored.RetainedLogits, wantLogits)
	}
	var greedy []int32
	for tok := range handle.Generate(ctx, metal.GenerateConfig{MaxTokens: 1}) {
		greedy = append(greedy, tok.ID)
	}
	if err := handle.Err(); err != nil {
		t.Fatalf("greedy Generate Err: %v", err)
	}
	if !reflect.DeepEqual(greedy, []int32{3}) {
		t.Fatalf("greedy generated = %v, want [3]", greedy)
	}
	if session.generateFromCacheCalls != 1 || session.generateSampledCacheCalls != 0 {
		t.Fatalf("greedy calls cache/sampled = %d/%d, want 1/0", session.generateFromCacheCalls, session.generateSampledCacheCalls)
	}

	sampledSession := newRetainedBoundaryOnlySession()
	sampledModel := testRetainedBoundaryOnlyTextModel(sampledSession)
	sampledHandle := sampledModel.NewSession()
	if sampledHandle == nil {
		t.Fatal("sampled NewSession() = nil, want retained-boundary-only native session handle")
	}
	if err := sampledHandle.(interface {
		RestoreKVBlocks(context.Context, metal.KVSnapshotBlockSource) error
	}).RestoreKVBlocks(ctx, nativeSessionTextBlockSource()); err != nil {
		t.Fatalf("sampled RestoreKVBlocks: %v", err)
	}
	var sampled []int32
	for tok := range sampledHandle.Generate(ctx, metal.GenerateConfig{
		MaxTokens:           1,
		Temperature:         0.8,
		TopK:                2,
		Seed:                7,
		SeedSet:             true,
		SuppressTokens:      []int32{1},
		MinTokensBeforeStop: 1,
	}) {
		sampled = append(sampled, tok.ID)
	}
	if err := sampledHandle.Err(); err != nil {
		t.Fatalf("sampled Generate Err: %v", err)
	}
	if !reflect.DeepEqual(sampled, []int32{3}) {
		t.Fatalf("sampled generated = %v, want [3]", sampled)
	}
	if sampledSession.generateSampledCacheCalls != 1 || sampledSession.generateFromCacheCalls != 0 {
		t.Fatalf("sampled calls sampled/cache = %d/%d, want 1/0", sampledSession.generateSampledCacheCalls, sampledSession.generateFromCacheCalls)
	}
}

func TestNativeTextModelNewSession_CaptureRangeRestore_Good(t *testing.T) {
	ctx := context.Background()
	first := newNativeSessionTextSession()
	second := newNativeSessionTextSession()
	model := testNativeTextSessionModel(first, second)
	handle := model.NewSession()
	if handle == nil {
		t.Fatal("NewSession() = nil, want native session handle")
	}
	prefiller, ok := handle.(interface {
		PrefillTokens(context.Context, []int32) error
	})
	if !ok {
		t.Fatal("native session handle does not expose token prefill")
	}
	if err := prefiller.PrefillTokens(ctx, []int32{1, 2, 3}); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	snapshot, err := handle.(interface {
		CaptureKVWithOptions(context.Context, metal.KVSnapshotCaptureOptions) (*metal.KVSnapshot, error)
	}).CaptureKVWithOptions(ctx, metal.KVSnapshotCaptureOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("CaptureKVWithOptions: %v", err)
	}
	if snapshot.SeqLen != 3 || snapshot.TokenOffset != 3 || !reflect.DeepEqual(snapshot.Tokens, []int32{1, 2, 3}) {
		t.Fatalf("snapshot timeline = seq %d offset %d tokens %v", snapshot.SeqLen, snapshot.TokenOffset, snapshot.Tokens)
	}
	if len(snapshot.Layers) != 1 || len(snapshot.Layers[0].KeyBytes) != 12 || snapshot.Layers[0].KeyDType != metal.DTypeBFloat16 {
		t.Fatalf("snapshot layer = %+v", snapshot.Layers)
	}
	var ranged []metal.KVSnapshotBlock
	if err := handle.RangeKVBlocks(ctx, 2, metal.KVSnapshotCaptureOptions{BlockStartToken: 2}, func(block metal.KVSnapshotBlock) (bool, error) {
		ranged = append(ranged, block)
		return true, nil
	}); err != nil {
		t.Fatalf("RangeKVBlocks: %v", err)
	}
	if len(ranged) != 1 || ranged[0].Index != 1 || ranged[0].TokenStart != 2 || ranged[0].TokenCount != 1 {
		t.Fatalf("ranged blocks = %+v, want only suffix block", ranged)
	}
	restored := model.NewSession()
	if restored == nil {
		t.Fatal("second NewSession() = nil")
	}
	restorer := restored.(interface {
		RestoreKV(context.Context, *metal.KVSnapshot) error
	})
	if err := restorer.RestoreKV(ctx, snapshot); err != nil {
		t.Fatalf("RestoreKV: %v", err)
	}
	if !reflect.DeepEqual(second.tokens, []int32{1, 2, 3}) {
		t.Fatalf("restored tokens = %v", second.tokens)
	}
	wantLogits := nativeTextF32ToBF16([]float32{0.1, 0.2, 0.3, 0.4})
	if !reflect.DeepEqual(second.restored.RetainedLogits, wantLogits) {
		t.Fatalf("snapshot restored retained logits = %v, want %v", second.restored.RetainedLogits, wantLogits)
	}
	if len(second.restoredBlocks) != 1 || second.restoredBlocks[0].TokenCount != 3 {
		t.Fatalf("restored blocks = %+v", second.restoredBlocks)
	}
}

func TestNativeTextSession_RestoreKVConvertsHeadSnapshots_Good(t *testing.T) {
	ctx := context.Background()
	session := newNativeSessionTextSession()
	model := testNativeTextSessionModel(session)
	handle := model.NewSession()
	snapshot := &metal.KVSnapshot{
		Version:       metal.KVSnapshotVersion,
		Architecture:  "gemma4",
		Tokens:        []int32{1, 2},
		TokenOffset:   2,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        2,
		HeadDim:       2,
		NumQueryHeads: 1,
		Layers: []metal.KVLayerSnapshot{{
			Layer:      0,
			CacheIndex: 4,
			CacheMode:  metal.KVCacheModeFixed,
			MaxSize:    8,
			Heads: []metal.KVHeadSnapshot{{
				Key:   []float32{1, 2, 3, 4},
				Value: []float32{5, 6, 7, 8},
			}},
		}},
	}
	if err := handle.(interface {
		RestoreKV(context.Context, *metal.KVSnapshot) error
	}).RestoreKV(ctx, snapshot); err != nil {
		t.Fatalf("RestoreKV head snapshot: %v", err)
	}
	if len(session.restoredBlocks) != 1 || len(session.restoredBlocks[0].Layers) != 1 {
		t.Fatalf("restored blocks = %+v, want one converted head layer", session.restoredBlocks)
	}
	layer := session.restoredBlocks[0].Layers[0]
	if layer.CacheIndex != 4 || layer.CacheMode != string(metal.KVCacheModeFixed) || layer.MaxSize != 8 {
		t.Fatalf("converted layer metadata = %d/%q/%d, want 4/fixed/8", layer.CacheIndex, layer.CacheMode, layer.MaxSize)
	}
	if layer.KVHeads != 1 || layer.HeadDim != 2 || layer.RowBytes != 4 {
		t.Fatalf("converted layer geometry = heads %d dim %d row %d, want 1/2/4", layer.KVHeads, layer.HeadDim, layer.RowBytes)
	}
	if want := nativeTextF32ToBF16([]float32{1, 2, 3, 4}); !reflect.DeepEqual(layer.KeyBytes, want) {
		t.Fatalf("converted key bytes = %v, want %v", layer.KeyBytes, want)
	}
	if want := nativeTextF32ToBF16([]float32{5, 6, 7, 8}); !reflect.DeepEqual(layer.ValueBytes, want) {
		t.Fatalf("converted value bytes = %v, want %v", layer.ValueBytes, want)
	}
}

func TestNativeTextSession_RestoreKVConvertsRawHeadSnapshots_Good(t *testing.T) {
	ctx := context.Background()
	session := newNativeSessionTextSession()
	model := testNativeTextSessionModel(session)
	handle := model.NewSession()
	head0Key := nativeTextF32ToBF16([]float32{1, 2, 5, 6})
	head1Key := nativeTextF32ToBF16([]float32{3, 4, 7, 8})
	head0Value := nativeTextF32ToBF16([]float32{11, 12, 15, 16})
	head1Value := nativeTextF32ToBF16([]float32{13, 14, 17, 18})
	snapshot := &metal.KVSnapshot{
		Version:       metal.KVSnapshotVersion,
		Architecture:  "gemma4",
		Tokens:        []int32{1, 2},
		TokenOffset:   2,
		NumLayers:     1,
		NumHeads:      2,
		SeqLen:        2,
		HeadDim:       2,
		NumQueryHeads: 2,
		Layers: []metal.KVLayerSnapshot{{
			Layer:      0,
			CacheIndex: 1,
			CacheMode:  metal.KVCacheModeFixed,
			Heads: []metal.KVHeadSnapshot{
				{KeyDType: metal.DTypeBFloat16, KeyBytes: head0Key, ValueDType: metal.DTypeBFloat16, ValueBytes: head0Value},
				{KeyDType: metal.DTypeBFloat16, KeyBytes: head1Key, ValueDType: metal.DTypeBFloat16, ValueBytes: head1Value},
			},
		}},
	}
	if err := handle.(interface {
		RestoreKV(context.Context, *metal.KVSnapshot) error
	}).RestoreKV(ctx, snapshot); err != nil {
		t.Fatalf("RestoreKV raw head snapshot: %v", err)
	}
	if len(session.restoredBlocks) != 1 || len(session.restoredBlocks[0].Layers) != 1 {
		t.Fatalf("restored blocks = %+v, want one converted raw head layer", session.restoredBlocks)
	}
	layer := session.restoredBlocks[0].Layers[0]
	if layer.KVHeads != 2 || layer.HeadDim != 2 || layer.RowBytes != 8 {
		t.Fatalf("converted raw layer geometry = heads %d dim %d row %d, want 2/2/8", layer.KVHeads, layer.HeadDim, layer.RowBytes)
	}
	wantKey := nativeTextF32ToBF16([]float32{1, 2, 3, 4, 5, 6, 7, 8})
	if !reflect.DeepEqual(layer.KeyBytes, wantKey) {
		t.Fatalf("converted raw key bytes = %v, want token-major rows %v", layer.KeyBytes, wantKey)
	}
	wantValue := nativeTextF32ToBF16([]float32{11, 12, 13, 14, 15, 16, 17, 18})
	if !reflect.DeepEqual(layer.ValueBytes, wantValue) {
		t.Fatalf("converted raw value bytes = %v, want token-major rows %v", layer.ValueBytes, wantValue)
	}
}

func TestNativeTextSession_RestoreKVConvertsRawLayerSlabSnapshots_Good(t *testing.T) {
	ctx := context.Background()
	session := newNativeSessionTextSession()
	model := testNativeTextSessionModel(session)
	handle := model.NewSession()
	headMajorKey := nativeTextF32ToBF16([]float32{1, 2, 5, 6, 3, 4, 7, 8})
	headMajorValue := nativeTextF32ToBF16([]float32{11, 12, 15, 16, 13, 14, 17, 18})
	snapshot := &metal.KVSnapshot{
		Version:       metal.KVSnapshotVersion,
		Architecture:  "gemma4",
		Tokens:        []int32{1, 2},
		TokenOffset:   2,
		NumLayers:     1,
		NumHeads:      2,
		SeqLen:        2,
		HeadDim:       2,
		NumQueryHeads: 2,
		Layers: []metal.KVLayerSnapshot{{
			Layer:      0,
			CacheIndex: 3,
			CacheMode:  metal.KVCacheModeFixed,
			MaxSize:    8,
			KeyDType:   metal.DTypeBFloat16,
			KeyBytes:   headMajorKey,
			KeyShape:   []int32{1, 2, 2, 2},
			ValueDType: metal.DTypeBFloat16,
			ValueBytes: headMajorValue,
			ValueShape: []int32{1, 2, 2, 2},
		}},
	}
	if err := handle.(interface {
		RestoreKV(context.Context, *metal.KVSnapshot) error
	}).RestoreKV(ctx, snapshot); err != nil {
		t.Fatalf("RestoreKV raw layer slab: %v", err)
	}
	if len(session.restoredBlocks) != 1 || len(session.restoredBlocks[0].Layers) != 1 {
		t.Fatalf("restored blocks = %+v, want one converted raw layer slab", session.restoredBlocks)
	}
	layer := session.restoredBlocks[0].Layers[0]
	if layer.KVHeads != 2 || layer.HeadDim != 2 || layer.RowBytes != 8 {
		t.Fatalf("converted raw slab geometry = heads %d dim %d row %d, want 2/2/8", layer.KVHeads, layer.HeadDim, layer.RowBytes)
	}
	wantKey := nativeTextF32ToBF16([]float32{1, 2, 3, 4, 5, 6, 7, 8})
	if !reflect.DeepEqual(layer.KeyBytes, wantKey) {
		t.Fatalf("converted raw slab key bytes = %v, want token-major rows %v", layer.KeyBytes, wantKey)
	}
	wantValue := nativeTextF32ToBF16([]float32{11, 12, 13, 14, 15, 16, 17, 18})
	if !reflect.DeepEqual(layer.ValueBytes, wantValue) {
		t.Fatalf("converted raw slab value bytes = %v, want token-major rows %v", layer.ValueBytes, wantValue)
	}
}

func TestNativeTextSession_RestoreKVConvertsFloat32LayerSlabSnapshots_Good(t *testing.T) {
	ctx := context.Background()
	session := newNativeSessionTextSession()
	model := testNativeTextSessionModel(session)
	handle := model.NewSession()
	headMajorKey := nativeTextF32RawBytes([]float32{1, 2, 5, 6, 3, 4, 7, 8})
	headMajorValue := nativeTextF32RawBytes([]float32{11, 12, 15, 16, 13, 14, 17, 18})
	snapshot := &metal.KVSnapshot{
		Version:       metal.KVSnapshotVersion,
		Architecture:  "gemma4",
		Tokens:        []int32{1, 2},
		TokenOffset:   2,
		NumLayers:     1,
		NumHeads:      2,
		SeqLen:        2,
		HeadDim:       2,
		NumQueryHeads: 2,
		Layers: []metal.KVLayerSnapshot{{
			Layer:      0,
			CacheIndex: 3,
			CacheMode:  metal.KVCacheModePaged,
			MaxSize:    8,
			KeyDType:   metal.DTypeFloat32,
			KeyBytes:   headMajorKey,
			KeyShape:   []int32{1, 2, 2, 2},
			ValueDType: metal.DTypeFloat32,
			ValueBytes: headMajorValue,
			ValueShape: []int32{1, 2, 2, 2},
		}},
	}
	if err := handle.(interface {
		RestoreKV(context.Context, *metal.KVSnapshot) error
	}).RestoreKV(ctx, snapshot); err != nil {
		t.Fatalf("RestoreKV float32 raw layer slab: %v", err)
	}
	if len(session.restoredBlocks) != 1 || len(session.restoredBlocks[0].Layers) != 1 {
		t.Fatalf("restored blocks = %+v, want one converted float32 raw layer slab", session.restoredBlocks)
	}
	layer := session.restoredBlocks[0].Layers[0]
	if layer.CacheIndex != 3 || layer.CacheMode != string(metal.KVCacheModePaged) || layer.MaxSize != 8 {
		t.Fatalf("converted float32 slab metadata = %d/%q/%d, want 3/paged/8", layer.CacheIndex, layer.CacheMode, layer.MaxSize)
	}
	if layer.KVHeads != 2 || layer.HeadDim != 2 || layer.RowBytes != 8 {
		t.Fatalf("converted float32 slab geometry = heads %d dim %d row %d, want 2/2/8", layer.KVHeads, layer.HeadDim, layer.RowBytes)
	}
	wantKey := nativeTextF32ToBF16([]float32{1, 2, 3, 4, 5, 6, 7, 8})
	if !reflect.DeepEqual(layer.KeyBytes, wantKey) {
		t.Fatalf("converted float32 slab key bytes = %v, want token-major rows %v", layer.KeyBytes, wantKey)
	}
	wantValue := nativeTextF32ToBF16([]float32{11, 12, 13, 14, 15, 16, 17, 18})
	if !reflect.DeepEqual(layer.ValueBytes, wantValue) {
		t.Fatalf("converted float32 slab value bytes = %v, want token-major rows %v", layer.ValueBytes, wantValue)
	}
}

func TestNativeTextSession_RestoreKVPreservesSlidingTailTokenOffset_Good(t *testing.T) {
	ctx := context.Background()
	session := newNativeSessionTextSession()
	model := testNativeTextSessionModel(session)
	handle := model.NewSession()
	snapshot := &metal.KVSnapshot{
		Version:       metal.KVSnapshotVersion,
		Architecture:  "gemma4",
		Tokens:        []int32{5, 6},
		TokenOffset:   6,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        2,
		HeadDim:       2,
		NumQueryHeads: 1,
		Layers: []metal.KVLayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			CacheMode:  metal.KVCacheModeFixed,
			MaxSize:    2,
			KeyDType:   metal.DTypeBFloat16,
			KeyBytes:   nativeTextF32ToBF16([]float32{5, 6, 7, 8}),
			KeyShape:   []int32{1, 1, 2, 2},
			ValueDType: metal.DTypeBFloat16,
			ValueBytes: nativeTextF32ToBF16([]float32{15, 16, 17, 18}),
			ValueShape: []int32{1, 1, 2, 2},
		}},
	}
	if err := handle.(interface {
		RestoreKV(context.Context, *metal.KVSnapshot) error
	}).RestoreKV(ctx, snapshot); err != nil {
		t.Fatalf("RestoreKV sliding tail: %v", err)
	}
	if session.pos != snapshot.TokenOffset {
		t.Fatalf("restored position = %d, want token offset %d", session.pos, snapshot.TokenOffset)
	}
	if len(session.restoredBlocks) != 2 {
		t.Fatalf("restored blocks = %+v, want expired prefix + live tail", session.restoredBlocks)
	}
	prefix, tail := session.restoredBlocks[0], session.restoredBlocks[1]
	if prefix.TokenStart != 0 || prefix.TokenCount != 4 || len(prefix.Layers) != 1 {
		t.Fatalf("prefix block = start %d count %d layers %d, want 0/4/1", prefix.TokenStart, prefix.TokenCount, len(prefix.Layers))
	}
	if len(prefix.Layers[0].KeyBytes) != 0 || len(prefix.Layers[0].ValueBytes) != 0 {
		t.Fatalf("expired prefix carried KV bytes key=%v value=%v", prefix.Layers[0].KeyBytes, prefix.Layers[0].ValueBytes)
	}
	if tail.TokenStart != 4 || tail.TokenCount != 2 || len(tail.Layers) != 1 {
		t.Fatalf("tail block = start %d count %d layers %d, want 4/2/1", tail.TokenStart, tail.TokenCount, len(tail.Layers))
	}
	if want := nativeTextF32ToBF16([]float32{5, 6, 7, 8}); !reflect.DeepEqual(tail.Layers[0].KeyBytes, want) {
		t.Fatalf("tail key bytes = %v, want %v", tail.Layers[0].KeyBytes, want)
	}
	if want := nativeTextF32ToBF16([]float32{15, 16, 17, 18}); !reflect.DeepEqual(tail.Layers[0].ValueBytes, want) {
		t.Fatalf("tail value bytes = %v, want %v", tail.Layers[0].ValueBytes, want)
	}
}

func TestNativeTextSession_RestoreKVBlocksConvertsFloat32LayerSlabSnapshots_Good(t *testing.T) {
	ctx := context.Background()
	session := newNativeSessionTextSession()
	model := testNativeTextSessionModel(session)
	handle := model.NewSession()
	block := nativeSessionTextMetalBlock(0, 0, []int32{1, 2}, true)
	block.Snapshot.Layers[0].CacheMode = metal.KVCacheModePaged
	block.Snapshot.Layers[0].MaxSize = 8
	block.Snapshot.Layers[0].KeyDType = metal.DTypeFloat32
	block.Snapshot.Layers[0].KeyBytes = nativeTextF32RawBytes([]float32{1, 2, 3, 4})
	block.Snapshot.Layers[0].ValueDType = metal.DTypeFloat32
	block.Snapshot.Layers[0].ValueBytes = nativeTextF32RawBytes([]float32{11, 12, 13, 14})
	source := metal.KVSnapshotBlockSource{
		TokenCount:   2,
		PrefixTokens: 2,
		BlockCount:   1,
		Load: func(_ context.Context, index int) (metal.KVSnapshotBlock, error) {
			if index != 0 {
				return metal.KVSnapshotBlock{}, core.NewError("test: block index out of range")
			}
			return block, nil
		},
	}
	if err := handle.(interface {
		RestoreKVBlocks(context.Context, metal.KVSnapshotBlockSource) error
	}).RestoreKVBlocks(ctx, source); err != nil {
		t.Fatalf("RestoreKVBlocks float32 raw layer slab: %v", err)
	}
	if len(session.restoredBlocks) != 1 || len(session.restoredBlocks[0].Layers) != 1 {
		t.Fatalf("restored blocks = %+v, want one converted float32 block layer", session.restoredBlocks)
	}
	layer := session.restoredBlocks[0].Layers[0]
	if layer.CacheMode != string(metal.KVCacheModePaged) || layer.MaxSize != 8 {
		t.Fatalf("restored float32 block metadata = %q/%d, want paged/8", layer.CacheMode, layer.MaxSize)
	}
	if want := nativeTextF32ToBF16([]float32{1, 2, 3, 4}); !reflect.DeepEqual(layer.KeyBytes, want) {
		t.Fatalf("restored float32 block key bytes = %v, want %v", layer.KeyBytes, want)
	}
	if want := nativeTextF32ToBF16([]float32{11, 12, 13, 14}); !reflect.DeepEqual(layer.ValueBytes, want) {
		t.Fatalf("restored float32 block value bytes = %v, want %v", layer.ValueBytes, want)
	}
}

func TestNativeTextSession_RestoreKVBlocksGenerateUsesRetainedBoundaryMetadata_Good(t *testing.T) {
	ctx := context.Background()
	session := newNativeSessionTextSession()
	model := testNativeTextSessionModel(session)
	handle := model.NewSession()
	source := metal.KVSnapshotBlockSource{
		TokenCount:   3,
		PrefixTokens: 3,
		BlockCount:   2,
		Load: func(_ context.Context, index int) (metal.KVSnapshotBlock, error) {
			switch index {
			case 0:
				return nativeSessionTextMetalBlock(0, 0, []int32{1, 2}, false), nil
			case 1:
				return nativeSessionTextMetalBlock(1, 2, []int32{3}, true), nil
			default:
				return metal.KVSnapshotBlock{}, nil
			}
		},
	}
	if err := handle.(interface {
		RestoreKVBlocks(context.Context, metal.KVSnapshotBlockSource) error
	}).RestoreKVBlocks(ctx, source); err != nil {
		t.Fatalf("RestoreKVBlocks: %v", err)
	}
	if !reflect.DeepEqual(session.tokens, []int32{1, 2, 3}) {
		t.Fatalf("tokens after RestoreKVBlocks = %v", session.tokens)
	}
	wantLogits := nativeTextF32ToBF16([]float32{0.1, 0.2, 0.3, 0.4})
	if !reflect.DeepEqual(session.restored.RetainedLogits, wantLogits) {
		t.Fatalf("restored retained logits = %v, want %v", session.restored.RetainedLogits, wantLogits)
	}
	var generated []int32
	for tok := range handle.Generate(ctx, metal.GenerateConfig{MaxTokens: 1}) {
		generated = append(generated, tok.ID)
	}
	if err := handle.Err(); err != nil {
		t.Fatalf("Generate Err: %v", err)
	}
	if !reflect.DeepEqual(generated, []int32{3}) {
		t.Fatalf("generated = %v, want [3]", generated)
	}
	if session.generateFromCacheCalls != 1 {
		t.Fatalf("GenerateFromCacheEach calls = %d, want 1", session.generateFromCacheCalls)
	}
	if session.generateFromLogitsCalls != 0 || session.generateSampledCalls != 0 || session.generateSampledCacheCalls != 0 {
		t.Fatalf("other generate calls logits/sampledLogits/sampledCache = %d/%d/%d, want 0/0/0", session.generateFromLogitsCalls, session.generateSampledCalls, session.generateSampledCacheCalls)
	}
}

func TestNativeTextSession_RestoreKVBlocksGraftsResidentTrustedPrefix_Good(t *testing.T) {
	ctx := context.Background()
	session := newNativeSessionTextSession()
	model := testNativeTextSessionModel(session)
	handle := model.NewSession()
	prefiller := handle.(interface {
		PrefillTokens(context.Context, []int32) error
	})
	if err := prefiller.PrefillTokens(ctx, []int32{1, 2}); err != nil {
		t.Fatalf("PrefillTokens trusted prefix: %v", err)
	}
	source := metal.KVSnapshotBlockSource{
		TokenCount:   3,
		PrefixTokens: 3,
		BlockCount:   1,
		Load: func(_ context.Context, index int) (metal.KVSnapshotBlock, error) {
			if index != 0 {
				return metal.KVSnapshotBlock{}, nil
			}
			return nativeSessionTextMetalBlock(1, 2, []int32{3}, true), nil
		},
	}
	if err := handle.(interface {
		RestoreKVBlocks(context.Context, metal.KVSnapshotBlockSource) error
	}).RestoreKVBlocks(ctx, source); err != nil {
		t.Fatalf("RestoreKVBlocks suffix-only trusted prefix: %v", err)
	}
	if !reflect.DeepEqual(session.tokens, []int32{1, 2, 3}) {
		t.Fatalf("tokens after trusted-prefix RestoreKVBlocks = %v", session.tokens)
	}
	if !reflect.DeepEqual(session.restored.CachedIDs, []int32{1, 2, 3}) {
		t.Fatalf("restored cached ids = %v, want full trusted prefix plus suffix", session.restored.CachedIDs)
	}
	if len(session.restoredBlocks) != 1 || session.restoredBlocks[0].Index != 1 || session.restoredBlocks[0].TokenStart != 2 || session.restoredBlocks[0].TokenCount != 1 {
		t.Fatalf("restored suffix blocks = %+v, want only absolute block 1 at token 2", session.restoredBlocks)
	}
	var generated []int32
	for tok := range handle.Generate(ctx, metal.GenerateConfig{MaxTokens: 1}) {
		generated = append(generated, tok.ID)
	}
	if err := handle.Err(); err != nil {
		t.Fatalf("Generate Err: %v", err)
	}
	if !reflect.DeepEqual(generated, []int32{3}) {
		t.Fatalf("generated after trusted-prefix restore = %v, want [3]", generated)
	}
}

func TestNativeTextSession_RestoreKVBlocksKeepsNonUniformTrustedPrefix_Good(t *testing.T) {
	ctx := context.Background()
	source := metal.KVSnapshotBlockSource{
		TokenCount:   4,
		PrefixTokens: 4,
		BlockCount:   1,
		Load: func(_ context.Context, index int) (metal.KVSnapshotBlock, error) {
			if index != 0 {
				return metal.KVSnapshotBlock{}, nil
			}
			return nativeSessionTextMetalBlock(2, 3, []int32{4}, true), nil
		},
	}
	restored, tokens, _, err := nativeTextStateSourceFromBlockSource(ctx, source, []int32{1, 2, 3})
	if err != nil {
		t.Fatalf("nativeTextStateSourceFromBlockSource non-uniform prefix: %v", err)
	}
	if !reflect.DeepEqual(tokens, []int32{1, 2, 3, 4}) {
		t.Fatalf("tokens = %v, want trusted prefix plus suffix", tokens)
	}
	trustedPrefix := reflect.ValueOf(restored).FieldByName("trustedPrefix").Int()
	firstBlock := reflect.ValueOf(restored).FieldByName("firstBlockIndex").Int()
	if trustedPrefix != 3 || firstBlock != 2 {
		t.Fatalf("trusted prefix metadata = prefix %d first block %d, want 3/2", trustedPrefix, firstBlock)
	}
}

func TestNativeTextSession_RestoreKVBlocksCarriesCacheModeMetadata_Good(t *testing.T) {
	ctx := context.Background()
	session := newNativeSessionTextSession()
	model := testNativeTextSessionModel(session)
	handle := model.NewSession()
	block := nativeSessionTextMetalBlock(0, 0, []int32{1, 2}, true)
	block.Snapshot.Layers[0].CacheIndex = 3
	block.Snapshot.Layers[0].CacheMode = metal.KVCacheModePaged
	block.Snapshot.Layers[0].MaxSize = 64
	source := metal.KVSnapshotBlockSource{
		TokenCount:   2,
		PrefixTokens: 2,
		BlockCount:   1,
		Load: func(_ context.Context, index int) (metal.KVSnapshotBlock, error) {
			if index != 0 {
				return metal.KVSnapshotBlock{}, nil
			}
			return block, nil
		},
	}
	if err := handle.(interface {
		RestoreKVBlocks(context.Context, metal.KVSnapshotBlockSource) error
	}).RestoreKVBlocks(ctx, source); err != nil {
		t.Fatalf("RestoreKVBlocks paged metadata: %v", err)
	}
	if len(session.restoredBlocks) != 1 || len(session.restoredBlocks[0].Layers) != 1 {
		t.Fatalf("restored blocks = %+v, want one layer block", session.restoredBlocks)
	}
	layer := session.restoredBlocks[0].Layers[0]
	if layer.CacheIndex != 3 || layer.CacheMode != string(metal.KVCacheModePaged) || layer.MaxSize != 64 {
		t.Fatalf("restored layer cache metadata = %d/%q/%d, want 3/paged/64", layer.CacheIndex, layer.CacheMode, layer.MaxSize)
	}
}

func TestNativeTextSession_SnapshotFromNativeBlockCarriesCacheModeMetadata_Good(t *testing.T) {
	model := testNativeTextSessionModel(newNativeSessionTextSession())
	handle := model.NewSession()
	session := handle.(*nativeTextSession)
	source := native.SessionStateBlockSource{
		Position:  2,
		CachedIDs: []int32{1, 2},
	}
	block := native.SessionStateBlock{
		Index:      0,
		TokenStart: 0,
		TokenCount: 2,
		Layers: []native.SessionStateLayerBlock{{
			Layer:      0,
			CacheIndex: 3,
			CacheMode:  string(metal.KVCacheModePaged),
			MaxSize:    64,
			KVHeads:    1,
			HeadDim:    2,
			RowBytes:   4,
			KeyBytes:   nativeSessionTextKVBytes([]int32{1, 2}, 0x10),
			ValueBytes: nativeSessionTextKVBytes([]int32{1, 2}, 0x20),
		}},
	}
	snapshot := session.snapshotFromNativeBlock(source, block, false, false)
	if len(snapshot.Layers) != 1 {
		t.Fatalf("snapshot layers = %d, want 1", len(snapshot.Layers))
	}
	layer := snapshot.Layers[0]
	if layer.CacheIndex != 3 || layer.CacheMode != metal.KVCacheModePaged || layer.MaxSize != 64 {
		t.Fatalf("snapshot layer cache metadata = %d/%q/%d, want 3/paged/64", layer.CacheIndex, layer.CacheMode, layer.MaxSize)
	}
}

func TestNativeTextSession_RestoreKVBlocksGenerateSampledUsesRetainedBoundaryMetadata_Good(t *testing.T) {
	ctx := context.Background()
	session := newNativeSessionTextSession()
	model := testNativeTextSessionModel(session)
	handle := model.NewSession()
	source := metal.KVSnapshotBlockSource{
		TokenCount:   3,
		PrefixTokens: 3,
		BlockCount:   2,
		Load: func(_ context.Context, index int) (metal.KVSnapshotBlock, error) {
			switch index {
			case 0:
				return nativeSessionTextMetalBlock(0, 0, []int32{1, 2}, false), nil
			case 1:
				return nativeSessionTextMetalBlock(1, 2, []int32{3}, true), nil
			default:
				return metal.KVSnapshotBlock{}, nil
			}
		},
	}
	if err := handle.(interface {
		RestoreKVBlocks(context.Context, metal.KVSnapshotBlockSource) error
	}).RestoreKVBlocks(ctx, source); err != nil {
		t.Fatalf("RestoreKVBlocks: %v", err)
	}
	wantLogits := nativeTextF32ToBF16([]float32{0.1, 0.2, 0.3, 0.4})
	if !reflect.DeepEqual(session.restored.RetainedLogits, wantLogits) {
		t.Fatalf("restored retained logits = %v, want %v", session.restored.RetainedLogits, wantLogits)
	}
	var generated []int32
	for tok := range handle.Generate(ctx, metal.GenerateConfig{
		MaxTokens:           1,
		Temperature:         0.8,
		TopK:                2,
		Seed:                7,
		SeedSet:             true,
		SuppressTokens:      []int32{1},
		MinTokensBeforeStop: 1,
	}) {
		generated = append(generated, tok.ID)
	}
	if err := handle.Err(); err != nil {
		t.Fatalf("Generate Err: %v", err)
	}
	if !reflect.DeepEqual(generated, []int32{3}) {
		t.Fatalf("sampled generated = %v, want [3]", generated)
	}
	if session.generateSampledCacheCalls != 1 {
		t.Fatalf("GenerateSampledFromCacheEach calls = %d, want 1", session.generateSampledCacheCalls)
	}
	if session.generateSampledCalls != 0 || session.generateFromLogitsCalls != 0 || session.generateFromCacheCalls != 0 {
		t.Fatalf("other generate calls sampledLogits/logits/cache = %d/%d/%d, want 0/0/0", session.generateSampledCalls, session.generateFromLogitsCalls, session.generateFromCacheCalls)
	}
}

func nativeSessionTextBlockSource() metal.KVSnapshotBlockSource {
	return metal.KVSnapshotBlockSource{
		TokenCount:   3,
		PrefixTokens: 3,
		BlockCount:   2,
		Load: func(_ context.Context, index int) (metal.KVSnapshotBlock, error) {
			switch index {
			case 0:
				return nativeSessionTextMetalBlock(0, 0, []int32{1, 2}, false), nil
			case 1:
				return nativeSessionTextMetalBlock(1, 2, []int32{3}, true), nil
			default:
				return metal.KVSnapshotBlock{}, nil
			}
		},
	}
}

func nativeSessionTextMetalBlock(index, start int, tokens []int32, final bool) metal.KVSnapshotBlock {
	keyBytes := nativeSessionTextKVBytes(tokens, 0x10)
	valueBytes := nativeSessionTextKVBytes(tokens, 0x20)
	snapshot := &metal.KVSnapshot{
		Version:       metal.KVSnapshotVersion,
		Architecture:  "gemma4",
		Tokens:        append([]int32(nil), tokens...),
		TokenOffset:   start + len(tokens),
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        len(tokens),
		HeadDim:       2,
		NumQueryHeads: 1,
		Layers: []metal.KVLayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			CacheMode:  metal.KVCacheModeFixed,
			KeyDType:   metal.DTypeBFloat16,
			KeyBytes:   keyBytes,
			KeyShape:   []int32{int32(len(tokens)), 1, 2},
			ValueDType: metal.DTypeBFloat16,
			ValueBytes: valueBytes,
			ValueShape: []int32{int32(len(tokens)), 1, 2},
		}},
	}
	if final {
		snapshot.LogitShape = []int32{1, 1, 4}
		snapshot.Logits = []float32{0.1, 0.2, 0.3, 0.4}
	}
	return metal.KVSnapshotBlock{
		Index:      index,
		TokenStart: start,
		TokenCount: len(tokens),
		Snapshot:   snapshot,
	}
}
