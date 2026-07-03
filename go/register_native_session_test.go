// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mlx

import (
	"bytes"
	"context"
	"encoding/binary"
	"image"
	"image/color"
	"image/png"
	"iter"
	"math"
	"reflect"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	memvid "dappco.re/go/inference/state"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/pkg/model"
	"dappco.re/go/mlx/pkg/native"
	pkgtokenizer "dappco.re/go/mlx/pkg/tokenizer"
)

type nativeSessionTextTokenModel struct {
	sessions       []*nativeSessionTextSession
	embedRows      map[int32][]byte
	embeddingBytes int
	embedIntoCalls int
	queryHeads     int
	opens          int
}

func (m *nativeSessionTextTokenModel) Embed(id int32) ([]byte, error) {
	if row, ok := m.embedRows[id]; ok {
		return row, nil
	}
	return []byte{byte(id)}, nil
}

func (m *nativeSessionTextTokenModel) EmbeddingBytes() int { return m.embeddingBytes }

func (m *nativeSessionTextTokenModel) EmbedInto(dst []byte, id int32) ([]byte, error) {
	m.embedIntoCalls++
	row, err := m.Embed(id)
	if err != nil {
		return nil, err
	}
	if len(dst) != len(row) {
		return nil, core.NewError("test nativeSessionTextTokenModel: dst size mismatch")
	}
	copy(dst, row)
	return dst, nil
}

func (m *nativeSessionTextTokenModel) DecodeForward([][]byte) ([][]byte, error) {
	return [][]byte{{0}}, nil
}

func (m *nativeSessionTextTokenModel) Head([]byte) ([]byte, error) {
	return nativeTextF32ToBF16([]float32{0.1, 0.2, 0.3, 0.4}), nil
}

func (m *nativeSessionTextTokenModel) Vocab() int { return 4 }

func (m *nativeSessionTextTokenModel) NumQueryHeads() int {
	if m == nil {
		return 0
	}
	return m.queryHeads
}

func (m *nativeSessionTextTokenModel) OpenSession() (model.DecodeStepper, error) {
	if m.opens >= len(m.sessions) {
		s := newNativeSessionTextSession()
		m.sessions = append(m.sessions, s)
	}
	sess := m.sessions[m.opens]
	m.opens++
	return sess, nil
}

type nativeSessionVisionTokenModel struct {
	*nativeSessionTextTokenModel
	accepts           bool
	imageTokenID      int32
	projected         []byte
	projectImageCalls int
}

func (m *nativeSessionVisionTokenModel) AcceptsImageInput() bool { return m.accepts }

func (m *nativeSessionVisionTokenModel) ImagePlaceholderTokenID() int32 { return m.imageTokenID }

func (m *nativeSessionVisionTokenModel) ImagePlaceholderBlock(softTokens int) string {
	if softTokens <= 0 {
		return ""
	}
	var b core.Builder
	b.WriteString("<|image>")
	for i := 0; i < softTokens; i++ {
		b.WriteString("<|image|>")
	}
	b.WriteString("<image|>")
	return b.String()
}

func (m *nativeSessionVisionTokenModel) ProjectImageFeatures([]byte) ([]byte, error) {
	m.projectImageCalls++
	return append([]byte(nil), m.projected...), nil
}

type nativeSessionVisionPixelTokenModel struct {
	*nativeSessionVisionTokenModel
	projectPixelsCalls int
}

func (m *nativeSessionVisionPixelTokenModel) ProjectImagePixels([]float32, int, int) ([]byte, error) {
	m.projectPixelsCalls++
	return append([]byte(nil), m.projected...), nil
}

type nativeSessionAudioTokenModel struct {
	*nativeSessionTextTokenModel
	accepts           bool
	audioTokenID      int32
	projected         []byte
	projectAudioCalls int
}

func (m *nativeSessionAudioTokenModel) AcceptsAudioInput() bool { return m.accepts }

func (m *nativeSessionAudioTokenModel) AudioPlaceholderTokenID() int32 { return m.audioTokenID }

func (m *nativeSessionAudioTokenModel) AudioPlaceholderBlock(softTokens int) string {
	if softTokens <= 0 {
		return ""
	}
	var b core.Builder
	b.WriteString("<|audio>")
	for i := 0; i < softTokens; i++ {
		b.WriteString("<|audio|>")
	}
	b.WriteString("<audio|>")
	return b.String()
}

func (m *nativeSessionAudioTokenModel) ProjectAudioFeatures([]byte, int, int) ([]byte, error) {
	m.projectAudioCalls++
	return append([]byte(nil), m.projected...), nil
}

type nativeSessionTextSession struct {
	tokens                     []int32
	prefillEmbeddings          [][]byte
	logits                     []byte
	pos                        int
	restored                   native.SessionStateBlockSource
	restoredBlocks             []native.SessionStateBlock
	generateFromLogitsCalls    int
	generateSampledCalls       int
	generateSampledCacheCalls  int
	generateFromCacheCalls     int
	generateSuppressCacheCalls int
	closeCalls                 int
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
	tokens                     []int32
	logits                     []byte
	pos                        int
	restored                   native.SessionStateBlockSource
	generateSampledCacheCalls  int
	generateFromCacheCalls     int
	generateSuppressCacheCalls int
	closeCalls                 int
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

func (s *retainedBoundaryOnlySession) PrefillTokenEmbeddings(tokens []int32, _ [][]byte) error {
	return s.PrefillTokens(tokens)
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

func (s *retainedBoundaryOnlySession) GenerateFromCacheEachWithSuppressionAndTransform(maxNew, _ int, suppress []int32, _ native.TokenTransform, yield func(int32) bool) ([]int32, error) {
	s.generateSuppressCacheCalls++
	if len(suppress) == 0 {
		return s.generate(maxNew, yield), nil
	}
	return s.generateFrom([]int32{2}, maxNew, yield), nil
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
	return s.generateFrom(out, maxNew, yield)
}

func (s *retainedBoundaryOnlySession) generateFrom(out []int32, maxNew int, yield func(int32) bool) []int32 {
	if maxNew <= 0 {
		return nil
	}
	if len(out) > maxNew {
		out = out[:maxNew]
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
	s.prefillEmbeddings = nil
	s.pos = len(tokens)
	return nil
}

func (s *nativeSessionTextSession) PrefillTokenEmbeddings(tokens []int32, embeddings [][]byte) error {
	s.tokens = append(s.tokens[:0], tokens...)
	s.prefillEmbeddings = s.prefillEmbeddings[:0]
	for _, emb := range embeddings {
		s.prefillEmbeddings = append(s.prefillEmbeddings, append([]byte(nil), emb...))
	}
	s.pos = len(tokens)
	return nil
}

func (s *nativeSessionTextSession) AppendTokens(tokens []int32) error {
	s.tokens = append(s.tokens, tokens...)
	s.pos = len(s.tokens)
	return nil
}

func (s *nativeSessionTextSession) WarmPromptCache(tokens []int32) error {
	return s.PrefillTokens(tokens)
}

func (s *nativeSessionTextSession) CachedPrefixLen(promptIDs []int32) int {
	lcp := 0
	for lcp < len(promptIDs) && lcp < len(s.tokens) && promptIDs[lcp] == s.tokens[lcp] {
		lcp++
	}
	return lcp
}

func (s *nativeSessionTextSession) GenerateCached(tokens []int32, maxNew, _ int) ([]int32, error) {
	if err := s.PrefillTokens(tokens); err != nil {
		return nil, err
	}
	return s.generate(maxNew, nil), nil
}

func (s *nativeSessionTextSession) ClearPromptCache() {
	s.tokens = nil
	s.logits = nil
	s.pos = 0
}

func (s *nativeSessionTextSession) BoundaryLogits() ([]byte, error) {
	return append([]byte(nil), s.logits...), nil
}

func (s *nativeSessionTextSession) GenerateFromCacheEach(maxNew, _ int, yield func(int32) bool) ([]int32, error) {
	s.generateFromCacheCalls++
	return s.generate(maxNew, yield), nil
}

func (s *nativeSessionTextSession) GenerateFromCacheEachWithSuppressionAndTransform(maxNew, _ int, suppress []int32, _ native.TokenTransform, yield func(int32) bool) ([]int32, error) {
	s.generateSuppressCacheCalls++
	if len(suppress) == 0 {
		return s.generate(maxNew, yield), nil
	}
	return s.generateFrom([]int32{2, 1}, maxNew, yield), nil
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
	return s.generateFrom(out, maxNew, yield)
}

func (s *nativeSessionTextSession) generateFrom(out []int32, maxNew int, yield func(int32) bool) []int32 {
	if maxNew <= 0 {
		return nil
	}
	if len(out) > maxNew {
		out = out[:maxNew]
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

func TestLoadNativeTextModelValidatesLoadOptionsBeforeDisk_Good(t *testing.T) {
	_, err := LoadNativeTextModel("missing-model-dir", WithPagedKVPageSize(-1))
	if err == nil {
		t.Fatal("LoadNativeTextModel accepted negative paged KV page size")
	}
	if !strings.Contains(err.Error(), "paged KV page size") {
		t.Fatalf("LoadNativeTextModel error = %v, want paged KV validation", err)
	}
}

func TestNativeTextModelCapabilitiesReportsPagedRuntime_Good(t *testing.T) {
	model := testNativeTextSessionModel()
	model.maxLen = 64
	model.pagedKVPageSize = 16
	model.pagedKVPrealloc = true

	report := model.Capabilities()
	if report.Runtime.Backend != "native" || !report.Runtime.NativeRuntime || report.Runtime.CacheMode != "paged" {
		t.Fatalf("runtime = %+v, want native paged runtime", report.Runtime)
	}
	if report.Model.ContextLength != 64 {
		t.Fatalf("model context length = %d, want 64", report.Model.ContextLength)
	}
	if len(report.CacheModes) != 1 || report.CacheModes[0] != "paged" {
		t.Fatalf("cache modes = %v, want [paged]", report.CacheModes)
	}
	if report.Labels["paged_kv_page_size"] != "16" || report.Labels["paged_kv_prealloc"] != "true" {
		t.Fatalf("labels = %v, want paged KV page/prealloc metadata", report.Labels)
	}
}

func TestEnableConversationContinuityNativeTextModel_Good(t *testing.T) {
	model := testNativeTextSessionModel(newNativeSessionTextSession())
	store := memvid.NewInMemoryStore(nil)

	continuity, err := EnableConversationContinuity(model, ConversationContinuityOptions{Store: store})
	if err != nil {
		t.Fatalf("EnableConversationContinuity(native): %v", err)
	}
	if continuity == nil {
		t.Fatal("EnableConversationContinuity(native) = nil manager")
	}
	if model.continuity != continuity {
		t.Fatal("native model did not retain the conversation continuity manager")
	}
}

func TestNativeTextModelChatUsesConversationContinuity_Good(t *testing.T) {
	model := testNativeTextSessionModel(newNativeSessionTextSession())
	tok, err := pkgtokenizer.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	model.tok = tok
	store := memvid.NewInMemoryStore(nil)
	if _, err := EnableConversationContinuity(model, ConversationContinuityOptions{Store: store}); err != nil {
		t.Fatalf("EnableConversationContinuity(native): %v", err)
	}

	var got []inference.Token
	for token := range model.Chat(context.Background(), []inference.Message{{Role: "user", Content: "hello"}}, inference.WithMaxTokens(1)) {
		got = append(got, token)
	}
	if err := resultError(model.Err()); err != nil {
		t.Fatalf("Chat Err() = %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("Chat yielded %d tokens, want 1", len(got))
	}
	stats := model.continuity.Stats()
	if stats.FreshConversations != 1 || stats.StatelessFallbacks != 0 {
		t.Fatalf("continuity stats = %+v, want one fresh conversation and no fallback", stats)
	}
}

func TestNativeTextModelAcceptsImages_Good(t *testing.T) {
	model := testNativeTextSessionModel()
	if model.AcceptsImages() {
		t.Fatal("AcceptsImages = true for text-only native token model, want false")
	}

	model.tm = &nativeSessionVisionTokenModel{
		nativeSessionTextTokenModel: &nativeSessionTextTokenModel{},
		accepts:                     true,
	}
	if model.AcceptsImages() {
		t.Fatal("AcceptsImages = true for token-model-only native image payload, want false until image chat is implemented")
	}

	model.tm = &nativeSessionVisionTokenModel{
		nativeSessionTextTokenModel: &nativeSessionTextTokenModel{},
		accepts:                     false,
	}
	if model.AcceptsImages() {
		t.Fatal("AcceptsImages = true when native token model declines images, want false")
	}
}

func TestNativeTextModelChatImagesNotSilentlyDropped_Bad(t *testing.T) {
	tok, err := pkgtokenizer.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	model := testNativeTextSessionModel(newNativeSessionTextSession())
	model.tok = tok
	for range model.Chat(context.Background(), []inference.Message{{
		Role:    "user",
		Content: "describe this",
		Images:  [][]byte{{1, 2, 3}},
	}}, inference.WithMaxTokens(1)) {
		t.Fatal("Chat yielded a token for an image message, want a capability error")
	}
	err = resultError(model.Err())
	if err == nil {
		t.Fatal("Chat Err() = nil for image message, want capability error")
	}
	if !strings.Contains(err.Error(), "image") {
		t.Fatalf("Chat Err() = %q, want image capability error", err)
	}
}

func TestNativeTextModelAcceptsAudioInputRequiresExtractor_Good(t *testing.T) {
	model := testNativeTextSessionModel()
	model.tm = &nativeSessionAudioTokenModel{
		nativeSessionTextTokenModel: &nativeSessionTextTokenModel{},
		accepts:                     true,
	}
	if model.AcceptsAudioInput() {
		t.Fatal("AcceptsAudioInput = true without an audio feature extractor, want false")
	}
	extractor, err := native.NewAudioFeatureExtractor(&native.AudioFeatureConfig{
		NumMelFilters: 4, SamplingRate: 16_000, FrameLength: 4, HopLength: 2, MaxFrequency: 8000,
	})
	if err != nil {
		t.Fatalf("NewAudioFeatureExtractor: %v", err)
	}
	model.audioFeatures = extractor
	if !model.AcceptsAudioInput() {
		t.Fatal("AcceptsAudioInput = false with audio-capable token model and extractor, want true")
	}

	model.tm = &nativeSessionAudioTokenModel{
		nativeSessionTextTokenModel: &nativeSessionTextTokenModel{},
		accepts:                     false,
	}
	if model.AcceptsAudioInput() {
		t.Fatal("AcceptsAudioInput = true when native token model declines audio, want false")
	}
}

func TestNativeTextModelAudioPromptEmbeddings_Good(t *testing.T) {
	model := testNativeTextSessionModel()
	textRows := map[int32][]byte{
		10: {0x10},
		11: {0x11},
		12: {0x12},
	}
	model.tm = &nativeSessionAudioTokenModel{
		nativeSessionTextTokenModel: &nativeSessionTextTokenModel{embedRows: textRows},
		accepts:                     true,
		audioTokenID:                77,
	}
	ids := []int32{10, 77, 11, 77, 12}
	features := [][]byte{{0xa1, 0xa2}}
	got, err := model.nativeAudioPromptEmbeddings(ids, 77, features)
	if err != nil {
		t.Fatalf("nativeAudioPromptEmbeddings: %v", err)
	}
	if !reflect.DeepEqual(got[1], []byte{0xa1}) || !reflect.DeepEqual(got[3], []byte{0xa2}) {
		t.Fatalf("audio rows = %v/%v, want projected rows", got[1], got[3])
	}
	if &got[1][0] != &features[0][0] || &got[3][0] != &features[0][1] {
		t.Fatal("audio feature rows were copied; want borrowed projected feature row views")
	}
	if &got[0][0] != &textRows[10][0] || &got[2][0] != &textRows[11][0] || &got[4][0] != &textRows[12][0] {
		t.Fatal("text embedding rows were copied; want borrowed token embedding row views")
	}
	if !reflect.DeepEqual(got[0], textRows[10]) || !reflect.DeepEqual(got[2], textRows[11]) || !reflect.DeepEqual(got[4], textRows[12]) {
		t.Fatalf("text rows changed: got %v", got)
	}
	if _, err := model.nativeAudioPromptEmbeddings(ids, 77, [][]byte{{0xa1}}); err == nil {
		t.Fatal("nativeAudioPromptEmbeddings with too few feature rows error = nil")
	}
}

func TestNativeTextModelAudioPromptEmbeddingsUsesEmbedInto_Good(t *testing.T) {
	model := testNativeTextSessionModel()
	textRows := map[int32][]byte{
		10: {0x10},
		11: {0x11},
		12: {0x12},
		77: {0x00},
	}
	tm := &nativeSessionTextTokenModel{embedRows: textRows, embeddingBytes: 1}
	model.tm = &nativeSessionAudioTokenModel{
		nativeSessionTextTokenModel: tm,
		accepts:                     true,
		audioTokenID:                77,
	}
	ids := []int32{10, 77, 11, 77, 12}
	features := [][]byte{{0xa1, 0xa2}}
	got, err := model.nativeAudioPromptEmbeddings(ids, 77, features)
	if err != nil {
		t.Fatalf("nativeAudioPromptEmbeddings: %v", err)
	}
	if tm.embedIntoCalls != len(ids) {
		t.Fatalf("EmbedInto calls = %d, want %d", tm.embedIntoCalls, len(ids))
	}
	if !reflect.DeepEqual(got[0], textRows[10]) || !reflect.DeepEqual(got[2], textRows[11]) || !reflect.DeepEqual(got[4], textRows[12]) {
		t.Fatalf("text rows = %v/%v/%v, want embedInto rows", got[0], got[2], got[4])
	}
	if &got[1][0] != &features[0][0] || &got[3][0] != &features[0][1] {
		t.Fatal("audio feature rows were copied; want borrowed projected feature row views")
	}
}

const rootImageTokenizerJSON = `{
  "model": {
    "type": "BPE",
    "vocab": {"▁": 1, "d": 2, "e": 3, "s": 4, "c": 5, "r": 6, "i": 7, "b": 8},
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<bos>", "special": true},
    {"id": 9, "content": "<start_of_turn>", "special": true},
    {"id": 10, "content": "<end_of_turn>", "special": true},
    {"id": 11, "content": "<|image>", "special": true},
    {"id": 12, "content": "<|image|>", "special": true},
    {"id": 13, "content": "<image|>", "special": true},
    {"id": 14, "content": "<eos>", "special": true}
  ]
}`

func writeRootImageTokenizer(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	path := core.PathJoin(dir, "tokenizer.json")
	if result := core.WriteFile(path, []byte(rootImageTokenizerJSON), 0o644); !result.OK {
		t.Fatalf("write tokenizer: %v", result.Value)
	}
	return path
}

func nativeTextTestPNG(t *testing.T, width, height int) []byte {
	t.Helper()
	img := image.NewNRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			img.SetNRGBA(x, y, color.NRGBA{R: uint8(x + 1), G: uint8(y + 1), B: 7, A: 255})
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("encode png: %v", err)
	}
	return buf.Bytes()
}

func TestNativeTextModelChatImagesPrefillsProjectedFeatures_Good(t *testing.T) {
	tok, err := pkgtokenizer.LoadTokenizer(writeRootImageTokenizer(t))
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	session := newNativeSessionTextSession()
	tm := &nativeSessionVisionTokenModel{
		nativeSessionTextTokenModel: &nativeSessionTextTokenModel{sessions: []*nativeSessionTextSession{session}},
		accepts:                     true,
		imageTokenID:                12,
		projected:                   []byte{0xa1, 0xa2},
	}
	model := testNativeTextSessionModel()
	model.tm = tm
	model.tok = tok
	model.maxLen = 64
	model.imageFeatures = &native.VisionImageFeatureConfig{
		PatchSize: 16, MaxSoftTokens: 2, PoolingKernelSize: 1, RescaleFactor: 1.0 / 255.0,
	}

	var got []int32
	for tok := range model.Chat(context.Background(), []inference.Message{{
		Role:    "user",
		Content: "describe",
		Images:  [][]byte{nativeTextTestPNG(t, 32, 16)},
	}}, inference.WithMaxTokens(1)) {
		got = append(got, tok.ID)
	}
	if err := resultError(model.Err()); err != nil {
		t.Fatalf("Chat Err: %v", err)
	}
	if !reflect.DeepEqual(got, []int32{3}) {
		t.Fatalf("generated = %v, want [3]", got)
	}
	if tm.projectImageCalls != 1 {
		t.Fatalf("ProjectImageFeatures calls = %d, want 1", tm.projectImageCalls)
	}
	if session.generateFromCacheCalls != 1 {
		t.Fatalf("GenerateFromCache calls = %d, want 1", session.generateFromCacheCalls)
	}
	prefillTokens := session.tokens[:len(session.prefillEmbeddings)]
	var imageSlots []int
	for i, id := range prefillTokens {
		if id == 12 {
			imageSlots = append(imageSlots, i)
		}
	}
	if len(imageSlots) != 2 {
		t.Fatalf("image token slots = %v in tokens %v, want 2", imageSlots, prefillTokens)
	}
	if !reflect.DeepEqual(session.prefillEmbeddings[imageSlots[0]], []byte{0xa1}) ||
		!reflect.DeepEqual(session.prefillEmbeddings[imageSlots[1]], []byte{0xa2}) {
		t.Fatalf("image embeddings = %v/%v, want projected rows", session.prefillEmbeddings[imageSlots[0]], session.prefillEmbeddings[imageSlots[1]])
	}
}

func TestNativeTextModelChatImagesCachesProjectedFeatures_Good(t *testing.T) {
	tok, err := pkgtokenizer.LoadTokenizer(writeRootImageTokenizer(t))
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	tm := &nativeSessionVisionTokenModel{
		nativeSessionTextTokenModel: &nativeSessionTextTokenModel{sessions: []*nativeSessionTextSession{
			newNativeSessionTextSession(), newNativeSessionTextSession(),
		}},
		accepts:      true,
		imageTokenID: 12,
		projected:    []byte{0xb1, 0xb2},
	}
	model := testNativeTextSessionModel()
	model.tm = tm
	model.tok = tok
	model.maxLen = 64
	model.imageFeatures = &native.VisionImageFeatureConfig{
		PatchSize: 16, MaxSoftTokens: 2, PoolingKernelSize: 1, RescaleFactor: 1.0 / 255.0,
	}
	imageBytes := nativeTextTestPNG(t, 32, 16)
	messages := []inference.Message{{Role: "user", Content: "describe", Images: [][]byte{imageBytes}}}

	for turn := 0; turn < 2; turn++ {
		for range model.Chat(context.Background(), messages, inference.WithMaxTokens(1)) {
		}
		if err := resultError(model.Err()); err != nil {
			t.Fatalf("turn %d Chat Err: %v", turn+1, err)
		}
	}
	if tm.projectImageCalls != 1 {
		t.Fatalf("ProjectImageFeatures calls = %d, want 1 for repeated image bytes", tm.projectImageCalls)
	}
}

func TestNativeTextModelChatImagesPrefersRawPixelsProjection_Good(t *testing.T) {
	tok, err := pkgtokenizer.LoadTokenizer(writeRootImageTokenizer(t))
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	session := newNativeSessionTextSession()
	base := &nativeSessionVisionTokenModel{
		nativeSessionTextTokenModel: &nativeSessionTextTokenModel{sessions: []*nativeSessionTextSession{session}},
		accepts:                     true,
		imageTokenID:                12,
		projected:                   []byte{0xc1, 0xc2},
	}
	tm := &nativeSessionVisionPixelTokenModel{nativeSessionVisionTokenModel: base}
	model := testNativeTextSessionModel()
	model.tm = tm
	model.tok = tok
	model.maxLen = 64
	model.imageFeatures = &native.VisionImageFeatureConfig{
		PatchSize: 16, MaxSoftTokens: 2, PoolingKernelSize: 1, RescaleFactor: 1.0 / 255.0,
	}

	for range model.Chat(context.Background(), []inference.Message{{
		Role:    "user",
		Content: "describe",
		Images:  [][]byte{nativeTextTestPNG(t, 32, 16)},
	}}, inference.WithMaxTokens(1)) {
	}
	if err := resultError(model.Err()); err != nil {
		t.Fatalf("Chat Err: %v", err)
	}
	if tm.projectPixelsCalls != 1 {
		t.Fatalf("ProjectImagePixels calls = %d, want 1", tm.projectPixelsCalls)
	}
	if tm.projectImageCalls != 0 {
		t.Fatalf("ProjectImageFeatures calls = %d, want 0 when raw pixel projection is available", tm.projectImageCalls)
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

func TestNativeTextSessionGenerateSuppressTokensUsesGreedyCachePath(t *testing.T) {
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

	var generated []int32
	for tok := range handle.Generate(ctx, metal.GenerateConfig{MaxTokens: 1, SuppressTokens: []int32{3}}) {
		generated = append(generated, tok.ID)
	}
	if err := handle.Err(); err != nil {
		t.Fatalf("Generate Err: %v", err)
	}
	if !reflect.DeepEqual(generated, []int32{2}) {
		t.Fatalf("suppressed greedy generated = %v, want [2]", generated)
	}
	if session.generateSuppressCacheCalls != 1 {
		t.Fatalf("GenerateFromCacheEachWithSuppressionAndTransform calls = %d, want 1", session.generateSuppressCacheCalls)
	}
	if session.generateSampledCacheCalls != 0 || session.generateFromCacheCalls != 0 {
		t.Fatalf("other generate calls sampled/cache = %d/%d, want 0/0", session.generateSampledCacheCalls, session.generateFromCacheCalls)
	}
}

func TestNativeTextSessionGenerateTraceTokenPhases_Good(t *testing.T) {
	ctx := context.Background()
	nativeModel := testNativeTextSessionModel(newNativeSessionTextSession())
	handle := nativeModel.NewSession()
	if handle == nil {
		t.Fatal("NewSession() = nil, want native session handle")
	}
	prefiller := handle.(interface {
		PrefillTokens(context.Context, []int32) error
	})
	if err := prefiller.PrefillTokens(ctx, []int32{1}); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}

	var generated []int32
	for tok := range handle.Generate(ctx, metal.GenerateConfig{MaxTokens: 2, TraceTokenPhases: true, TraceTokenText: true}) {
		generated = append(generated, tok.ID)
	}
	if err := handle.Err(); err != nil {
		t.Fatalf("Generate Err: %v", err)
	}
	if !reflect.DeepEqual(generated, []int32{3, 2}) {
		t.Fatalf("generated = %v, want [3 2]", generated)
	}
	tracer := handle.(interface {
		LastTokenPhases() []metal.TokenPhaseTrace
	})
	phases := tracer.LastTokenPhases()
	if len(phases) != len(generated) {
		t.Fatalf("TokenPhases len = %d, want %d: %+v", len(phases), len(generated), phases)
	}
	for i := range phases {
		if phases[i].Step != i {
			t.Fatalf("phase[%d].Step = %d, want %d", i, phases[i].Step, i)
		}
		if phases[i].TokenID != generated[i] {
			t.Fatalf("phase[%d].TokenID = %d, want %d", i, phases[i].TokenID, generated[i])
		}
		if phases[i].TotalDuration <= 0 || phases[i].ForwardDuration <= 0 {
			t.Fatalf("phase[%d] durations = total %s forward %s, want positive native wall timing", i, phases[i].TotalDuration, phases[i].ForwardDuration)
		}
	}
	phases[0].Step = 99
	if again := tracer.LastTokenPhases(); again[0].Step == 99 {
		t.Fatal("LastTokenPhases returned aliased storage")
	}

	for range handle.Generate(ctx, metal.GenerateConfig{MaxTokens: 1}) {
	}
	if err := handle.Err(); err != nil {
		t.Fatalf("second Generate Err: %v", err)
	}
	if phases := tracer.LastTokenPhases(); len(phases) != 0 {
		t.Fatalf("untraced Generate left stale phases: %+v", phases)
	}
}

func TestNativeTextSessionGenerateMinTokensBeforeStopUsesStagedGreedyCachePath(t *testing.T) {
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

	var generated []int32
	for tok := range handle.Generate(ctx, metal.GenerateConfig{
		MaxTokens:           2,
		StopTokens:          []int32{3},
		MinTokensBeforeStop: 1,
	}) {
		generated = append(generated, tok.ID)
	}
	if err := handle.Err(); err != nil {
		t.Fatalf("Generate Err: %v", err)
	}
	if !reflect.DeepEqual(generated, []int32{2, 3}) {
		t.Fatalf("min-stop greedy generated = %v, want [2 3]", generated)
	}
	if session.generateSuppressCacheCalls != 1 {
		t.Fatalf("GenerateFromCacheEachWithSuppressionAndTransform calls = %d, want 1", session.generateSuppressCacheCalls)
	}
	if session.generateFromCacheCalls != 1 {
		t.Fatalf("GenerateFromCacheEach calls = %d, want 1", session.generateFromCacheCalls)
	}
	if session.generateSampledCacheCalls != 0 {
		t.Fatalf("GenerateSampledFromCacheEach calls = %d, want 0", session.generateSampledCacheCalls)
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

func TestNativeTextSessionGenerateUpdatesModelMetrics_Good(t *testing.T) {
	ctx := context.Background()
	session := newNativeSessionTextSession()
	model := testNativeTextSessionModel(session)
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
	var out []int32
	for tok := range handle.Generate(ctx, metal.GenerateConfig{MaxTokens: 2}) {
		out = append(out, tok.ID)
	}
	if !reflect.DeepEqual(out, []int32{3, 2}) {
		t.Fatalf("Generate tokens = %v, want [3 2]", out)
	}
	metrics := model.Metrics()
	if metrics.PromptTokens != 3 {
		t.Fatalf("PromptTokens = %d, want 3", metrics.PromptTokens)
	}
	if metrics.GeneratedTokens != len(out) {
		t.Fatalf("GeneratedTokens = %d, want %d", metrics.GeneratedTokens, len(out))
	}
	if metrics.DecodeDuration <= 0 || metrics.TotalDuration <= 0 {
		t.Fatalf("metrics durations = decode %s total %s, want positive", metrics.DecodeDuration, metrics.TotalDuration)
	}
	if metrics.DecodeTokensPerSec <= 0 {
		t.Fatalf("DecodeTokensPerSec = %f, want positive", metrics.DecodeTokensPerSec)
	}
}

func TestNativeTextSessionPrefillAndAppendChunks_Good(t *testing.T) {
	ctx := context.Background()
	tok, err := pkgtokenizer.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	session := newNativeSessionTextSession()
	model := testNativeTextSessionModel(session)
	model.tok = tok
	handle := model.NewSession()
	if handle == nil {
		t.Fatal("NewSession() = nil, want native session handle")
	}
	prefiller, ok := handle.(interface {
		PrefillChunks(context.Context, iter.Seq[string]) error
	})
	if !ok {
		t.Fatal("native session handle does not expose chunk prefill")
	}
	appender, ok := handle.(interface {
		AppendPromptChunks(context.Context, iter.Seq[string]) error
	})
	if !ok {
		t.Fatal("native session handle does not expose chunk append")
	}
	chunks := func(yield func(string) bool) {
		yield("hello")
		yield("")
		yield("hello")
	}
	wantPrefill := append([]int32(nil), tok.Encode("hello")...)
	wantPrefill = append(wantPrefill, stripNativeImplicitChunkBOS(tok, tok.Encode("hello"))...)
	if err := prefiller.PrefillChunks(ctx, iter.Seq[string](chunks)); err != nil {
		t.Fatalf("PrefillChunks: %v", err)
	}
	if !reflect.DeepEqual(session.tokens, wantPrefill) {
		t.Fatalf("PrefillChunks tokens = %v, want %v", session.tokens, wantPrefill)
	}
	appendChunks := func(yield func(string) bool) {
		yield("hello")
	}
	wantAppend := append([]int32(nil), wantPrefill...)
	wantAppend = append(wantAppend, stripNativeImplicitChunkBOS(tok, tok.Encode("hello"))...)
	if err := appender.AppendPromptChunks(ctx, iter.Seq[string](appendChunks)); err != nil {
		t.Fatalf("AppendPromptChunks: %v", err)
	}
	if !reflect.DeepEqual(session.tokens, wantAppend) {
		t.Fatalf("AppendPromptChunks tokens = %v, want %v", session.tokens, wantAppend)
	}
}

func TestNativeTextModelCaptureKVWithOptions_Good(t *testing.T) {
	ctx := context.Background()
	tok, err := pkgtokenizer.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	session := newNativeSessionTextSession()
	model := testNativeTextSessionModel(session)
	model.tok = tok
	wantTokens := tok.Encode("hello")

	snapshot, err := model.CaptureKVWithOptions(ctx, "hello", metal.KVSnapshotCaptureOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("CaptureKVWithOptions: %v", err)
	}
	if snapshot.SeqLen != len(wantTokens) || snapshot.TokenOffset != len(wantTokens) || !reflect.DeepEqual(snapshot.Tokens, wantTokens) {
		t.Fatalf("snapshot timeline = seq %d offset %d tokens %v, want %d/%d/%v", snapshot.SeqLen, snapshot.TokenOffset, snapshot.Tokens, len(wantTokens), len(wantTokens), wantTokens)
	}
	if len(snapshot.Layers) != 1 || len(snapshot.Layers[0].KeyBytes) != len(wantTokens)*4 || snapshot.Layers[0].KeyDType != metal.DTypeBFloat16 {
		t.Fatalf("snapshot layer = %+v", snapshot.Layers)
	}
	if session.closeCalls != 1 {
		t.Fatalf("one-shot capture Close calls = %d, want 1", session.closeCalls)
	}
}

func TestNativeTextModelInspectAttentionFromNativeKV_Good(t *testing.T) {
	ctx := context.Background()
	tok, err := pkgtokenizer.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	session := newNativeSessionTextSession()
	model := testNativeTextSessionModel(session)
	model.tok = tok
	wantTokens := tok.Encode("hello")

	snapshot, err := model.InspectAttention(ctx, "hello")
	if err != nil {
		t.Fatalf("InspectAttention: %v", err)
	}
	if snapshot == nil {
		t.Fatal("InspectAttention = nil, want snapshot")
	}
	if snapshot.NumLayers != 1 || snapshot.NumHeads != 1 || snapshot.SeqLen != len(wantTokens) || snapshot.HeadDim != 2 || snapshot.NumQueryHeads != 1 {
		t.Fatalf("InspectAttention shape = %+v, want native KV shape", snapshot)
	}
	if snapshot.Architecture != "gemma4" {
		t.Fatalf("InspectAttention architecture = %q, want gemma4", snapshot.Architecture)
	}
	if snapshot.HasQueries() {
		t.Fatal("InspectAttention HasQueries = true, want K-only native snapshot")
	}
	if len(snapshot.Keys) != 1 || len(snapshot.Keys[0]) != 1 {
		t.Fatalf("InspectAttention keys shape = %+v, want one layer/head", snapshot.Keys)
	}
	wantKey := nativeTextBF16ToF32(nativeSessionTextKVBytes(wantTokens, 0x10))
	if !reflect.DeepEqual(snapshot.Keys[0][0], wantKey) {
		t.Fatalf("InspectAttention key = %v, want %v", snapshot.Keys[0][0], wantKey)
	}
	if session.closeCalls != 1 {
		t.Fatalf("one-shot InspectAttention Close calls = %d, want 1", session.closeCalls)
	}
}

func TestNativeTextModelCaptureKVChunksWithOptions_Good(t *testing.T) {
	ctx := context.Background()
	tok, err := pkgtokenizer.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	session := newNativeSessionTextSession()
	model := testNativeTextSessionModel(session)
	model.tok = tok
	chunks := func(yield func(string) bool) {
		yield("hello")
		yield("")
		yield("hello")
	}
	wantTokens := append([]int32(nil), tok.Encode("hello")...)
	wantTokens = append(wantTokens, stripNativeImplicitChunkBOS(tok, tok.Encode("hello"))...)

	snapshot, err := model.CaptureKVChunksWithOptions(ctx, iter.Seq[string](chunks), metal.KVSnapshotCaptureOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("CaptureKVChunksWithOptions: %v", err)
	}
	if snapshot.SeqLen != len(wantTokens) || snapshot.TokenOffset != len(wantTokens) || !reflect.DeepEqual(snapshot.Tokens, wantTokens) {
		t.Fatalf("chunk snapshot timeline = seq %d offset %d tokens %v, want %d/%d/%v", snapshot.SeqLen, snapshot.TokenOffset, snapshot.Tokens, len(wantTokens), len(wantTokens), wantTokens)
	}
	if len(snapshot.Layers) != 1 || len(snapshot.Layers[0].KeyBytes) != len(wantTokens)*4 || snapshot.Layers[0].KeyDType != metal.DTypeBFloat16 {
		t.Fatalf("chunk snapshot layer = %+v", snapshot.Layers)
	}
	if session.closeCalls != 1 {
		t.Fatalf("one-shot chunk capture Close calls = %d, want 1", session.closeCalls)
	}
}

func TestNativeTextModelRestorePromptCacheFromKV_Good(t *testing.T) {
	ctx := context.Background()
	session := newNativeSessionTextSession()
	model := testNativeTextSessionModel(session)
	snapshot := nativeSessionTextMetalBlock(0, 0, []int32{1, 2, 3}, true).Snapshot

	if err := model.RestorePromptCacheFromKV(ctx, snapshot); err != nil {
		t.Fatalf("RestorePromptCacheFromKV: %v", err)
	}
	if !reflect.DeepEqual(session.tokens, []int32{1, 2, 3}) {
		t.Fatalf("restored prompt cache tokens = %v, want [1 2 3]", session.tokens)
	}
	if session.pos != 3 {
		t.Fatalf("restored prompt cache position = %d, want 3", session.pos)
	}
	stats, err := model.CacheStats(ctx)
	if err != nil {
		t.Fatalf("CacheStats: %v", err)
	}
	if stats.Blocks != 1 {
		t.Fatalf("restored prompt cache stats blocks = %d, want 1", stats.Blocks)
	}
}

func TestNativeTextModelRestorePromptCacheFromKVBlocks_Good(t *testing.T) {
	ctx := context.Background()
	session := newNativeSessionTextSession()
	model := testNativeTextSessionModel(session)

	if err := model.RestorePromptCacheFromKVBlocks(ctx, nativeSessionTextBlockSource()); err != nil {
		t.Fatalf("RestorePromptCacheFromKVBlocks: %v", err)
	}
	if !reflect.DeepEqual(session.tokens, []int32{1, 2, 3}) {
		t.Fatalf("block-restored prompt cache tokens = %v, want [1 2 3]", session.tokens)
	}
	if session.pos != 3 {
		t.Fatalf("block-restored prompt cache position = %d, want 3", session.pos)
	}
	stats, err := model.CacheStats(ctx)
	if err != nil {
		t.Fatalf("CacheStats: %v", err)
	}
	if stats.Blocks != 1 {
		t.Fatalf("block-restored prompt cache stats blocks = %d, want 1", stats.Blocks)
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

func TestNativeTextSession_RestoreKVInfersUnderreportedRawLayerHeads_Good(t *testing.T) {
	ctx := context.Background()
	session := newNativeSessionTextSession()
	model := testNativeTextSessionModel(session)
	handle := model.NewSession()
	key := nativeTextF32ToBF16([]float32{1, 2, 3, 4, 5, 6, 7, 8})
	value := nativeTextF32ToBF16([]float32{11, 12, 13, 14, 15, 16, 17, 18})
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
			CacheIndex: 3,
			CacheMode:  metal.KVCacheModeFixed,
			KeyDType:   metal.DTypeBFloat16,
			KeyBytes:   key,
			KeyShape:   []int32{2, 1, 2},
			ValueDType: metal.DTypeBFloat16,
			ValueBytes: value,
			ValueShape: []int32{2, 1, 2},
		}},
	}
	if err := handle.(interface {
		RestoreKV(context.Context, *metal.KVSnapshot) error
	}).RestoreKV(ctx, snapshot); err != nil {
		t.Fatalf("RestoreKV underreported raw layer slab: %v", err)
	}
	if len(session.restoredBlocks) != 1 || len(session.restoredBlocks[0].Layers) != 1 {
		t.Fatalf("restored blocks = %+v, want one converted raw layer slab", session.restoredBlocks)
	}
	layer := session.restoredBlocks[0].Layers[0]
	if layer.KVHeads != 2 || layer.HeadDim != 2 || layer.RowBytes != 8 {
		t.Fatalf("inferred raw slab geometry = heads %d dim %d row %d, want 2/2/8", layer.KVHeads, layer.HeadDim, layer.RowBytes)
	}
	if !reflect.DeepEqual(layer.KeyBytes, key) || !reflect.DeepEqual(layer.ValueBytes, value) {
		t.Fatalf("inferred raw slab bytes = %v/%v, want %v/%v", layer.KeyBytes, layer.ValueBytes, key, value)
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

func TestNativeTextSession_RestoreKVBlocksPreservesSlidingTailTokenOffset_Good(t *testing.T) {
	ctx := context.Background()
	session := newNativeSessionTextSession()
	model := testNativeTextSessionModel(session)
	handle := model.NewSession()
	tail := nativeSessionTextMetalBlock(1, 4, []int32{5, 6}, true)
	tail.Snapshot.TokenOffset = 6
	tail.Snapshot.Layers[0].MaxSize = 2
	source := metal.KVSnapshotBlockSource{
		TokenCount:   6,
		PrefixTokens: 6,
		BlockCount:   1,
		Load: func(_ context.Context, index int) (metal.KVSnapshotBlock, error) {
			if index != 0 {
				return metal.KVSnapshotBlock{}, core.NewError("test: block index out of range")
			}
			return tail, nil
		},
	}
	if err := handle.(interface {
		RestoreKVBlocks(context.Context, metal.KVSnapshotBlockSource) error
	}).RestoreKVBlocks(ctx, source); err != nil {
		t.Fatalf("RestoreKVBlocks sliding tail: %v", err)
	}
	if session.pos != 6 {
		t.Fatalf("restored position = %d, want token offset 6", session.pos)
	}
	if len(session.restoredBlocks) != 2 {
		t.Fatalf("restored blocks = %+v, want expired prefix + live tail", session.restoredBlocks)
	}
	prefix, restoredTail := session.restoredBlocks[0], session.restoredBlocks[1]
	if prefix.TokenStart != 0 || prefix.TokenCount != 4 || len(prefix.Layers) != 1 {
		t.Fatalf("prefix block = start %d count %d layers %d, want 0/4/1", prefix.TokenStart, prefix.TokenCount, len(prefix.Layers))
	}
	if len(prefix.Layers[0].KeyBytes) != 0 || len(prefix.Layers[0].ValueBytes) != 0 {
		t.Fatalf("expired prefix carried KV bytes key=%v value=%v", prefix.Layers[0].KeyBytes, prefix.Layers[0].ValueBytes)
	}
	if restoredTail.TokenStart != 4 || restoredTail.TokenCount != 2 || len(restoredTail.Layers) != 1 {
		t.Fatalf("tail block = start %d count %d layers %d, want 4/2/1", restoredTail.TokenStart, restoredTail.TokenCount, len(restoredTail.Layers))
	}
	if want := nativeSessionTextKVBytes([]int32{5, 6}, 0x10); !reflect.DeepEqual(restoredTail.Layers[0].KeyBytes, want) {
		t.Fatalf("tail key bytes = %v, want %v", restoredTail.Layers[0].KeyBytes, want)
	}
	if want := nativeSessionTextKVBytes([]int32{5, 6}, 0x20); !reflect.DeepEqual(restoredTail.Layers[0].ValueBytes, want) {
		t.Fatalf("tail value bytes = %v, want %v", restoredTail.Layers[0].ValueBytes, want)
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

func TestNativeTextSession_SnapshotFromNativeBlockCarriesQueryHeads_Good(t *testing.T) {
	model := testNativeTextSessionModel(newNativeSessionTextSession())
	model.tm.(*nativeSessionTextTokenModel).queryHeads = 8
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
			CacheIndex: 0,
			KVHeads:    2,
			HeadDim:    2,
			RowBytes:   8,
			KeyBytes:   make([]byte, 16),
			ValueBytes: make([]byte, 16),
		}},
	}

	snapshot := session.snapshotFromNativeBlock(source, block, false, false)
	if snapshot.NumHeads != 2 {
		t.Fatalf("snapshot NumHeads = %d, want KV heads 2", snapshot.NumHeads)
	}
	if snapshot.NumQueryHeads != 8 {
		t.Fatalf("snapshot NumQueryHeads = %d, want model query heads 8", snapshot.NumQueryHeads)
	}
}

func TestNativeTextSession_SnapshotFromNativeBlockUsesPhysicalRowHeads_Good(t *testing.T) {
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
			CacheIndex: 0,
			KVHeads:    1,
			HeadDim:    2,
			RowBytes:   16,
			KeyBytes:   make([]byte, 32),
			ValueBytes: make([]byte, 32),
		}},
	}

	snapshot := session.snapshotFromNativeBlock(source, block, false, false)
	if snapshot.NumHeads != 4 {
		t.Fatalf("snapshot NumHeads = %d, want physical KV heads 4", snapshot.NumHeads)
	}
	layer := snapshot.Layers[0]
	wantShape := []int32{2, 4, 2}
	if !reflect.DeepEqual(layer.KeyShape, wantShape) || !reflect.DeepEqual(layer.ValueShape, wantShape) {
		t.Fatalf("snapshot layer shapes = %v/%v, want %v", layer.KeyShape, layer.ValueShape, wantShape)
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

func TestNativeTextModelGenerateNativeSpeculativeUsesMTP(t *testing.T) {
	tok, err := pkgtokenizer.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	targetSession := newNativeSessionTextSession()
	draftSession := newNativeSessionTextSession()
	targetTM := &nativeSessionTextTokenModel{sessions: []*nativeSessionTextSession{targetSession}}
	draftTM := &nativeSessionTextTokenModel{sessions: []*nativeSessionTextSession{draftSession}}
	target := &nativeTextModel{tm: targetTM, tok: tok, maxLen: 32}
	draft := &nativeTextModel{tm: draftTM, tok: tok, maxLen: 32}
	oldDecode := nativeTextMTPDecode
	defer func() { nativeTextMTPDecode = oldDecode }()
	var called bool
	nativeTextMTPDecode = func(target, draft model.DecodeStepper, prompt []int32, maxNew, eos, k int, yield func(int32) bool) (*native.MTPResult, bool, error) {
		called = true
		if yield != nil {
			t.Fatal("direct GenerateNativeSpeculative yield = non-nil, want buffered path")
		}
		if target != targetSession || draft != draftSession {
			t.Fatalf("MTP sessions = %T/%T, want opened target/draft sessions", target, draft)
		}
		if !reflect.DeepEqual(prompt, []int32{0, 10}) {
			t.Fatalf("MTP prompt ids = %v, want tokenizer ids [0 10]", prompt)
		}
		if maxNew != 3 || k != 2 {
			t.Fatalf("MTP args maxNew=%d k=%d, want 3/2", maxNew, k)
		}
		return &native.MTPResult{Tokens: []int32{10, 10}, Drafted: 2, Accepted: 1, Rounds: 1}, true, nil
	}

	result, handled, err := target.GenerateNativeSpeculative(context.Background(), draft, "hello", SpeculativeDecodeConfig{
		MaxTokens:   3,
		DraftTokens: 2,
	})
	if err != nil {
		t.Fatalf("GenerateNativeSpeculative: %v", err)
	}
	if !handled || !called {
		t.Fatalf("GenerateNativeSpeculative handled/called = %v/%v, want true/true", handled, called)
	}
	if result.Mode != SpeculativeDecodeModeMTP || result.Text != " hello hello" {
		t.Fatalf("result mode/text = %q/%q, want mtp/tokenizer-decoded text", result.Mode, result.Text)
	}
	if result.Metrics.DraftTokens != 2 || result.Metrics.AcceptedTokens != 1 || result.Metrics.RejectedTokens != 1 || result.Metrics.TargetCalls != 1 {
		t.Fatalf("result metrics = %+v, want MTP accounting", result.Metrics)
	}
	if targetSession.closeCalls != 1 || draftSession.closeCalls != 1 {
		t.Fatalf("session close calls target/draft = %d/%d, want 1/1", targetSession.closeCalls, draftSession.closeCalls)
	}
}

func TestNativeTextModelGenerateNativeSpeculativeUsesSampledMTP(t *testing.T) {
	tok, err := pkgtokenizer.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	targetSession := newNativeSessionTextSession()
	draftSession := newNativeSessionTextSession()
	targetTM := &nativeSessionTextTokenModel{sessions: []*nativeSessionTextSession{targetSession}}
	draftTM := &nativeSessionTextTokenModel{sessions: []*nativeSessionTextSession{draftSession}}
	target := &nativeTextModel{tm: targetTM, tok: tok, maxLen: 32}
	draft := &nativeTextModel{tm: draftTM, tok: tok, maxLen: 32}
	oldDecode := nativeTextMTPSampledDecode
	defer func() { nativeTextMTPSampledDecode = oldDecode }()
	var called bool
	nativeTextMTPSampledDecode = func(target, draft model.DecodeStepper, prompt []int32, maxNew int, stopTokens []int32, targetSampler, draftSampler *model.Sampler, params model.SampleParams, k int, yield func(int32) bool) (*native.MTPResult, bool, error) {
		called = true
		if yield != nil {
			t.Fatal("direct GenerateNativeSpeculative sampled yield = non-nil, want buffered path")
		}
		if target != targetSession || draft != draftSession {
			t.Fatalf("sampled MTP sessions = %T/%T, want opened target/draft sessions", target, draft)
		}
		if !reflect.DeepEqual(prompt, []int32{0, 10}) {
			t.Fatalf("sampled MTP prompt ids = %v, want tokenizer ids [0 10]", prompt)
		}
		if maxNew != 4 || k != 3 {
			t.Fatalf("sampled MTP args maxNew=%d k=%d, want 4/3", maxNew, k)
		}
		if !reflect.DeepEqual(stopTokens, []int32{7, 8}) {
			t.Fatalf("sampled stop tokens = %v, want [7 8]", stopTokens)
		}
		if targetSampler == nil || draftSampler == nil || targetSampler == draftSampler {
			t.Fatalf("sampled samplers target/draft = %p/%p, want distinct non-nil samplers", targetSampler, draftSampler)
		}
		if params.Temperature != 0.8 || params.TopK != 5 || params.TopP != 0.9 || params.MinP != 0.02 {
			t.Fatalf("sampled params temp/topK/topP/minP = %.1f/%d/%.1f/%.2f, want 0.8/5/0.9/0.02", params.Temperature, params.TopK, params.TopP, params.MinP)
		}
		if params.MinTokensBeforeStop != 1 || params.RepeatPenalty != 1.1 || !reflect.DeepEqual(params.SuppressTokens, []int32{9}) {
			t.Fatalf("sampled params suppress/minStop/repeat = %v/%d/%.1f, want [9]/1/1.1", params.SuppressTokens, params.MinTokensBeforeStop, params.RepeatPenalty)
		}
		return &native.MTPResult{Tokens: []int32{10, 10, 10}, Drafted: 3, Accepted: 2, Rounds: 2}, true, nil
	}

	result, handled, err := target.GenerateNativeSpeculative(context.Background(), draft, "hello", SpeculativeDecodeConfig{
		MaxTokens:   4,
		DraftTokens: 3,
		GenerateConfig: GenerateConfig{
			MaxTokens:           4,
			Temperature:         0.8,
			TopK:                5,
			TopP:                0.9,
			MinP:                0.02,
			Seed:                123,
			SeedSet:             true,
			StopTokens:          []int32{7, 8},
			SuppressTokens:      []int32{9},
			MinTokensBeforeStop: 1,
			RepeatPenalty:       1.1,
		},
	})
	if err != nil {
		t.Fatalf("GenerateNativeSpeculative sampled: %v", err)
	}
	if !handled || !called {
		t.Fatalf("GenerateNativeSpeculative sampled handled/called = %v/%v, want true/true", handled, called)
	}
	if result.Mode != SpeculativeDecodeModeMTP || result.Text != " hello hello hello" {
		t.Fatalf("sampled result mode/text = %q/%q, want mtp/tokenizer-decoded text", result.Mode, result.Text)
	}
	if result.Metrics.DraftTokens != 3 || result.Metrics.AcceptedTokens != 2 || result.Metrics.RejectedTokens != 1 || result.Metrics.TargetCalls != 2 {
		t.Fatalf("sampled result metrics = %+v, want MTP accounting", result.Metrics)
	}
	if targetSession.closeCalls != 1 || draftSession.closeCalls != 1 {
		t.Fatalf("sampled session close calls target/draft = %d/%d, want 1/1", targetSession.closeCalls, draftSession.closeCalls)
	}
}

func TestNativeSpeculativeTextModelGenerateUsesNativeMTP(t *testing.T) {
	tok, err := pkgtokenizer.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	targetSession := newNativeSessionTextSession()
	draftSession := newNativeSessionTextSession()
	target := &nativeTextModel{
		tm:     &nativeSessionTextTokenModel{sessions: []*nativeSessionTextSession{targetSession}},
		tok:    tok,
		maxLen: 32,
	}
	draft := &nativeTextModel{
		tm:     &nativeSessionTextTokenModel{sessions: []*nativeSessionTextSession{draftSession}},
		tok:    tok,
		maxLen: 32,
	}
	spec := &nativeSpeculativeTextModel{nativeTextModel: target, draft: draft, draftTokens: 2}
	if !IsSpeculativeTextModel(spec) {
		t.Fatal("IsSpeculativeTextModel(native wrapper) = false, want true")
	}
	oldDecode := nativeTextMTPDecode
	defer func() { nativeTextMTPDecode = oldDecode }()
	var called bool
	var order []string
	nativeTextMTPDecode = func(target, draft model.DecodeStepper, prompt []int32, maxNew, eos, k int, yield func(int32) bool) (*native.MTPResult, bool, error) {
		called = true
		if target != targetSession || draft != draftSession {
			t.Fatalf("wrapper MTP sessions = %T/%T, want opened target/draft sessions", target, draft)
		}
		if !reflect.DeepEqual(prompt, []int32{0, 10}) || maxNew != 3 || k != 2 {
			t.Fatalf("wrapper MTP args prompt=%v maxNew=%d k=%d, want [0 10]/3/2", prompt, maxNew, k)
		}
		if yield == nil {
			t.Fatal("wrapper MTP yield = nil, want streaming sink")
		}
		order = append(order, "hook-before")
		if !yield(10) || !yield(10) {
			t.Fatal("wrapper MTP yield returned false")
		}
		order = append(order, "hook-after")
		return &native.MTPResult{Tokens: []int32{10, 10}, Drafted: 2, Accepted: 1, Rounds: 1}, true, nil
	}

	var out []inference.Token
	for token := range spec.Generate(context.Background(), "hello", inference.WithMaxTokens(3), inference.WithTemperature(0)) {
		order = append(order, "caller")
		out = append(out, token)
	}
	if !called {
		t.Fatal("native speculative wrapper did not call native MTP")
	}
	if got := nativeSessionTextTokenText(out); got != " hello hello" {
		t.Fatalf("wrapper output = %q, want tokenizer-decoded native MTP text", got)
	}
	if !reflect.DeepEqual(order, []string{"hook-before", "caller", "caller", "hook-after"}) {
		t.Fatalf("wrapper streaming order = %v, want hook to yield directly into caller", order)
	}
	metrics := spec.MTPMetrics()
	if metrics == nil {
		t.Fatal("wrapper MTPMetrics = nil, want acceptance counters")
	}
	if metrics.ProposedTokens != 2 || metrics.AcceptedTokens != 1 || metrics.RejectedTokens != 1 || metrics.TargetVerifyCalls != 1 {
		t.Fatalf("wrapper MTPMetrics = %+v, want native MTP counters", metrics)
	}
	if target.Metrics().GeneratedTokens != 2 {
		t.Fatalf("target generated metrics = %+v, want generated token count 2", target.Metrics())
	}
	if targetSession.closeCalls != 1 || draftSession.closeCalls != 1 {
		t.Fatalf("wrapper close calls target/draft = %d/%d, want 1/1", targetSession.closeCalls, draftSession.closeCalls)
	}
}

func TestNativeSpeculativeTextModelChatUsesConversationContinuity_Good(t *testing.T) {
	tok, err := pkgtokenizer.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	targetSession := newNativeSessionTextSession()
	draftSession := newNativeSessionTextSession()
	target := &nativeTextModel{
		tm:     &nativeSessionTextTokenModel{sessions: []*nativeSessionTextSession{targetSession}},
		tok:    tok,
		maxLen: 32,
	}
	draft := &nativeTextModel{
		tm:     &nativeSessionTextTokenModel{sessions: []*nativeSessionTextSession{draftSession}},
		tok:    tok,
		maxLen: 32,
	}
	spec := &nativeSpeculativeTextModel{nativeTextModel: target, draft: draft, draftTokens: 2}
	if _, err := EnableConversationContinuity(spec, ConversationContinuityOptions{Store: memvid.NewInMemoryStore(nil)}); err != nil {
		t.Fatalf("EnableConversationContinuity(native speculative): %v", err)
	}
	spec.setNativeMTPMetrics(&metal.MTPMetrics{ProposedTokens: 99})

	oldDecode := nativeTextMTPDecode
	defer func() { nativeTextMTPDecode = oldDecode }()
	var mtpCalled bool
	nativeTextMTPDecode = func(model.DecodeStepper, model.DecodeStepper, []int32, int, int, int, func(int32) bool) (*native.MTPResult, bool, error) {
		mtpCalled = true
		return &native.MTPResult{}, true, nil
	}

	var out []inference.Token
	for token := range spec.Chat(context.Background(), []inference.Message{{Role: "user", Content: "hello"}}, inference.WithMaxTokens(1), inference.WithTemperature(0)) {
		out = append(out, token)
	}
	if err := resultError(spec.Err()); err != nil {
		t.Fatalf("Chat Err() = %v", err)
	}
	if mtpCalled {
		t.Fatal("native speculative Chat entered MTP despite accepted conversation continuity")
	}
	if len(out) != 1 {
		t.Fatalf("Chat yielded %d tokens, want continuity-generated token", len(out))
	}
	if metrics := spec.MTPMetrics(); metrics != nil {
		t.Fatalf("MTPMetrics = %+v after continuity turn, want nil", metrics)
	}
	stats := target.continuity.Stats()
	if stats.FreshConversations != 1 || stats.StatelessFallbacks != 0 {
		t.Fatalf("continuity stats = %+v, want one fresh conversation and no fallback", stats)
	}
}

func TestNativeSpeculativeTextModelGenerateUsesNativeGemma4Assistant(t *testing.T) {
	tok, err := pkgtokenizer.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	targetSession := newNativeSessionTextSession()
	target := &nativeTextModel{
		tm:     &nativeSessionTextTokenModel{sessions: []*nativeSessionTextSession{targetSession}},
		tok:    tok,
		maxLen: 32,
	}
	pair := &native.AssistantPair{}
	spec := &nativeSpeculativeTextModel{nativeTextModel: target, nativeAssistant: pair, draftTokens: 3}
	if !IsSpeculativeTextModel(spec) {
		t.Fatal("IsSpeculativeTextModel(native assistant wrapper) = false, want true")
	}
	oldDecode := nativeTextAssistantDecode
	defer func() { nativeTextAssistantDecode = oldDecode }()
	var called bool
	var order []string
	nativeTextAssistantDecode = func(gotPair *native.AssistantPair, target model.DecodeStepper, prompt []int32, maxNew, eos, draftTokens int, suppress []int32, yield func(int32) bool) (native.AssistantGenerateResult, bool, error) {
		called = true
		if gotPair != pair {
			t.Fatal("native assistant hook did not receive wrapper pair")
		}
		if target != targetSession {
			t.Fatalf("native assistant target session = %T, want opened target session", target)
		}
		if !reflect.DeepEqual(prompt, []int32{0, 10}) || maxNew != 4 || draftTokens != 3 {
			t.Fatalf("native assistant args prompt=%v maxNew=%d draft=%d, want [0 10]/4/3", prompt, maxNew, draftTokens)
		}
		if len(suppress) != 0 {
			t.Fatalf("native assistant suppress = %v, want none", suppress)
		}
		if yield == nil {
			t.Fatal("native assistant yield = nil, want streaming sink")
		}
		order = append(order, "hook-before")
		if !yield(10) || !yield(10) || !yield(10) {
			t.Fatal("native assistant yield returned false")
		}
		order = append(order, "hook-after")
		return native.AssistantGenerateResult{
			Tokens:             []int32{10, 10, 10},
			PromptTokens:       2,
			TargetTokens:       3,
			DraftTokens:        5,
			AcceptedTokens:     4,
			RejectedTokens:     1,
			TargetVerifyCalls:  2,
			TargetCalls:        2,
			DraftCalls:         2,
			DraftTokenSchedule: []int{3, 2},
		}, true, nil
	}

	var out []inference.Token
	for token := range spec.Generate(context.Background(), "hello", inference.WithMaxTokens(4), inference.WithTemperature(0)) {
		order = append(order, "caller")
		out = append(out, token)
	}
	if !called {
		t.Fatal("native speculative wrapper did not call native Gemma 4 assistant")
	}
	if got := nativeSessionTextTokenText(out); got != " hello hello hello" {
		t.Fatalf("native assistant wrapper output = %q, want tokenizer-decoded text", got)
	}
	if !reflect.DeepEqual(order, []string{"hook-before", "caller", "caller", "caller", "hook-after"}) {
		t.Fatalf("native assistant streaming order = %v, want hook to yield directly into caller", order)
	}
	metrics := spec.MTPMetrics()
	if metrics == nil {
		t.Fatal("native assistant MTPMetrics = nil, want acceptance counters")
	}
	if metrics.ProposedTokens != 5 || metrics.AcceptedTokens != 4 || metrics.RejectedTokens != 1 || metrics.TargetVerifyCalls != 2 || metrics.DraftCalls != 2 {
		t.Fatalf("native assistant MTPMetrics = %+v, want assistant counters", metrics)
	}
	if target.Metrics().GeneratedTokens != 3 {
		t.Fatalf("target generated metrics = %+v, want generated token count 3", target.Metrics())
	}
	if targetSession.closeCalls != 1 {
		t.Fatalf("target session close calls = %d, want 1", targetSession.closeCalls)
	}
	if err := resultError(spec.Close()); err != nil {
		t.Fatalf("Close: %v", err)
	}
}

func TestNativeSpeculativeTextModelGemma4AssistantRepeatPenaltyFallsBack(t *testing.T) {
	tok, err := pkgtokenizer.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	target := &nativeTextModel{
		tm:     &nativeSessionTextTokenModel{},
		tok:    tok,
		maxLen: 32,
	}
	spec := &nativeSpeculativeTextModel{
		nativeTextModel: target,
		nativeAssistant: &native.AssistantPair{},
		draftTokens:     3,
	}
	oldSampled := nativeTextAssistantSampledDecode
	defer func() { nativeTextAssistantSampledDecode = oldSampled }()
	var assistantCalled bool
	nativeTextAssistantSampledDecode = func(*native.AssistantPair, model.DecodeStepper, []int32, int, []int32, *model.Sampler, model.SampleParams, int, func(int32) bool) (native.AssistantGenerateResult, bool, error) {
		assistantCalled = true
		return native.AssistantGenerateResult{}, true, core.NewError("assistant sampled path should not run for repeat penalty")
	}

	var out []inference.Token
	for token := range spec.Generate(context.Background(), "hello", inference.WithMaxTokens(2), inference.WithTemperature(0), inference.WithRepeatPenalty(2)) {
		out = append(out, token)
	}
	if err := resultError(spec.Err()); err != nil {
		t.Fatalf("repeat-penalty fallback Err = %v", err)
	}
	if assistantCalled {
		t.Fatal("native assistant sampled verifier ran for repeat-penalty request; want plain target fallback")
	}
	if got := nativeSessionTextTokenText(out); got != "eh" {
		t.Fatalf("repeat-penalty fallback output = %q, want plain target sampled output", got)
	}
	if metrics := spec.MTPMetrics(); metrics != nil {
		t.Fatalf("repeat-penalty fallback MTPMetrics = %+v, want nil", metrics)
	}
	if target.Metrics().GeneratedTokens != 2 {
		t.Fatalf("repeat-penalty fallback target metrics = %+v, want 2 generated tokens", target.Metrics())
	}
}

func TestNativeSpeculativeTextModelGemma4AssistantProbeSinkFallsBack(t *testing.T) {
	tok, err := pkgtokenizer.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	target := &nativeTextModel{
		tm:     &nativeSessionTextTokenModel{},
		tok:    tok,
		maxLen: 32,
	}
	spec := &nativeSpeculativeTextModel{
		nativeTextModel: target,
		nativeAssistant: &native.AssistantPair{},
		draftTokens:     3,
	}
	oldDecode := nativeTextAssistantDecode
	defer func() { nativeTextAssistantDecode = oldDecode }()
	var assistantCalled bool
	nativeTextAssistantDecode = func(*native.AssistantPair, model.DecodeStepper, []int32, int, int, int, []int32, func(int32) bool) (native.AssistantGenerateResult, bool, error) {
		assistantCalled = true
		return native.AssistantGenerateResult{}, true, core.NewError("assistant greedy path should not run for probe sink")
	}
	spec.SetProbeSink(inference.ProbeSinkFunc(func(inference.ProbeEvent) {}))

	var out []inference.Token
	for token := range spec.Generate(context.Background(), "hello", inference.WithMaxTokens(2), inference.WithTemperature(0)) {
		out = append(out, token)
	}
	if err := resultError(spec.Err()); err != nil {
		t.Fatalf("probe-sink fallback Err = %v", err)
	}
	if assistantCalled {
		t.Fatal("native assistant verifier ran for probe-sink request; want plain target fallback")
	}
	if got := nativeSessionTextTokenText(out); got != "ee" {
		t.Fatalf("probe-sink fallback output = %q, want plain target greedy output", got)
	}
	if metrics := spec.MTPMetrics(); metrics != nil {
		t.Fatalf("probe-sink fallback MTPMetrics = %+v, want nil", metrics)
	}
	if target.Metrics().GeneratedTokens != 2 {
		t.Fatalf("probe-sink fallback target metrics = %+v, want 2 generated tokens", target.Metrics())
	}
}

func TestNativeSpeculativeTextModelGemma4AssistantUsesWarmPromptCache(t *testing.T) {
	tok, err := pkgtokenizer.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	targetSession := newNativeSessionTextSession()
	targetTM := &nativeSessionTextTokenModel{sessions: []*nativeSessionTextSession{targetSession}}
	target := &nativeTextModel{
		tm:     targetTM,
		tok:    tok,
		maxLen: 32,
	}
	spec := &nativeSpeculativeTextModel{
		nativeTextModel: target,
		nativeAssistant: &native.AssistantPair{},
		draftTokens:     3,
	}
	if err := spec.WarmPromptCache(context.Background(), "hello"); err != nil {
		t.Fatalf("WarmPromptCache: %v", err)
	}
	if targetTM.opens != 1 {
		t.Fatalf("OpenSession calls after WarmPromptCache = %d, want 1", targetTM.opens)
	}

	oldDecode := nativeTextAssistantDecode
	defer func() { nativeTextAssistantDecode = oldDecode }()
	nativeTextAssistantDecode = func(_ *native.AssistantPair, gotTarget model.DecodeStepper, prompt []int32, maxNew, _ int, draftTokens int, _ []int32, yield func(int32) bool) (native.AssistantGenerateResult, bool, error) {
		if gotTarget != targetSession {
			t.Fatalf("native assistant target session = %p, want warmed prompt-cache session %p", gotTarget, targetSession)
		}
		if !reflect.DeepEqual(prompt, []int32{0, 10}) || maxNew != 1 || draftTokens != 3 {
			t.Fatalf("native assistant args prompt=%v maxNew=%d draft=%d, want [0 10]/1/3", prompt, maxNew, draftTokens)
		}
		if yield != nil && !yield(10) {
			t.Fatal("native assistant warm-cache yield returned false")
		}
		return native.AssistantGenerateResult{
			Tokens:            []int32{10},
			PromptTokens:      len(prompt),
			TargetTokens:      1,
			DraftTokens:       1,
			AcceptedTokens:    1,
			TargetVerifyCalls: 1,
			TargetCalls:       1,
			DraftCalls:        1,
		}, true, nil
	}

	var out []inference.Token
	for token := range spec.Generate(context.Background(), "hello", inference.WithMaxTokens(1), inference.WithTemperature(0)) {
		out = append(out, token)
	}
	if err := resultError(spec.Err()); err != nil {
		t.Fatalf("Generate Err = %v", err)
	}
	if got := nativeSessionTextTokenText(out); got != " hello" {
		t.Fatalf("warm-cache assistant output = %q, want tokenizer-decoded assistant output", got)
	}
	if targetTM.opens != 1 {
		t.Fatalf("OpenSession calls after assistant generate = %d, want warmed session reuse only", targetTM.opens)
	}
	if targetSession.closeCalls != 0 {
		t.Fatalf("warmed prompt-cache session close calls = %d, want 0", targetSession.closeCalls)
	}
}

func TestNativeSpeculativeTextModelGemma4AssistantUsesWarmPromptPrefix(t *testing.T) {
	tok, err := pkgtokenizer.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	targetSession := newNativeSessionTextSession()
	targetTM := &nativeSessionTextTokenModel{sessions: []*nativeSessionTextSession{targetSession}}
	target := &nativeTextModel{
		tm:     targetTM,
		tok:    tok,
		maxLen: 32,
	}
	spec := &nativeSpeculativeTextModel{
		nativeTextModel: target,
		nativeAssistant: &native.AssistantPair{},
		draftTokens:     3,
	}
	if err := spec.WarmPromptCache(context.Background(), "hello"); err != nil {
		t.Fatalf("WarmPromptCache: %v", err)
	}
	if targetTM.opens != 1 {
		t.Fatalf("OpenSession calls after WarmPromptCache = %d, want 1", targetTM.opens)
	}

	promptIDs := tok.Encode("hello hello")
	oldDecode := nativeTextAssistantDecode
	defer func() { nativeTextAssistantDecode = oldDecode }()
	nativeTextAssistantDecode = func(_ *native.AssistantPair, gotTarget model.DecodeStepper, prompt []int32, maxNew, _ int, draftTokens int, _ []int32, yield func(int32) bool) (native.AssistantGenerateResult, bool, error) {
		if gotTarget != targetSession {
			t.Fatalf("native assistant target session = %p, want shared-prefix prompt-cache session %p", gotTarget, targetSession)
		}
		if !reflect.DeepEqual(prompt, promptIDs) || maxNew != 1 || draftTokens != 3 {
			t.Fatalf("native assistant args prompt=%v maxNew=%d draft=%d, want %v/1/3", prompt, maxNew, draftTokens, promptIDs)
		}
		if yield != nil && !yield(10) {
			t.Fatal("native assistant prefix-cache yield returned false")
		}
		return native.AssistantGenerateResult{
			Tokens:            []int32{10},
			PromptTokens:      len(prompt),
			TargetTokens:      1,
			DraftTokens:       1,
			AcceptedTokens:    1,
			TargetVerifyCalls: 1,
			TargetCalls:       1,
			DraftCalls:        1,
		}, true, nil
	}

	for range spec.Generate(context.Background(), "hello hello", inference.WithMaxTokens(1), inference.WithTemperature(0)) {
	}
	if err := resultError(spec.Err()); err != nil {
		t.Fatalf("Generate Err = %v", err)
	}
	if targetTM.opens != 1 {
		t.Fatalf("OpenSession calls after shared-prefix assistant generate = %d, want warmed session reuse only", targetTM.opens)
	}
}

func TestNativeSpeculativeTextModelGemma4AssistantUpdatesWarmPromptPrefixCacheEntry(t *testing.T) {
	tok, err := pkgtokenizer.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	targetSession := newNativeSessionTextSession()
	target := &nativeTextModel{
		tm:     &nativeSessionTextTokenModel{sessions: []*nativeSessionTextSession{targetSession}},
		tok:    tok,
		maxLen: 32,
	}
	spec := &nativeSpeculativeTextModel{
		nativeTextModel: target,
		nativeAssistant: &native.AssistantPair{},
		draftTokens:     3,
	}
	if err := spec.WarmPromptCache(context.Background(), "hello"); err != nil {
		t.Fatalf("WarmPromptCache: %v", err)
	}
	promptIDs := tok.Encode("hello hello")

	oldDecode := nativeTextAssistantDecode
	defer func() { nativeTextAssistantDecode = oldDecode }()
	nativeTextAssistantDecode = func(_ *native.AssistantPair, gotTarget model.DecodeStepper, prompt []int32, _, _ int, _ int, _ []int32, yield func(int32) bool) (native.AssistantGenerateResult, bool, error) {
		if gotTarget != targetSession {
			t.Fatalf("native assistant target session = %p, want warmed prompt-cache session %p", gotTarget, targetSession)
		}
		if !reflect.DeepEqual(prompt, promptIDs) {
			t.Fatalf("native assistant prompt = %v, want %v", prompt, promptIDs)
		}
		if yield != nil && !yield(10) {
			t.Fatal("native assistant cache-entry yield returned false")
		}
		return native.AssistantGenerateResult{
			Tokens:            []int32{10},
			PromptTokens:      len(prompt),
			TargetTokens:      1,
			DraftTokens:       1,
			AcceptedTokens:    1,
			TargetVerifyCalls: 1,
			TargetCalls:       1,
			DraftCalls:        1,
		}, true, nil
	}

	for range spec.Generate(context.Background(), "hello hello", inference.WithMaxTokens(1), inference.WithTemperature(0)) {
	}
	if err := resultError(spec.Err()); err != nil {
		t.Fatalf("Generate Err = %v", err)
	}
	entries, err := spec.CacheEntries(context.Background(), nil)
	if err != nil {
		t.Fatalf("CacheEntries: %v", err)
	}
	if len(entries) != 1 || entries[0].TokenCount != len(promptIDs) {
		t.Fatalf("CacheEntries after assistant prefix generate = %+v, want one full-prompt block with %d tokens", entries, len(promptIDs))
	}
}

func TestNativeSpeculativeAssistantTokenizerValidation(t *testing.T) {
	target, err := pkgtokenizer.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer(target): %v", err)
	}
	if err := validateNativeSpeculativeAssistantTokenizer(target, writeNativeAssistantTokenizerDir(t, rootTokenizerJSON)); err != nil {
		t.Fatalf("validateNativeSpeculativeAssistantTokenizer(matching): %v", err)
	}
	err = validateNativeSpeculativeAssistantTokenizer(target, writeNativeAssistantTokenizerDir(t, rootTokenizerWithoutBOSJSON))
	if err != errMLXSpeculativeTokenizersDiffer {
		t.Fatalf("validateNativeSpeculativeAssistantTokenizer(mismatch) = %v, want tokenizer mismatch", err)
	}
}

func writeNativeAssistantTokenizerDir(t *testing.T, body string) string {
	t.Helper()
	dir := t.TempDir()
	path := core.PathJoin(dir, "tokenizer.json")
	if result := core.WriteFile(path, []byte(body), 0o644); !result.OK {
		t.Fatalf("write assistant tokenizer: %v", result.Value)
	}
	return dir
}

func nativeSessionTextTokenText(tokens []inference.Token) string {
	var out core.Builder
	for _, token := range tokens {
		out.WriteString(token.Text)
	}
	return out.String()
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
