// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"encoding/binary"
	"reflect"
	"testing"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/mlx/kv"
	"dappco.re/go/mlx/pkg/model"
	g4 "dappco.re/go/mlx/pkg/model/gemma4"
)

func sessionStateFixture(t testing.TB) (*BF16Model, model.Arch, int) {
	t.Helper()
	const dModel, nHeads, nKV, headDim, dFF = 512, 8, 4, 64, 1024
	const vocab, nL, maxLen = 64, 3, 96
	layers := make([]DecodeLayerWeights, nL)
	types := make([]string, nL)
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
		types[li] = "full_attention"
	}
	specs := model.DeriveLayers(types, 0)
	embed := toBF16Bytes(syntheticFloat32(vocab*dModel, 21))
	g := &BF16Model{Layers: layers, Embed: embed, FinalNorm: toBF16Bytes(syntheticFloat32(dModel, 22)), LMHead: embed, Tied: true}
	arch := model.Arch{
		Hidden: dModel, Heads: nHeads, KVHeads: nKV, HeadDim: headDim, FF: dFF, Vocab: vocab,
		GlobalHeadDim: headDim, GlobalKVHeads: nKV,
		Eps: 1e-5, AttnScale: 0.125, RopeBase: 10000, RopeScale: 1, RopeLocalBase: 10000,
		RotaryDim: headDim, RotaryDimLocal: headDim, Layer: specs,
	}
	return g, arch, maxLen
}

func newSessionStateFixture(t testing.TB) *ArchSession {
	t.Helper()
	g, arch, maxLen := sessionStateFixture(t)
	s, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	return s
}

func icbSessionStateFixture(t testing.TB) (*QuantModel, model.Arch, int) {
	t.Helper()
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 32
	const numLayers, pliDim, gs, bits = 2, 64, 64, 4
	const maxLen = 24
	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: numLayers, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim,
		VocabSize: vocab, RMSNormEps: 1e-6,
		HiddenSizePerLayerInput: pliDim, VocabSizePerLayerInput: vocab,
		Quantization: &model.QuantConfig{GroupSize: gs, Bits: bits},
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	ts := quantGemma4Tensors(t, arch, gs, bits)
	addPLETensors(t, ts, arch, gs, bits)
	lm, err := model.Assemble(ts, arch, model.StandardWeightNames())
	if err != nil {
		t.Fatalf("model.Assemble: %v", err)
	}
	g, err := loadedToQuant(lm, gs, bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	return g, arch, maxLen
}

func newICBSessionStateFixture(t testing.TB, g *QuantModel, arch model.Arch, maxLen int) *ArchSession {
	t.Helper()
	s, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession: %v", err)
	}
	if s.state.icb == nil {
		t.Fatal("fixture must build an ICB replay session")
	}
	return s
}

func firstOwnedCacheLayer(t testing.TB, s *ArchSession) int {
	t.Helper()
	for li, spec := range s.state.specs {
		if spec.OwnsCache() {
			return li
		}
	}
	t.Fatal("fixture has no cache-owning layer")
	return 0
}

func emptySessionStateBlob(pos, layers, cachedIDs int) []byte {
	blob := make([]byte, 12+4+4*cachedIDs)
	binary.LittleEndian.PutUint32(blob[0:], sessionStateMagic)
	binary.LittleEndian.PutUint32(blob[4:], uint32(pos))
	binary.LittleEndian.PutUint32(blob[8:], uint32(layers))
	binary.LittleEndian.PutUint32(blob[12:], uint32(cachedIDs))
	for i := 0; i < cachedIDs; i++ {
		binary.LittleEndian.PutUint32(blob[16+4*i:], uint32(i+1))
	}
	return blob
}

func TestSessionStateSnapshotCacheViewsUseCachedContentsPointers(t *testing.T) {
	requireNativeRuntime(t)

	t.Run("layer buffers", func(t *testing.T) {
		s := newSessionStateFixture(t)
		li := firstOwnedCacheLayer(t, s)
		if s.state.lb[li].kCachePtr == nil || s.state.lb[li].vCachePtr == nil {
			t.Fatal("layer KV cache contents pointers were not cached at construction")
		}
		k, v, kPtr, vPtr, err := s.snapshotCacheViews(li)
		if err != nil {
			t.Fatalf("snapshotCacheViews: %v", err)
		}
		if k != s.state.lb[li].kCache || v != s.state.lb[li].vCache {
			t.Fatal("snapshotCacheViews returned unexpected layer cache buffers")
		}
		if kPtr != s.state.lb[li].kCachePtr || vPtr != s.state.lb[li].vCachePtr {
			t.Fatal("snapshotCacheViews did not return cached layer cache pointers")
		}
		if kPtr != (*byte)(k.Contents()) || vPtr != (*byte)(v.Contents()) {
			t.Fatal("cached layer cache pointers do not reference Metal buffer contents")
		}
	})

	t.Run("icb replay", func(t *testing.T) {
		g, arch, maxLen := icbSessionStateFixture(t)
		s := newICBSessionStateFixture(t, g, arch, maxLen)
		li := firstOwnedCacheLayer(t, s)
		if len(s.state.icb.kCachePtrs) != len(s.state.icb.kCaches) || len(s.state.icb.vCachePtrs) != len(s.state.icb.vCaches) {
			t.Fatal("ICB KV cache pointer slices do not match cache slices")
		}
		if s.state.icb.kCachePtrs[li] == nil || s.state.icb.vCachePtrs[li] == nil {
			t.Fatal("ICB KV cache contents pointers were not cached at record time")
		}
		k, v, kPtr, vPtr, err := s.snapshotCacheViews(li)
		if err != nil {
			t.Fatalf("snapshotCacheViews ICB: %v", err)
		}
		if k != s.state.icb.kCaches[li] || v != s.state.icb.vCaches[li] {
			t.Fatal("snapshotCacheViews returned unexpected ICB cache buffers")
		}
		if kPtr != s.state.icb.kCachePtrs[li] || vPtr != s.state.icb.vCachePtrs[li] {
			t.Fatal("snapshotCacheViews did not return cached ICB cache pointers")
		}
		if kPtr != (*byte)(k.Contents()) || vPtr != (*byte)(v.Contents()) {
			t.Fatal("cached ICB cache pointers do not reference Metal buffer contents")
		}
	})
}

func TestSessionStateBlocksRestoreGenerateFromCacheBoundary(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	source, err := saved.StateBlockSource(2)
	if err != nil {
		t.Fatalf("StateBlockSource: %v", err)
	}
	if source.Position != saved.Pos() {
		t.Fatalf("block source position = %d, want %d", source.Position, saved.Pos())
	}
	if source.BlockCount != 3 {
		t.Fatalf("block source count = %d, want 3", source.BlockCount)
	}
	if !idsEqual(source.CachedIDs, prompt) {
		t.Fatalf("block source cached ids = %v, want %v", source.CachedIDs, prompt)
	}

	rangedTokens := 0
	if err := saved.RangeStateBlocks(2, func(block SessionStateBlock) (bool, error) {
		if block.TokenCount <= 0 {
			t.Fatalf("block %d token count = %d, want > 0", block.Index, block.TokenCount)
		}
		if len(block.Layers) == 0 {
			t.Fatalf("block %d has no layer payloads", block.Index)
		}
		for _, layer := range block.Layers {
			wantBytes := layer.RowBytes * block.TokenCount
			if len(layer.KeyBytes) != wantBytes || len(layer.ValueBytes) != wantBytes {
				t.Fatalf("block %d layer %d bytes = %d/%d, want %d", block.Index, layer.Layer, len(layer.KeyBytes), len(layer.ValueBytes), wantBytes)
			}
		}
		rangedTokens += block.TokenCount
		return true, nil
	}); err != nil {
		t.Fatalf("RangeStateBlocks: %v", err)
	}
	if rangedTokens != saved.Pos() {
		t.Fatalf("ranged tokens = %d, want %d", rangedTokens, saved.Pos())
	}

	restored := newSessionStateFixture(t)
	if err := restored.RestoreStateBlocks(source); err != nil {
		t.Fatalf("RestoreStateBlocks: %v", err)
	}
	if restored.Pos() != saved.Pos() {
		t.Fatalf("restored pos = %d, want %d", restored.Pos(), saved.Pos())
	}
	if !idsEqual(restored.cachedIDs, prompt) {
		t.Fatalf("restored cached ids = %v, want %v", restored.cachedIDs, prompt)
	}
	got, err := restored.GenerateFromCache(3, -1)
	if err != nil {
		t.Fatalf("GenerateFromCache after block restore: %v", err)
	}
	cold := newSessionStateFixture(t)
	want, err := cold.Generate(prompt, 3, -1)
	if err != nil {
		t.Fatalf("cold Generate: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("block-restored GenerateFromCache = %v, want cold prompt continuation %v", got, want)
	}
}

func TestSessionStateBlocksRestoreGenerateFromBoundaryLogits(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	logits, err := saved.BoundaryLogits()
	if err != nil {
		t.Fatalf("BoundaryLogits: %v", err)
	}
	source, err := saved.StateBlockSource(2)
	if err != nil {
		t.Fatalf("StateBlockSource: %v", err)
	}

	restored := newSessionStateFixture(t)
	if err := restored.RestoreStateBlocks(source); err != nil {
		t.Fatalf("RestoreStateBlocks: %v", err)
	}
	restored.resetRetainedHidden()
	got, err := restored.GenerateFromCacheLogitsEach(logits, 3, -1, nil)
	if err != nil {
		t.Fatalf("GenerateFromCacheLogitsEach after block restore: %v", err)
	}
	cold := newSessionStateFixture(t)
	want, err := cold.Generate(prompt, 3, -1)
	if err != nil {
		t.Fatalf("cold Generate: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("logit-restored GenerateFromCache = %v, want cold prompt continuation %v", got, want)
	}
	if restored.Pos() != len(prompt)+len(got) {
		t.Fatalf("restored pos after logit continuation = %d, want %d", restored.Pos(), len(prompt)+len(got))
	}
}

func TestSessionStateBlocksRestoreGenerateSampledFromBoundaryLogits(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}
	params := model.SampleParams{Temperature: 0.8, TopK: 4, TopP: 0.9, MinTokensBeforeStop: 1}
	stopTokens := []int32{63}
	const seed = 0x5eed1234

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	logits, err := saved.BoundaryLogits()
	if err != nil {
		t.Fatalf("BoundaryLogits: %v", err)
	}
	source, err := saved.StateBlockSource(2)
	if err != nil {
		t.Fatalf("StateBlockSource: %v", err)
	}

	restored := newSessionStateFixture(t)
	if err := restored.RestoreStateBlocks(source); err != nil {
		t.Fatalf("RestoreStateBlocks: %v", err)
	}
	restored.resetRetainedHidden()
	got, err := restored.GenerateSampledFromCacheLogitsEach(logits, 3, stopTokens, model.NewSampler(seed), params, nil, nil)
	if err != nil {
		t.Fatalf("GenerateSampledFromCacheLogitsEach after block restore: %v", err)
	}
	cold := newSessionStateFixture(t)
	want, err := cold.GenerateSampledEach(prompt, 3, stopTokens, model.NewSampler(seed), params, nil, nil)
	if err != nil {
		t.Fatalf("cold GenerateSampledEach: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("sampled logit-restored GenerateFromCache = %v, want cold prompt continuation %v", got, want)
	}
	if restored.Pos() != len(prompt)+len(got) {
		t.Fatalf("restored pos after sampled logit continuation = %d, want %d", restored.Pos(), len(prompt)+len(got))
	}
}

func TestSessionStateBlocksRestoreGenerateSampledFromRetainedHidden(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}
	params := model.SampleParams{Temperature: 0.8, TopK: 4, TopP: 0.9, MinTokensBeforeStop: 1}
	stopTokens := []int32{63}
	const seed = 0x5eed1234

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	source, err := saved.StateBlockSource(2)
	if err != nil {
		t.Fatalf("StateBlockSource: %v", err)
	}
	if len(source.RetainedHidden) == 0 {
		t.Fatal("StateBlockSource retained hidden is empty")
	}

	restored := newSessionStateFixture(t)
	if err := restored.RestoreStateBlocks(source); err != nil {
		t.Fatalf("RestoreStateBlocks: %v", err)
	}
	got, err := restored.GenerateSampledFromCacheEach(3, stopTokens, model.NewSampler(seed), params, nil, nil)
	if err != nil {
		t.Fatalf("GenerateSampledFromCacheEach after block restore: %v", err)
	}
	cold := newSessionStateFixture(t)
	want, err := cold.GenerateSampledEach(prompt, 3, stopTokens, model.NewSampler(seed), params, nil, nil)
	if err != nil {
		t.Fatalf("cold GenerateSampledEach: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("sampled retained-hidden GenerateFromCache = %v, want cold prompt continuation %v", got, want)
	}
	if restored.Pos() != len(prompt)+len(got) {
		t.Fatalf("restored pos after sampled retained-hidden continuation = %d, want %d", restored.Pos(), len(prompt)+len(got))
	}
}

func TestSessionStateBlocksRestoreGenerateSampledFromRetainedLogits(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}
	params := model.SampleParams{Temperature: 0.8, TopK: 4, TopP: 0.9, MinTokensBeforeStop: 1}
	stopTokens := []int32{63}
	const seed = 0x5eed1234

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	logits, err := saved.BoundaryLogits()
	if err != nil {
		t.Fatalf("BoundaryLogits: %v", err)
	}
	source, err := saved.StateBlockSource(2)
	if err != nil {
		t.Fatalf("StateBlockSource: %v", err)
	}
	if !bytes.Equal(source.RetainedLogits, logits) {
		t.Fatal("StateBlockSource did not carry retained boundary logits")
	}
	source.RetainedHidden = nil

	restored := newSessionStateFixture(t)
	if err := restored.RestoreStateBlocks(source); err != nil {
		t.Fatalf("RestoreStateBlocks: %v", err)
	}
	got, err := restored.GenerateSampledFromCacheEach(3, stopTokens, model.NewSampler(seed), params, nil, nil)
	if err != nil {
		t.Fatalf("GenerateSampledFromCacheEach after logit-only block restore: %v", err)
	}
	cold := newSessionStateFixture(t)
	want, err := cold.GenerateSampledEach(prompt, 3, stopTokens, model.NewSampler(seed), params, nil, nil)
	if err != nil {
		t.Fatalf("cold GenerateSampledEach: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("sampled retained-logit GenerateFromCache = %v, want cold prompt continuation %v", got, want)
	}
}

func TestSessionStateBlocksGenerateSampledFromRetainedHiddenAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}
	params := model.SampleParams{Temperature: 0.8, TopK: 4, TopP: 0.9}
	stopTokens := []int32{63}

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	source, err := saved.StateBlockSource(2)
	if err != nil {
		t.Fatalf("StateBlockSource: %v", err)
	}
	restored := newSessionStateFixture(t)
	if err := restored.RestoreStateBlocks(source); err != nil {
		t.Fatalf("RestoreStateBlocks warmup: %v", err)
	}
	if _, err := restored.GenerateSampledFromCacheEach(1, stopTokens, model.NewSampler(1), params, nil, nil); err != nil {
		t.Fatalf("GenerateSampledFromCacheEach warmup: %v", err)
	}

	seed := uint64(10)
	allocs := testing.AllocsPerRun(5, func() {
		if err := restored.RestoreStateBlocks(source); err != nil {
			t.Fatalf("RestoreStateBlocks: %v", err)
		}
		seed++
		if _, err := restored.GenerateSampledFromCacheEach(1, stopTokens, model.NewSampler(seed), params, nil, nil); err != nil {
			t.Fatalf("GenerateSampledFromCacheEach: %v", err)
		}
	})
	if allocs > 120 {
		t.Fatalf("restored retained-hidden sampled wake allocations = %.0f, want <= 120", allocs)
	}
}

func TestSessionStateBlocksGenerateSampledFromRetainedLogitsAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}
	params := model.SampleParams{Temperature: 0.8, TopK: 4, TopP: 0.9}
	stopTokens := []int32{63}

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	source, err := saved.StateBlockSource(2)
	if err != nil {
		t.Fatalf("StateBlockSource: %v", err)
	}
	if len(source.RetainedLogits) == 0 {
		t.Fatal("StateBlockSource did not retain boundary logits")
	}
	source.RetainedHidden = nil
	restored := newSessionStateFixture(t)
	if err := restored.RestoreStateBlocks(source); err != nil {
		t.Fatalf("RestoreStateBlocks warmup: %v", err)
	}
	if _, err := restored.GenerateSampledFromCacheEach(1, stopTokens, model.NewSampler(1), params, nil, nil); err != nil {
		t.Fatalf("GenerateSampledFromCacheEach warmup: %v", err)
	}

	seed := uint64(20)
	allocs := testing.AllocsPerRun(5, func() {
		if err := restored.RestoreStateBlocks(source); err != nil {
			t.Fatalf("RestoreStateBlocks: %v", err)
		}
		seed++
		if _, err := restored.GenerateSampledFromCacheEach(1, stopTokens, model.NewSampler(seed), params, nil, nil); err != nil {
			t.Fatalf("GenerateSampledFromCacheEach: %v", err)
		}
	})
	if allocs > 20 {
		t.Fatalf("restored retained-logit sampled wake allocations = %.0f, want <= 20", allocs)
	}
}

func TestSessionStateBlocksGenerateSampledFromRetainedHiddenTopPOnlyAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}
	params := model.SampleParams{Temperature: 1, TopP: 0.72}
	stopTokens := []int32{63}

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	source, err := saved.StateBlockSource(2)
	if err != nil {
		t.Fatalf("StateBlockSource: %v", err)
	}
	source.RetainedLogits = nil
	restored := newSessionStateFixture(t)
	if err := restored.RestoreStateBlocks(source); err != nil {
		t.Fatalf("RestoreStateBlocks warmup: %v", err)
	}
	if _, err := restored.GenerateSampledFromCacheEach(1, stopTokens, model.NewSampler(1), params, nil, nil); err != nil {
		t.Fatalf("GenerateSampledFromCacheEach warmup: %v", err)
	}

	seed := uint64(30)
	allocs := testing.AllocsPerRun(5, func() {
		if err := restored.RestoreStateBlocks(source); err != nil {
			t.Fatalf("RestoreStateBlocks: %v", err)
		}
		seed++
		if _, err := restored.GenerateSampledFromCacheEach(1, stopTokens, model.NewSampler(seed), params, nil, nil); err != nil {
			t.Fatalf("GenerateSampledFromCacheEach: %v", err)
		}
	})
	if allocs > 25 {
		t.Fatalf("restored retained-hidden TopP-only sampled wake allocations = %.0f, want <= 25", allocs)
	}
}

func TestSessionStateBlocksGenerateSampledFromRetainedLogitsTopPOnlyLargeVocabAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 128
	const maxLen = 24
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	saved, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession saved: %v", err)
	}
	prompt := []int32{1, 2, 3, 4, 5}
	params := model.SampleParams{Temperature: 1, TopP: 0.72}
	stopTokens := []int32{int32(vocab - 1)}
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	source, err := saved.StateBlockSource(2)
	if err != nil {
		t.Fatalf("StateBlockSource: %v", err)
	}
	if len(source.RetainedLogits) == 0 {
		t.Fatal("StateBlockSource did not retain boundary logits")
	}
	source.RetainedHidden = nil
	restored, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession restored: %v", err)
	}
	if err := restored.RestoreStateBlocks(source); err != nil {
		t.Fatalf("RestoreStateBlocks warmup: %v", err)
	}
	if _, err := restored.GenerateSampledFromCacheEach(1, stopTokens, model.NewSampler(1), params, nil, nil); err != nil {
		t.Fatalf("GenerateSampledFromCacheEach warmup: %v", err)
	}

	const paritySeed = 99
	if err := restored.RestoreStateBlocks(source); err != nil {
		t.Fatalf("RestoreStateBlocks parity: %v", err)
	}
	got, err := restored.GenerateSampledFromCacheEach(1, stopTokens, model.NewSampler(paritySeed), params, nil, nil)
	if err != nil {
		t.Fatalf("GenerateSampledFromCacheEach parity: %v", err)
	}
	cold, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession cold: %v", err)
	}
	want, err := cold.GenerateSampledEach(prompt, 1, stopTokens, model.NewSampler(paritySeed), params, nil, nil)
	if err != nil {
		t.Fatalf("cold GenerateSampledEach: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("retained-logit large-vocab TopP-only wake = %v, want cold sampled continuation %v", got, want)
	}

	seed := uint64(40)
	allocs := testing.AllocsPerRun(5, func() {
		if err := restored.RestoreStateBlocks(source); err != nil {
			t.Fatalf("RestoreStateBlocks: %v", err)
		}
		seed++
		if _, err := restored.GenerateSampledFromCacheEach(1, stopTokens, model.NewSampler(seed), params, nil, nil); err != nil {
			t.Fatalf("GenerateSampledFromCacheEach: %v", err)
		}
	})
	if allocs > 12 {
		t.Fatalf("restored retained-logit large-vocab TopP-only sampled wake allocations = %.0f, want <= 12", allocs)
	}
}

func TestArchSessionRetainedHiddenUsesPinnedNoCopyBuffer(t *testing.T) {
	requireNativeRuntime(t)
	sess := newSessionStateFixture(t)
	n := sess.arch.Hidden * bf16Size
	first := toBF16Bytes(syntheticFloat32(sess.arch.Hidden, 31))
	second := toBF16Bytes(syntheticFloat32(sess.arch.Hidden, 32))

	sess.rememberRetainedHidden(first)
	if sess.retainedHiddenPinned == nil || sess.retainedHiddenPinned.buf == nil {
		t.Fatal("retained hidden was not stored in a pinned no-copy buffer")
	}
	if len(sess.retainedHidden) != n || !bytes.Equal(sess.retainedHidden, first) {
		t.Fatal("retained hidden did not preserve first boundary contents")
	}
	if unsafe.Pointer(&sess.retainedHidden[0]) != unsafe.Pointer(&sess.retainedHiddenPinned.bytes[0]) {
		t.Fatal("retained hidden slice does not point at pinned backing bytes")
	}
	buf := sess.retainedHiddenBuffer()
	if buf == nil || buf.GetID() != sess.retainedHiddenPinned.buf.GetID() {
		t.Fatal("retainedHiddenBuffer did not return the session-owned no-copy buffer")
	}
	backing := unsafe.Pointer(&sess.retainedHidden[0])
	bufID := sess.retainedHiddenPinned.buf.GetID()

	sess.rememberRetainedHidden(second)
	if unsafe.Pointer(&sess.retainedHidden[0]) != backing {
		t.Fatal("retained hidden backing changed across same-shape boundary updates")
	}
	if sess.retainedHiddenPinned.buf.GetID() != bufID {
		t.Fatal("retained hidden no-copy buffer changed across same-shape boundary updates")
	}
	if !bytes.Equal(sess.retainedHidden, second) {
		t.Fatal("retained hidden did not refresh second boundary contents")
	}
}

func TestBoundaryLogitsUsesRetainedHiddenNoCopyHeadPath(t *testing.T) {
	requireNativeRuntime(t)
	sess := newSessionStateFixture(t)
	if sess.headEnc == nil {
		t.Fatal("session fixture did not build resident head encoder")
	}
	hidden := toBF16Bytes(syntheticFloat32(sess.arch.Hidden, 37))
	sess.rememberRetainedHidden(hidden)
	if sess.retainedHiddenBuffer() == nil {
		t.Fatal("retained hidden did not expose its pinned no-copy buffer")
	}
	sess.sampleHeadLogits = make([]byte, sess.arch.Vocab*bf16Size)

	head := sess.head
	headCalls := 0
	sess.head = func(hidden []byte, skipSoftcap bool) ([]byte, error) {
		headCalls++
		return head(hidden, skipSoftcap)
	}
	logits, err := sess.BoundaryLogits()
	if err != nil {
		t.Fatalf("BoundaryLogits: %v", err)
	}
	if len(logits) != sess.arch.Vocab*bf16Size {
		t.Fatalf("BoundaryLogits length = %d, want %d", len(logits), sess.arch.Vocab*bf16Size)
	}
	if headCalls != 0 {
		t.Fatalf("BoundaryLogits generic head calls = %d, want retained no-copy head path", headCalls)
	}
	if buf := sess.retainedLogitsBuffer(); buf == nil {
		t.Fatal("BoundaryLogits did not retain logits in a pinned no-copy buffer")
	}
	if len(sess.retainedLogits) == 0 || unsafe.Pointer(&logits[0]) != unsafe.Pointer(&sess.retainedLogits[0]) {
		t.Fatal("BoundaryLogits did not return retained logits backing")
	}
	if cap(sess.sampleHeadLogits) != 0 {
		t.Fatalf("BoundaryLogits retained transient head logits scratch cap = %d, want 0", cap(sess.sampleHeadLogits))
	}
	allocs := testing.AllocsPerRun(10, func() {
		sess.resetRetainedLogits()
		if _, err := sess.BoundaryLogits(); err != nil {
			t.Fatalf("BoundaryLogits allocation run: %v", err)
		}
	})
	if allocs > 1 {
		t.Fatalf("BoundaryLogits retained-hidden no-copy allocations = %.0f, want <= 1", allocs)
	}
}

func TestArchSessionRetainedLogitsUsesPinnedNoCopyBuffer(t *testing.T) {
	requireNativeRuntime(t)
	sess := newSessionStateFixture(t)
	n := sess.arch.Vocab * bf16Size
	first := toBF16Bytes(syntheticFloat32(sess.arch.Vocab, 41))
	second := toBF16Bytes(syntheticFloat32(sess.arch.Vocab, 42))

	sess.rememberRetainedLogits(first)
	if sess.retainedLogitsPinned == nil || sess.retainedLogitsPinned.buf == nil {
		t.Fatal("retained logits were not stored in a pinned no-copy buffer")
	}
	if len(sess.retainedLogits) != n || !bytes.Equal(sess.retainedLogits, first) {
		t.Fatal("retained logits did not preserve first boundary contents")
	}
	if unsafe.Pointer(&sess.retainedLogits[0]) != unsafe.Pointer(&sess.retainedLogitsPinned.bytes[0]) {
		t.Fatal("retained logits slice does not point at pinned backing bytes")
	}
	buf := sess.retainedLogitsBuffer()
	if buf == nil || buf.GetID() != sess.retainedLogitsPinned.buf.GetID() {
		t.Fatal("retainedLogitsBuffer did not return the session-owned no-copy buffer")
	}
	backing := unsafe.Pointer(&sess.retainedLogits[0])
	bufID := sess.retainedLogitsPinned.buf.GetID()

	sess.rememberRetainedLogits(second)
	if unsafe.Pointer(&sess.retainedLogits[0]) != backing {
		t.Fatal("retained logits backing changed across same-shape boundary updates")
	}
	if sess.retainedLogitsPinned.buf.GetID() != bufID {
		t.Fatal("retained logits no-copy buffer changed across same-shape boundary updates")
	}
	if !bytes.Equal(sess.retainedLogits, second) {
		t.Fatal("retained logits did not refresh second boundary contents")
	}
}

func TestSessionStateRangeBlocksSkipsTrustedPrefix(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5, 6, 7}

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	var got []SessionStateBlock
	if err := saved.RangeStateBlocksFrom(4, 2, func(block SessionStateBlock) (bool, error) {
		got = append(got, block)
		return true, nil
	}); err != nil {
		t.Fatalf("RangeStateBlocksFrom: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("ranged block count = %d, want 2", len(got))
	}
	if got[0].Index != 2 || got[0].TokenStart != 4 || got[0].TokenCount != 2 {
		t.Fatalf("first yielded block = index %d start %d count %d, want index 2 start 4 count 2", got[0].Index, got[0].TokenStart, got[0].TokenCount)
	}
	if got[1].Index != 3 || got[1].TokenStart != 6 || got[1].TokenCount != 1 {
		t.Fatalf("second yielded block = index %d start %d count %d, want index 3 start 6 count 1", got[1].Index, got[1].TokenStart, got[1].TokenCount)
	}

	source, err := saved.StateBlockSourceFrom(4, 2)
	if err != nil {
		t.Fatalf("StateBlockSourceFrom: %v", err)
	}
	if source.BlockCount != len(got) {
		t.Fatalf("source block count = %d, want %d", source.BlockCount, len(got))
	}
	for i := 0; i < source.BlockCount; i++ {
		block, err := source.Load(i)
		if err != nil {
			t.Fatalf("source.Load(%d): %v", i, err)
		}
		if block.Index != got[i].Index || block.TokenStart != got[i].TokenStart || block.TokenCount != got[i].TokenCount {
			t.Fatalf("source block %d = index %d start %d count %d, want index %d start %d count %d", i, block.Index, block.TokenStart, block.TokenCount, got[i].Index, got[i].TokenStart, got[i].TokenCount)
		}
	}
}

func TestSessionStateBlockSourceTrustPrefixBlocks(t *testing.T) {
	source := SessionStateBlockSource{Position: 7, BlockCount: 2}
	if err := source.TrustPrefixBlocks(2, 2); err != nil {
		t.Fatalf("TrustPrefixBlocks: %v", err)
	}
	if got := source.trustedPrefixTokens(); got != 4 {
		t.Fatalf("trusted prefix = %d, want 4", got)
	}
	if source.firstBlockIndex != 2 || source.totalBlockCount != 4 {
		t.Fatalf("block grid = first %d total %d, want 2/4", source.firstBlockIndex, source.totalBlockCount)
	}
	if err := source.TrustPrefixBlocks(2, 0); err != nil {
		t.Fatalf("TrustPrefixBlocks reset: %v", err)
	}
	if got := source.trustedPrefixTokens(); got != 0 {
		t.Fatalf("trusted prefix after reset = %d, want 0", got)
	}
	if err := source.TrustPrefixBlocks(0, 1); err == nil {
		t.Fatal("TrustPrefixBlocks zero block size error = nil")
	}
	if err := source.TrustPrefixBlocks(4, 3); err == nil {
		t.Fatal("TrustPrefixBlocks oversized prefix error = nil")
	}
}

func TestSessionStateBlockSourceTrustPrefixTokens(t *testing.T) {
	source := SessionStateBlockSource{Position: 7, BlockCount: 2}
	if err := source.TrustPrefixTokens(3, 2); err != nil {
		t.Fatalf("TrustPrefixTokens: %v", err)
	}
	if got := source.trustedPrefixTokens(); got != 3 {
		t.Fatalf("trusted token prefix = %d, want 3", got)
	}
	if source.firstBlockIndex != 2 || source.totalBlockCount != 4 {
		t.Fatalf("block grid = first %d total %d, want 2/4", source.firstBlockIndex, source.totalBlockCount)
	}
	if err := source.TrustPrefixTokens(0, 0); err != nil {
		t.Fatalf("TrustPrefixTokens reset: %v", err)
	}
	if got := source.trustedPrefixTokens(); got != 0 {
		t.Fatalf("trusted token prefix after reset = %d, want 0", got)
	}
	if err := source.TrustPrefixTokens(-1, 1); err == nil {
		t.Fatal("TrustPrefixTokens negative prefix error = nil")
	}
	if err := source.TrustPrefixTokens(8, 1); err == nil {
		t.Fatal("TrustPrefixTokens oversized prefix error = nil")
	}
	if err := source.TrustPrefixTokens(3, 0); err == nil {
		t.Fatal("TrustPrefixTokens positive prefix with zero first block error = nil")
	}
}

func TestSessionStateBlockSourceCarriesSlidingCacheMetadata(t *testing.T) {
	requireNativeRuntime(t)
	g, arch, maxLen := sessionStateFixture(t)
	arch.SlidingWindow = 4
	arch.Layer[0].Attention = model.SlidingAttention
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	if err := sess.PrefillTokens([]int32{1, 2, 3}); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	source, err := sess.StateBlockSource(2)
	if err != nil {
		t.Fatalf("StateBlockSource: %v", err)
	}
	block, err := source.Load(0)
	if err != nil {
		t.Fatalf("Load(0): %v", err)
	}
	if len(block.Layers) == 0 {
		t.Fatal("state block has no layers")
	}
	layer := block.Layers[0]
	if layer.Layer != 0 {
		t.Fatalf("first layer = %d, want sliding layer 0", layer.Layer)
	}
	if layer.CacheMode != "fixed" || layer.MaxSize != arch.SlidingWindow {
		t.Fatalf("sliding layer cache metadata = %q/%d, want fixed/%d", layer.CacheMode, layer.MaxSize, arch.SlidingWindow)
	}
}

func TestSessionStateBlockSourceSplitsSlidingWindowBoundary(t *testing.T) {
	requireNativeRuntime(t)
	g, arch, maxLen := sessionStateFixture(t)
	arch.SlidingWindow = 4
	arch.Layer[0].Attention = model.SlidingAttention
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	if err := sess.PrefillTokens([]int32{1, 2, 3, 4, 5, 6, 7}); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	source, err := sess.StateBlockSource(2)
	if err != nil {
		t.Fatalf("StateBlockSource: %v", err)
	}
	want := []struct {
		index int
		start int
		count int
	}{
		{0, 0, 2},
		{1, 2, 1},
		{2, 3, 1},
		{3, 4, 2},
		{4, 6, 1},
	}
	if source.BlockCount != len(want) {
		t.Fatalf("block count = %d, want %d", source.BlockCount, len(want))
	}
	for i, w := range want {
		block, err := source.Load(i)
		if err != nil {
			t.Fatalf("Load(%d): %v", i, err)
		}
		if block.Index != w.index || block.TokenStart != w.start || block.TokenCount != w.count {
			t.Fatalf("block %d = index %d start %d count %d, want index %d start %d count %d", i, block.Index, block.TokenStart, block.TokenCount, w.index, w.start, w.count)
		}
	}
}

func TestSessionStateFillBlockMapsSlidingRingRows(t *testing.T) {
	keyRows := []byte{
		4, 0,
		5, 0,
		2, 0,
		3, 0,
	}
	valueRows := []byte{
		14, 0,
		15, 0,
		12, 0,
		13, 0,
	}
	layers := make([]SessionStateLayerBlock, 1)
	block, err := fillStateBlock(2, 2, 3, 6, []sessionStateLayerView{{
		layer:      0,
		cacheIndex: 0,
		cacheMode:  nativeStateCacheModeFixed,
		maxSize:    4,
		cacheRows:  4,
		kvHeads:    1,
		headDim:    1,
		rowBytes:   2,
		keyBytes:   keyRows,
		valueBytes: valueRows,
	}}, layers)
	if err != nil {
		t.Fatalf("fillStateBlock sliding ring: %v", err)
	}
	if block.TokenStart != 4 || block.TokenCount != 2 {
		t.Fatalf("block range = %d/%d, want 4/2", block.TokenStart, block.TokenCount)
	}
	if !bytes.Equal(block.Layers[0].KeyBytes, []byte{4, 0, 5, 0}) {
		t.Fatalf("sliding key rows = %v, want logical rows 4,5", block.Layers[0].KeyBytes)
	}
	if !bytes.Equal(block.Layers[0].ValueBytes, []byte{14, 0, 15, 0}) {
		t.Fatalf("sliding value rows = %v, want logical rows 4,5", block.Layers[0].ValueBytes)
	}
	if unsafe.Pointer(&block.Layers[0].KeyBytes[0]) != unsafe.Pointer(&keyRows[0]) {
		t.Fatal("contiguous sliding key rows were copied instead of returned as a resident view")
	}
	if unsafe.Pointer(&block.Layers[0].ValueBytes[0]) != unsafe.Pointer(&valueRows[0]) {
		t.Fatal("contiguous sliding value rows were copied instead of returned as a resident view")
	}
}

func TestSessionStateBlockBoundariesSplitSlidingRingWrap(t *testing.T) {
	sess := &ArchSession{}
	got := append([]int(nil), sess.stateBlockBoundaries(3, 10, []sessionStateLayerView{{
		maxSize:   4,
		cacheRows: 4,
	}})...)
	want := []int{0, 3, 6, 8, 9, 10}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("sliding boundaries = %v, want %v", got, want)
	}
}

func TestSessionStateFillBlockOmitsExpiredSlidingRows(t *testing.T) {
	layers := make([]SessionStateLayerBlock, 1)
	block, err := fillStateBlock(0, 2, 3, 6, []sessionStateLayerView{{
		layer:      0,
		cacheIndex: 0,
		cacheMode:  nativeStateCacheModeFixed,
		maxSize:    4,
		cacheRows:  4,
		kvHeads:    1,
		headDim:    1,
		rowBytes:   2,
		keyBytes:   []byte{4, 0, 5, 0, 2, 0, 3, 0},
		valueBytes: []byte{14, 0, 15, 0, 12, 0, 13, 0},
	}}, layers)
	if err != nil {
		t.Fatalf("fillStateBlock expired sliding rows: %v", err)
	}
	if block.TokenStart != 0 || block.TokenCount != 2 {
		t.Fatalf("block range = %d/%d, want 0/2", block.TokenStart, block.TokenCount)
	}
	if len(block.Layers) != 1 || block.Layers[0].Layer != 0 {
		t.Fatalf("block layers = %+v, want metadata-only sliding layer", block.Layers)
	}
	if len(block.Layers[0].KeyBytes) != 0 || len(block.Layers[0].ValueBytes) != 0 {
		t.Fatalf("expired sliding rows carried KV bytes key=%v value=%v, want metadata-only", block.Layers[0].KeyBytes, block.Layers[0].ValueBytes)
	}
	if block.Layers[0].CacheMode != nativeStateCacheModeFixed || block.Layers[0].MaxSize != 4 {
		t.Fatalf("expired sliding metadata = %q/%d, want fixed/4", block.Layers[0].CacheMode, block.Layers[0].MaxSize)
	}
}

func TestSessionStateRestoreBlockMapsSlidingRingRows(t *testing.T) {
	keyRows := make([]byte, 8)
	valueRows := make([]byte, 8)
	err := restoreStateBlock(2, 4, 6, 1, []sessionStateLayerView{{
		layer:      0,
		cacheIndex: 0,
		cacheMode:  nativeStateCacheModeFixed,
		maxSize:    4,
		cacheRows:  4,
		kvHeads:    1,
		headDim:    1,
		rowBytes:   2,
		keyBytes:   keyRows,
		valueBytes: valueRows,
	}}, SessionStateBlock{
		Index:      2,
		TokenStart: 4,
		TokenCount: 2,
		Layers: []SessionStateLayerBlock{{
			Layer:      0,
			CacheIndex: 0,
			CacheMode:  nativeStateCacheModeFixed,
			MaxSize:    4,
			KVHeads:    1,
			HeadDim:    1,
			RowBytes:   2,
			KeyBytes:   []byte{4, 0, 5, 0},
			ValueBytes: []byte{14, 0, 15, 0},
		}},
	})
	if err != nil {
		t.Fatalf("restoreStateBlock sliding ring: %v", err)
	}
	if !bytes.Equal(keyRows[:4], []byte{4, 0, 5, 0}) {
		t.Fatalf("restored key ring prefix = %v, want logical rows 4,5 in slots 0,1", keyRows)
	}
	if !bytes.Equal(valueRows[:4], []byte{14, 0, 15, 0}) {
		t.Fatalf("restored value ring prefix = %v, want logical rows 4,5 in slots 0,1", valueRows)
	}
}

func TestSessionStateRestoreBlockSkipsExpiredSlidingRows(t *testing.T) {
	keyRows := []byte{4, 0, 5, 0, 2, 0, 3, 0}
	valueRows := []byte{14, 0, 15, 0, 12, 0, 13, 0}
	origKey := append([]byte(nil), keyRows...)
	origValue := append([]byte(nil), valueRows...)
	err := restoreStateBlock(0, 0, 6, 1, []sessionStateLayerView{{
		layer:      0,
		cacheIndex: 0,
		cacheMode:  nativeStateCacheModeFixed,
		maxSize:    4,
		cacheRows:  4,
		kvHeads:    1,
		headDim:    1,
		rowBytes:   2,
		keyBytes:   keyRows,
		valueBytes: valueRows,
	}}, SessionStateBlock{
		Index:      0,
		TokenStart: 0,
		TokenCount: 2,
		Layers: []SessionStateLayerBlock{{
			Layer:      0,
			CacheIndex: 0,
			CacheMode:  nativeStateCacheModeFixed,
			MaxSize:    4,
			KVHeads:    1,
			HeadDim:    1,
			RowBytes:   2,
		}},
	})
	if err != nil {
		t.Fatalf("restoreStateBlock expired sliding rows: %v", err)
	}
	if !bytes.Equal(keyRows, origKey) || !bytes.Equal(valueRows, origValue) {
		t.Fatalf("expired sliding restore mutated rows key=%v value=%v", keyRows, valueRows)
	}
}

func TestSessionStateRestoreBlockRejectsFixedMaxSizeMismatch(t *testing.T) {
	err := restoreStateBlock(0, 0, 2, 1, []sessionStateLayerView{{
		layer:      0,
		cacheIndex: 0,
		cacheMode:  nativeStateCacheModeFixed,
		maxSize:    4,
		cacheRows:  4,
		kvHeads:    1,
		headDim:    1,
		rowBytes:   2,
		keyBytes:   make([]byte, 8),
		valueBytes: make([]byte, 8),
	}}, SessionStateBlock{
		Index:      0,
		TokenStart: 0,
		TokenCount: 2,
		Layers: []SessionStateLayerBlock{{
			Layer:      0,
			CacheIndex: 0,
			CacheMode:  nativeStateCacheModeFixed,
			MaxSize:    6,
			KVHeads:    1,
			HeadDim:    1,
			RowBytes:   2,
			KeyBytes:   []byte{1, 0, 2, 0},
			ValueBytes: []byte{11, 0, 12, 0},
		}},
	})
	if err == nil {
		t.Fatal("restoreStateBlock fixed max-size mismatch error = nil")
	}
}

func TestSessionStateRestoreBlockRejectsCacheModeMismatch(t *testing.T) {
	err := restoreStateBlock(0, 0, 2, 1, []sessionStateLayerView{{
		layer:      0,
		cacheIndex: 0,
		cacheMode:  nativeStateCacheModeFixed,
		maxSize:    4,
		cacheRows:  4,
		kvHeads:    1,
		headDim:    1,
		rowBytes:   2,
		keyBytes:   make([]byte, 8),
		valueBytes: make([]byte, 8),
	}}, SessionStateBlock{
		Index:      0,
		TokenStart: 0,
		TokenCount: 2,
		Layers: []SessionStateLayerBlock{{
			Layer:      0,
			CacheIndex: 0,
			CacheMode:  "paged",
			MaxSize:    4,
			KVHeads:    1,
			HeadDim:    1,
			RowBytes:   2,
			KeyBytes:   []byte{1, 0, 2, 0},
			ValueBytes: []byte{11, 0, 12, 0},
		}},
	})
	if err == nil {
		t.Fatal("restoreStateBlock cache-mode mismatch error = nil")
	}
}

func TestSessionStateRestoreBlocksGraftsTrustedPrefix(t *testing.T) {
	requireNativeRuntime(t)
	prefix := []int32{1, 2, 3, 4}
	suffix := []int32{5, 6, 7}
	prompt := append(append([]int32(nil), prefix...), suffix...)

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens full prompt: %v", err)
	}
	source, err := saved.StateBlockSourceFrom(len(prefix), 2)
	if err != nil {
		t.Fatalf("StateBlockSourceFrom: %v", err)
	}

	empty := newSessionStateFixture(t)
	if err := empty.RestoreStateBlocks(source); err == nil {
		t.Fatal("RestoreStateBlocks skipped-prefix into empty session error = nil")
	}

	restored := newSessionStateFixture(t)
	if err := restored.PrefillTokens(prefix); err != nil {
		t.Fatalf("PrefillTokens prefix: %v", err)
	}
	if err := restored.RestoreStateBlocks(source); err != nil {
		t.Fatalf("RestoreStateBlocks skipped-prefix: %v", err)
	}
	if restored.Pos() != len(prompt) {
		t.Fatalf("restored pos = %d, want %d", restored.Pos(), len(prompt))
	}
	if !idsEqual(restored.cachedIDs, prompt) {
		t.Fatalf("restored cached ids = %v, want %v", restored.cachedIDs, prompt)
	}
	got, err := restored.GenerateFromCache(3, -1)
	if err != nil {
		t.Fatalf("GenerateFromCache after skipped-prefix block restore: %v", err)
	}
	cold := newSessionStateFixture(t)
	want, err := cold.Generate(prompt, 3, -1)
	if err != nil {
		t.Fatalf("cold Generate: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("skipped-prefix block-restored GenerateFromCache = %v, want cold prompt continuation %v", got, want)
	}
}

func TestSessionStateRestoreBlocksGraftsExactTrustedPrefix(t *testing.T) {
	requireNativeRuntime(t)
	prefix := []int32{1, 2, 3}
	suffix := []int32{4}
	prompt := append(append([]int32(nil), prefix...), suffix...)

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens full prompt: %v", err)
	}
	sourceAll, err := saved.StateBlockSource(2)
	if err != nil {
		t.Fatalf("StateBlockSource: %v", err)
	}
	parentBlock, err := sourceAll.Load(1)
	if err != nil {
		t.Fatalf("Load parent suffix block: %v", err)
	}
	if parentBlock.TokenStart != 2 || parentBlock.TokenCount != 2 {
		t.Fatalf("parent block = start %d count %d, want 2/2", parentBlock.TokenStart, parentBlock.TokenCount)
	}
	suffixLayers := make([]SessionStateLayerBlock, len(parentBlock.Layers))
	for i, layer := range parentBlock.Layers {
		rowOff := (len(prefix) - parentBlock.TokenStart) * layer.RowBytes
		rowEnd := rowOff + layer.RowBytes
		if rowOff < 0 || rowEnd > len(layer.KeyBytes) || rowEnd > len(layer.ValueBytes) {
			t.Fatalf("suffix layer %d row slice [%d:%d] outside key/value payloads", i, rowOff, rowEnd)
		}
		suffixLayers[i] = layer
		suffixLayers[i].KeyBytes = append([]byte(nil), layer.KeyBytes[rowOff:rowEnd]...)
		suffixLayers[i].ValueBytes = append([]byte(nil), layer.ValueBytes[rowOff:rowEnd]...)
	}
	source := SessionStateBlockSource{
		Position:           len(prompt),
		CachedIDs:          append([]int32(nil), prompt...),
		CachedPromptIDs:    append([]int32(nil), prompt...),
		CachedPromptHidden: append([]byte(nil), sourceAll.CachedPromptHidden...),
		CachedPromptLogits: append([]byte(nil), sourceAll.CachedPromptLogits...),
		RetainedHidden:     append([]byte(nil), sourceAll.RetainedHidden...),
		RetainedLogits:     append([]byte(nil), sourceAll.RetainedLogits...),
		BlockCount:         1,
		Load: func(index int) (SessionStateBlock, error) {
			if index != 0 {
				return SessionStateBlock{}, core.NewError("test: block index out of range")
			}
			return SessionStateBlock{
				Index:      2,
				TokenStart: len(prefix),
				TokenCount: len(suffix),
				Layers:     suffixLayers,
			}, nil
		},
	}
	if err := source.TrustPrefixTokens(len(prefix), 2); err != nil {
		t.Fatalf("TrustPrefixTokens exact prefix: %v", err)
	}

	empty := newSessionStateFixture(t)
	if err := empty.RestoreStateBlocks(source); err == nil {
		t.Fatal("RestoreStateBlocks exact-prefix into empty session error = nil")
	}

	restored := newSessionStateFixture(t)
	if err := restored.PrefillTokens(prefix); err != nil {
		t.Fatalf("PrefillTokens prefix: %v", err)
	}
	if err := restored.RestoreStateBlocks(source); err != nil {
		t.Fatalf("RestoreStateBlocks exact-prefix: %v", err)
	}
	if restored.Pos() != len(prompt) {
		t.Fatalf("restored pos = %d, want %d", restored.Pos(), len(prompt))
	}
	if !idsEqual(restored.cachedIDs, prompt) {
		t.Fatalf("restored cached ids = %v, want %v", restored.cachedIDs, prompt)
	}
}

func TestSessionStateBlocksRestorePromptCacheEntry(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}

	saved := newSessionStateFixture(t)
	if err := saved.WarmPromptCache(prompt); err != nil {
		t.Fatalf("WarmPromptCache: %v", err)
	}
	source, err := saved.StateBlockSource(2)
	if err != nil {
		t.Fatalf("StateBlockSource: %v", err)
	}

	restored := newSessionStateFixture(t)
	if err := restored.RestoreStateBlocks(source); err != nil {
		t.Fatalf("RestoreStateBlocks: %v", err)
	}
	if hit := restored.CachedPrefixLen(prompt); hit != len(prompt) {
		t.Fatalf("restored exact prompt-cache hit = %d, want %d", hit, len(prompt))
	}
	head := restored.head
	headCalls := 0
	restored.greedy = nil
	restored.head = func(hidden []byte, skipSoftcap bool) ([]byte, error) {
		headCalls++
		return head(hidden, skipSoftcap)
	}
	got, err := restored.GenerateCached(prompt, 3, -1)
	if err != nil {
		t.Fatalf("GenerateCached after RestoreStateBlocks: %v", err)
	}
	if headCalls != len(got)-1 {
		t.Fatalf("restored exact prompt-cache head calls = %d, want %d", headCalls, len(got)-1)
	}

	cold := newSessionStateFixture(t)
	want, err := cold.Generate(prompt, 3, -1)
	if err != nil {
		t.Fatalf("cold Generate: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("block-restored prompt-cache generation = %v, want %v", got, want)
	}
}

func TestArchSessionCaptureKVRootSnapshotUsesNativeLayerSlabs(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}

	sess := newSessionStateFixture(t)
	if err := sess.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	snapshot, err := sess.CaptureKVWithOptions(kv.CaptureOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("CaptureKVWithOptions: %v", err)
	}
	if snapshot.Version != kv.SnapshotVersion {
		t.Fatalf("snapshot version = %d, want %d", snapshot.Version, kv.SnapshotVersion)
	}
	if !idsEqual(snapshot.Tokens, prompt) || snapshot.TokenOffset != len(prompt) || snapshot.SeqLen != len(prompt) {
		t.Fatalf("snapshot tokens/offset/seq = %v/%d/%d, want %v/%d/%d", snapshot.Tokens, snapshot.TokenOffset, snapshot.SeqLen, prompt, len(prompt), len(prompt))
	}
	if snapshot.NumLayers != len(sess.state.specs) || len(snapshot.Layers) != len(sess.state.specs) {
		t.Fatalf("snapshot layers = %d/%d, want %d", snapshot.NumLayers, len(snapshot.Layers), len(sess.state.specs))
	}
	layer := snapshot.Layers[0]
	if layer.Layer != 0 || layer.CacheIndex != sess.state.specs[0].CacheIndex {
		t.Fatalf("layer identity = %d/%d, want layer 0 cache %d", layer.Layer, layer.CacheIndex, sess.state.specs[0].CacheIndex)
	}
	wantShape := []int32{1, int32(kvHeadsOf(sess.state.specs[0], sess.arch.KVHeads)), int32(len(prompt)), int32(headDimOf(sess.state.specs[0], sess.arch.HeadDim))}
	if !reflect.DeepEqual(layer.KeyShape, wantShape) || !reflect.DeepEqual(layer.ValueShape, wantShape) {
		t.Fatalf("layer shapes = %v/%v, want %v", layer.KeyShape, layer.ValueShape, wantShape)
	}
	if layer.KeyDType != "bfloat16" || layer.ValueDType != "bfloat16" {
		t.Fatalf("layer dtypes = %q/%q, want bfloat16", layer.KeyDType, layer.ValueDType)
	}
	wantBytes := int(wantShape[1] * wantShape[2] * wantShape[3] * bf16Size)
	if len(layer.KeyBytes) != wantBytes || len(layer.ValueBytes) != wantBytes {
		t.Fatalf("layer byte lengths = %d/%d, want %d", len(layer.KeyBytes), len(layer.ValueBytes), wantBytes)
	}
	if len(layer.Heads) != 0 {
		t.Fatalf("raw-only snapshot carried %d per-head float snapshots, want none", len(layer.Heads))
	}
}

func TestArchSessionRestoreKVRootSnapshotContinuesFromNativeLayerSlabs(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	snapshot, err := saved.CaptureKVWithOptions(kv.CaptureOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("CaptureKVWithOptions: %v", err)
	}

	restored := newSessionStateFixture(t)
	if err := restored.RestoreKV(snapshot); err != nil {
		t.Fatalf("RestoreKV: %v", err)
	}
	if restored.Pos() != saved.Pos() {
		t.Fatalf("restored pos = %d, want %d", restored.Pos(), saved.Pos())
	}
	if !idsEqual(restored.cachedIDs, prompt) {
		t.Fatalf("restored cached ids = %v, want %v", restored.cachedIDs, prompt)
	}
	got, err := restored.GenerateFromCache(3, -1)
	if err != nil {
		t.Fatalf("GenerateFromCache after RestoreKV: %v", err)
	}
	cold := newSessionStateFixture(t)
	want, err := cold.Generate(prompt, 3, -1)
	if err != nil {
		t.Fatalf("cold Generate: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("restored GenerateFromCache = %v, want cold continuation %v", got, want)
	}
}

func TestArchSessionRangeKVBlocksRootSnapshots(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}

	sess := newSessionStateFixture(t)
	if err := sess.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	var blocks []kv.Block
	err := sess.RangeKVBlocks(2, kv.CaptureOptions{RawKVOnly: true}, func(block kv.Block) (bool, error) {
		blocks = append(blocks, block)
		return true, nil
	})
	if err != nil {
		t.Fatalf("RangeKVBlocks: %v", err)
	}
	if len(blocks) != 3 {
		t.Fatalf("RangeKVBlocks yielded %d blocks, want 3", len(blocks))
	}
	for i, block := range blocks {
		if block.Index != i {
			t.Fatalf("block %d index = %d, want %d", i, block.Index, i)
		}
		if block.Snapshot == nil {
			t.Fatalf("block %d snapshot = nil", i)
		}
		if block.Snapshot.TokenOffset != block.TokenStart+block.TokenCount || block.Snapshot.SeqLen != block.TokenCount {
			t.Fatalf("block %d offset/seq = %d/%d, want %d/%d", i, block.Snapshot.TokenOffset, block.Snapshot.SeqLen, block.TokenStart+block.TokenCount, block.TokenCount)
		}
		if !idsEqual(block.Snapshot.Tokens, prompt[block.TokenStart:block.TokenStart+block.TokenCount]) {
			t.Fatalf("block %d tokens = %v, want %v", i, block.Snapshot.Tokens, prompt[block.TokenStart:block.TokenStart+block.TokenCount])
		}
	}
	layer := blocks[0].Snapshot.Layers[0]
	wantShape := []int32{1, int32(kvHeadsOf(sess.state.specs[0], sess.arch.KVHeads)), 2, int32(headDimOf(sess.state.specs[0], sess.arch.HeadDim))}
	if !reflect.DeepEqual(layer.KeyShape, wantShape) || !reflect.DeepEqual(layer.ValueShape, wantShape) {
		t.Fatalf("first block layer shapes = %v/%v, want %v", layer.KeyShape, layer.ValueShape, wantShape)
	}
	if len(blocks[0].Snapshot.Logits) != 0 {
		t.Fatalf("non-final block carried %d logits, want none", len(blocks[0].Snapshot.Logits))
	}
	if len(blocks[len(blocks)-1].Snapshot.Logits) != sess.arch.Vocab {
		t.Fatalf("final block logits = %d, want vocab %d", len(blocks[len(blocks)-1].Snapshot.Logits), sess.arch.Vocab)
	}
}

func TestArchSessionRangeKVBlocksHonoursBlockStartToken(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5, 6}

	sess := newSessionStateFixture(t)
	if err := sess.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	var blocks []kv.Block
	err := sess.RangeKVBlocks(2, kv.CaptureOptions{RawKVOnly: true, BlockStartToken: 4}, func(block kv.Block) (bool, error) {
		blocks = append(blocks, block)
		return true, nil
	})
	if err != nil {
		t.Fatalf("RangeKVBlocks: %v", err)
	}
	if len(blocks) != 1 {
		t.Fatalf("RangeKVBlocks yielded %d blocks, want 1", len(blocks))
	}
	if blocks[0].Index != 2 || blocks[0].TokenStart != 4 || blocks[0].TokenCount != 2 {
		t.Fatalf("block identity = index %d start %d count %d, want 2/4/2", blocks[0].Index, blocks[0].TokenStart, blocks[0].TokenCount)
	}
	if !idsEqual(blocks[0].Snapshot.Tokens, prompt[4:]) {
		t.Fatalf("block tokens = %v, want %v", blocks[0].Snapshot.Tokens, prompt[4:])
	}
}

func TestArchSessionRestoreKVBlocksRootSnapshotsContinues(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	source, err := saved.KVBlockSource(2, kv.CaptureOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("KVBlockSource: %v", err)
	}

	restored := newSessionStateFixture(t)
	if err := restored.RestoreKVBlocks(source); err != nil {
		t.Fatalf("RestoreKVBlocks: %v", err)
	}
	if restored.Pos() != saved.Pos() {
		t.Fatalf("restored pos = %d, want %d", restored.Pos(), saved.Pos())
	}
	if !idsEqual(restored.cachedIDs, prompt) {
		t.Fatalf("restored cached ids = %v, want %v", restored.cachedIDs, prompt)
	}
	got, err := restored.GenerateFromCache(3, -1)
	if err != nil {
		t.Fatalf("GenerateFromCache after RestoreKVBlocks: %v", err)
	}
	cold := newSessionStateFixture(t)
	want, err := cold.Generate(prompt, 3, -1)
	if err != nil {
		t.Fatalf("cold Generate: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("restored GenerateFromCache = %v, want cold continuation %v", got, want)
	}
}

func TestArchSessionRestoreKVBlocksHonoursPrefixTokens(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5, 6}
	prefix := prompt[:4]

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens(saved): %v", err)
	}
	source, err := saved.KVBlockSource(2, kv.CaptureOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("KVBlockSource: %v", err)
	}
	source.PrefixTokens = len(prefix)

	restored := newSessionStateFixture(t)
	if err := restored.RestoreKVBlocks(source); err != nil {
		t.Fatalf("RestoreKVBlocks: %v", err)
	}
	if restored.Pos() != len(prefix) {
		t.Fatalf("restored pos = %d, want %d", restored.Pos(), len(prefix))
	}
	if !idsEqual(restored.cachedIDs, prefix) {
		t.Fatalf("restored cached ids = %v, want %v", restored.cachedIDs, prefix)
	}
	if _, err := restored.GenerateFromCache(1, -1); err == nil {
		t.Fatal("GenerateFromCache after prefix-only RestoreKVBlocks error = nil")
	}
}

func TestArchSessionRestoreKVBlocksRootSnapshotsTrustedPrefix(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5, 6}
	prefix := prompt[:4]

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens(saved): %v", err)
	}
	source, err := saved.KVBlockSource(2, kv.CaptureOptions{RawKVOnly: true, BlockStartToken: len(prefix)})
	if err != nil {
		t.Fatalf("KVBlockSource: %v", err)
	}
	if source.TrustedPrefixTokens != len(prefix) || source.FirstBlockIndex != 2 {
		t.Fatalf("trusted prefix/index = %d/%d, want %d/2", source.TrustedPrefixTokens, source.FirstBlockIndex, len(prefix))
	}

	restored := newSessionStateFixture(t)
	if err := restored.PrefillTokens(prefix); err != nil {
		t.Fatalf("PrefillTokens(restored prefix): %v", err)
	}
	if err := restored.RestoreKVBlocks(source); err != nil {
		t.Fatalf("RestoreKVBlocks: %v", err)
	}
	if restored.Pos() != saved.Pos() {
		t.Fatalf("restored pos = %d, want %d", restored.Pos(), saved.Pos())
	}
	if !idsEqual(restored.cachedIDs, prompt) {
		t.Fatalf("restored cached ids = %v, want %v", restored.cachedIDs, prompt)
	}
	got, err := restored.GenerateFromCache(3, -1)
	if err != nil {
		t.Fatalf("GenerateFromCache after RestoreKVBlocks trusted prefix: %v", err)
	}
	cold := newSessionStateFixture(t)
	want, err := cold.Generate(prompt, 3, -1)
	if err != nil {
		t.Fatalf("cold Generate: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("trusted-prefix GenerateFromCache = %v, want cold continuation %v", got, want)
	}
}

func TestArchSessionRestoreKVBlocksRejectsTrustedPrefixMismatch(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5, 6}
	prefix := prompt[:4]

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens(saved): %v", err)
	}
	source, err := saved.KVBlockSource(2, kv.CaptureOptions{RawKVOnly: true, BlockStartToken: len(prefix)})
	if err != nil {
		t.Fatalf("KVBlockSource: %v", err)
	}

	restored := newSessionStateFixture(t)
	if err := restored.PrefillTokens([]int32{1, 2, 3, 7}); err != nil {
		t.Fatalf("PrefillTokens(restored prefix): %v", err)
	}
	if err := restored.RestoreKVBlocks(source); err == nil {
		t.Fatal("RestoreKVBlocks mismatch error = nil")
	}
}

func TestArchSessionRestoreKVBlocksAllTrustedPrefixContinues(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5, 6}

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens(saved): %v", err)
	}
	source, err := saved.KVBlockSource(2, kv.CaptureOptions{RawKVOnly: true, BlockStartToken: len(prompt)})
	if err != nil {
		t.Fatalf("KVBlockSource: %v", err)
	}
	if source.BlockCount != 0 || source.TrustedPrefixTokens != len(prompt) {
		t.Fatalf("source blocks/trusted prefix = %d/%d, want 0/%d", source.BlockCount, source.TrustedPrefixTokens, len(prompt))
	}

	restored := newSessionStateFixture(t)
	if err := restored.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens(restored): %v", err)
	}
	if err := restored.RestoreKVBlocks(source); err != nil {
		t.Fatalf("RestoreKVBlocks: %v", err)
	}
	got, err := restored.GenerateFromCache(3, -1)
	if err != nil {
		t.Fatalf("GenerateFromCache after all-trusted RestoreKVBlocks: %v", err)
	}
	cold := newSessionStateFixture(t)
	want, err := cold.Generate(prompt, 3, -1)
	if err != nil {
		t.Fatalf("cold Generate: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("all-trusted GenerateFromCache = %v, want cold continuation %v", got, want)
	}
}

func TestSessionStateNoRuntimeValidation(t *testing.T) {
	icbSession := &ArchSession{state: archDecodeState{icb: &archICBReplay{}}}
	if _, err := icbSession.SerializeState(); err != nil {
		t.Fatalf("SerializeState(empty ICB) error = %v", err)
	}
	if err := icbSession.RestoreState(emptySessionStateBlob(0, 0, 0)); err != nil {
		t.Fatalf("RestoreState(empty ICB) error = %v", err)
	}

	if err := (&ArchSession{}).RestoreState(nil); err == nil {
		t.Fatal("RestoreState(nil) error = nil")
	}
	if err := (&ArchSession{}).RestoreState(emptySessionStateBlob(0, 1, 0)); err == nil {
		t.Fatal("RestoreState(layer mismatch) error = nil")
	}

	legacy := make([]byte, 12)
	binary.LittleEndian.PutUint32(legacy[0:], sessionStateMagic)
	binary.LittleEndian.PutUint32(legacy[4:], 7)
	if err := (&ArchSession{}).RestoreState(legacy); err != nil {
		t.Fatalf("RestoreState(legacy snapshot) error = %v", err)
	}

	if err := (&ArchSession{}).RestoreState(append(legacy, 0)); err == nil {
		t.Fatal("RestoreState(truncated metadata length) error = nil")
	}
	truncatedIDs := emptySessionStateBlob(0, 0, 1)[:16]
	if err := (&ArchSession{}).RestoreState(truncatedIDs); err == nil {
		t.Fatal("RestoreState(truncated metadata ids) error = nil")
	}
	trailing := append(emptySessionStateBlob(0, 0, 1), 0)
	if err := (&ArchSession{}).RestoreState(trailing); err == nil {
		t.Fatal("RestoreState(trailing metadata) error = nil")
	}
}

func TestSessionStateSerializeZeroLayerCachedIDs(t *testing.T) {
	saved := &ArchSession{pos: 3, cachedIDs: []int32{7, 8, 9}}
	blob, err := saved.SerializeState()
	if err != nil {
		t.Fatalf("SerializeState: %v", err)
	}

	restored := &ArchSession{}
	if err := restored.RestoreState(blob); err != nil {
		t.Fatalf("RestoreState: %v", err)
	}
	if restored.Pos() != saved.Pos() {
		t.Fatalf("restored pos = %d, want %d", restored.Pos(), saved.Pos())
	}
	next := []int32{7, 8, 9, 10}
	if got := restored.CachedPrefixLen(next); got != len(saved.cachedIDs) {
		t.Fatalf("restored cached prefix = %d, want %d", got, len(saved.cachedIDs))
	}
}

func TestSessionStateRestoresPromptCacheEntry(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}

	a := newSessionStateFixture(t)
	if err := a.WarmPromptCache(prompt); err != nil {
		t.Fatalf("WarmPromptCache: %v", err)
	}
	blob, err := a.SerializeState()
	if err != nil {
		t.Fatalf("SerializeState: %v", err)
	}

	b := newSessionStateFixture(t)
	if err := b.RestoreState(blob); err != nil {
		t.Fatalf("RestoreState: %v", err)
	}
	if hit := b.CachedPrefixLen(prompt); hit != len(prompt) {
		t.Fatalf("restored exact prompt-cache hit = %d, want %d", hit, len(prompt))
	}
	head := b.head
	headCalls := 0
	b.greedy = nil
	b.head = func(hidden []byte, skipSoftcap bool) ([]byte, error) {
		headCalls++
		return head(hidden, skipSoftcap)
	}
	got, err := b.GenerateCached(prompt, 3, -1)
	if err != nil {
		t.Fatalf("GenerateCached after RestoreState: %v", err)
	}
	if headCalls != len(got)-1 {
		t.Fatalf("restored exact prompt-cache head calls = %d, want %d", headCalls, len(got)-1)
	}

	cold := newSessionStateFixture(t)
	want, err := cold.Generate(prompt, 3, -1)
	if err != nil {
		t.Fatalf("cold Generate: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("generated length = %d, want %d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("token %d after restored prompt-cache entry = %d, want %d", i, got[i], want[i])
		}
	}
}

// TestSessionStateRoundTrip proves native conversation continuity: a session is decoded, snapshotted
// with SerializeState, and the snapshot is RestoreState'd into a FRESH session — which then continues
// the conversation TOKEN-IDENTICALLY to the original. This is save/resume across a process restart with
// no cgo, the no-cgo equivalent of metal's EnableConversationContinuity.
func TestSessionStateRoundTrip(t *testing.T) {
	requireNativeRuntime(t)

	// session A: decode a first turn, then snapshot.
	a := newSessionStateFixture(t)
	if _, err := a.Generate([]int32{1, 2, 3, 4, 5}, 6, -1); err != nil {
		t.Fatalf("A turn 1: %v", err)
	}
	blob, err := a.SerializeState()
	if err != nil {
		t.Fatalf("SerializeState: %v", err)
	}

	// session B: fresh, restore A's snapshot.
	b := newSessionStateFixture(t)
	if err := b.RestoreState(blob); err != nil {
		t.Fatalf("RestoreState: %v", err)
	}
	if b.Pos() != a.Pos() {
		t.Fatalf("restored pos %d != saved pos %d", b.Pos(), a.Pos())
	}

	// both continue the conversation with the same next turn — must produce identical tokens.
	cont := []int32{20, 21, 22}
	genA, err := a.Generate(cont, 8, -1)
	if err != nil {
		t.Fatalf("A turn 2: %v", err)
	}
	genB, err := b.Generate(cont, 8, -1)
	if err != nil {
		t.Fatalf("B turn 2: %v", err)
	}
	if len(genA) != len(genB) {
		t.Fatalf("continuation length mismatch: A=%d B=%d", len(genA), len(genB))
	}
	for i := range genA {
		if genA[i] != genB[i] {
			t.Fatalf("token %d diverged after restore: A=%d B=%d", i, genA[i], genB[i])
		}
	}
	t.Logf("native continuity: serialize→restore→continue is token-identical over %d continuation tokens (snapshot %d bytes)", len(genA), len(blob))
}

func TestSessionStateRoundTripICBReplay(t *testing.T) {
	requireNativeRuntime(t)
	g, arch, maxLen := icbSessionStateFixture(t)

	a := newICBSessionStateFixture(t, g, arch, maxLen)
	if _, err := a.Generate([]int32{1, 5, 3, 2}, 4, -1); err != nil {
		t.Fatalf("A turn 1: %v", err)
	}
	blob, err := a.SerializeState()
	if err != nil {
		t.Fatalf("SerializeState ICB: %v", err)
	}

	b := newICBSessionStateFixture(t, g, arch, maxLen)
	if err := b.RestoreState(blob); err != nil {
		t.Fatalf("RestoreState ICB: %v", err)
	}
	if b.Pos() != a.Pos() {
		t.Fatalf("restored ICB pos %d != saved pos %d", b.Pos(), a.Pos())
	}

	cont := []int32{7, 8}
	genA, err := a.Generate(cont, 5, -1)
	if err != nil {
		t.Fatalf("A turn 2: %v", err)
	}
	genB, err := b.Generate(cont, 5, -1)
	if err != nil {
		t.Fatalf("B turn 2: %v", err)
	}
	if len(genA) != len(genB) {
		t.Fatalf("ICB continuation length mismatch: A=%d B=%d", len(genA), len(genB))
	}
	for i := range genA {
		if genA[i] != genB[i] {
			t.Fatalf("ICB token %d diverged after restore: A=%d B=%d", i, genA[i], genB[i])
		}
	}
}

// TestSessionStateRoundTripRestoresCachedPrefixMetadata proves state restore
// preserves the prompt-cache metadata that lets GenerateCached reuse resident KV
// rows. Token parity alone is insufficient here: a restored session can produce
// the same tokens by cold re-prefilling, but then the native engine has lost the
// resource-saving prefix hit that metal's prompt-cache restore path preserves.
func TestSessionStateRoundTripRestoresCachedPrefixMetadata(t *testing.T) {
	requireNativeRuntime(t)
	a := newSessionStateFixture(t)
	prompt := []int32{1, 2, 3, 4, 5}
	if _, err := a.GenerateCached(prompt, 6, -1); err != nil {
		t.Fatalf("GenerateCached warmup: %v", err)
	}
	nextPrompt := []int32{1, 2, 3, 4, 5, 6}
	wantHit := a.CachedPrefixLen(nextPrompt)
	if wantHit != len(prompt) {
		t.Fatalf("warm CachedPrefixLen = %d, want %d", wantHit, len(prompt))
	}
	blob, err := a.SerializeState()
	if err != nil {
		t.Fatalf("SerializeState: %v", err)
	}

	b := newSessionStateFixture(t)
	if err := b.RestoreState(blob); err != nil {
		t.Fatalf("RestoreState: %v", err)
	}
	if got := b.CachedPrefixLen(nextPrompt); got != wantHit {
		t.Fatalf("restored CachedPrefixLen = %d, want %d", got, wantHit)
	}
}

func TestSessionStateSerializeCachedPrefixAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	s := newSessionStateFixture(t)
	if _, err := s.GenerateCached([]int32{1, 2, 3, 4, 5}, 6, -1); err != nil {
		t.Fatalf("GenerateCached warmup: %v", err)
	}
	allocs := testing.AllocsPerRun(20, func() {
		if _, err := s.SerializeState(); err != nil {
			t.Fatalf("SerializeState: %v", err)
		}
	})
	if allocs > 82 {
		t.Fatalf("SerializeState allocations = %.0f, want <= 82", allocs)
	}
}

func TestSessionStateRangeBlocksAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	s := newSessionStateFixture(t)
	if _, err := s.GenerateCached([]int32{1, 2, 3, 4, 5}, 6, -1); err != nil {
		t.Fatalf("GenerateCached warmup: %v", err)
	}
	if err := s.RangeStateBlocks(2, func(block SessionStateBlock) (bool, error) {
		return true, nil
	}); err != nil {
		t.Fatalf("RangeStateBlocks warmup: %v", err)
	}
	allocs := testing.AllocsPerRun(20, func() {
		if err := s.RangeStateBlocks(2, func(block SessionStateBlock) (bool, error) {
			if block.TokenCount == 0 {
				t.Fatal("empty block")
			}
			return true, nil
		}); err != nil {
			t.Fatalf("RangeStateBlocks: %v", err)
		}
	})
	if allocs > 0 {
		t.Fatalf("RangeStateBlocks allocations = %.0f, want 0", allocs)
	}
}

func TestSessionStateRestoreBlocksAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	saved := newSessionStateFixture(t)
	prompt := []int32{1, 2, 3, 4, 5}
	if err := saved.WarmPromptCache(prompt); err != nil {
		t.Fatalf("WarmPromptCache: %v", err)
	}
	source, err := saved.StateBlockSource(2)
	if err != nil {
		t.Fatalf("StateBlockSource: %v", err)
	}
	restored := newSessionStateFixture(t)
	if err := restored.RestoreStateBlocks(source); err != nil {
		t.Fatalf("RestoreStateBlocks warmup: %v", err)
	}
	allocs := testing.AllocsPerRun(20, func() {
		if err := restored.RestoreStateBlocks(source); err != nil {
			t.Fatalf("RestoreStateBlocks: %v", err)
		}
	})
	if allocs > 0 {
		t.Fatalf("RestoreStateBlocks allocations = %.0f, want 0", allocs)
	}
}
