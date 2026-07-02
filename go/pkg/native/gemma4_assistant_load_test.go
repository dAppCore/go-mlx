// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"encoding/binary"
	"sort"
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/gguf"
	"dappco.re/go/mlx/pkg/model"
	"dappco.re/go/mlx/pkg/safetensors"
)

func TestLoadGemma4AssistantDirLoadsMetadataAndTensors(t *testing.T) {
	dir := writeNativeAssistantDir(t, nativeAssistantTinyTensors(true))

	assistant, err := LoadGemma4AssistantDir(dir)
	if err != nil {
		t.Fatalf("LoadGemma4AssistantDir: %v", err)
	}
	defer assistant.Close()

	if assistant.ModelType() != "gemma4_assistant" {
		t.Fatalf("ModelType = %q, want gemma4_assistant", assistant.ModelType())
	}
	if assistant.Tokenizer() == nil {
		t.Fatal("Tokenizer = nil, want loaded assistant tokenizer")
	}
	if assistant.NumLayers() != 2 {
		t.Fatalf("NumLayers = %d, want 2", assistant.NumLayers())
	}
	if assistant.BackboneHiddenSize != 8 || assistant.NumCentroids != 2 || !assistant.UseOrderedEmbeddings {
		t.Fatalf("assistant metadata backbone=%d centroids=%d ordered=%v", assistant.BackboneHiddenSize, assistant.NumCentroids, assistant.UseOrderedEmbeddings)
	}
	if assistant.Arch.Hidden != 4 || assistant.Arch.Vocab != 8 || assistant.Arch.FF != 8 {
		t.Fatalf("assistant Arch = %+v, want hidden/vocab/ff 4/8/8", assistant.Arch)
	}
	if tok, ok := assistant.Tensor("masked_embedding.token_ordering"); !ok || tok.Dtype != "I64" || len(tok.Shape) != 1 || tok.Shape[0] != 8 {
		t.Fatalf("token_ordering tensor = %+v, ok=%v; want I64 [8]", tok, ok)
	}
}

func TestLoadGemma4AssistantDirAcceptsFlatTextConfig(t *testing.T) {
	dir := writeNativeAssistantFlatDir(t, nativeAssistantTinyTensors(true), true)

	assistant, err := LoadGemma4AssistantDir(dir)
	if err != nil {
		t.Fatalf("LoadGemma4AssistantDir(flat config): %v", err)
	}
	defer assistant.Close()

	if assistant.Arch.Hidden != 4 || assistant.Arch.Vocab != 8 || assistant.Arch.FF != 8 {
		t.Fatalf("assistant flat Arch = %+v, want hidden/vocab/ff 4/8/8", assistant.Arch)
	}
	if assistant.BackboneHiddenSize != 8 || assistant.NumCentroids != 2 || !assistant.UseOrderedEmbeddings {
		t.Fatalf("assistant flat metadata backbone=%d centroids=%d ordered=%v", assistant.BackboneHiddenSize, assistant.NumCentroids, assistant.UseOrderedEmbeddings)
	}
}

func TestLoadGemma4UnifiedAssistantDirReportsAssistantModelType(t *testing.T) {
	dir := writeNativeAssistantDirWithModelType(t, nativeAssistantTinyTensors(true), true, "gemma4_unified_assistant")

	assistant, err := LoadGemma4AssistantDir(dir)
	if err != nil {
		t.Fatalf("LoadGemma4AssistantDir(unified assistant): %v", err)
	}
	defer assistant.Close()

	if assistant.Config.ModelType != "gemma4_unified_assistant" {
		t.Fatalf("Config.ModelType = %q, want raw unified assistant model type", assistant.Config.ModelType)
	}
	if assistant.ModelType() != "gemma4_assistant" {
		t.Fatalf("ModelType = %q, want public assistant model type", assistant.ModelType())
	}
}

func TestLoadGemma4AssistantDirRejectsMissingRequiredTensor(t *testing.T) {
	tensors := nativeAssistantTinyTensors(false)
	delete(tensors, "post_projection.weight")
	dir := writeNativeAssistantDir(t, tensors)

	assistant, err := LoadGemma4AssistantDir(dir)
	if assistant != nil {
		t.Fatalf("LoadGemma4AssistantDir assistant = %v, want nil on invalid tensor set", assistant)
	}
	if err == nil {
		t.Fatal("LoadGemma4AssistantDir error = nil, want missing post_projection.weight")
	}
	if !core.Contains(err.Error(), "post_projection.weight") {
		t.Fatalf("LoadGemma4AssistantDir error = %v, want post_projection.weight", err)
	}
}

func TestLoadGemma4AssistantPairDirsValidatesTargetCompatibility(t *testing.T) {
	targetDir := writeNativeAssistantTargetDir(t, 8, []string{"sliding_attention", "full_attention"})
	assistantDir := writeNativeAssistantDir(t, nativeAssistantTinyTensors(true))

	pair, err := LoadGemma4AssistantPairDirs(targetDir, assistantDir)
	if err != nil {
		t.Fatalf("LoadGemma4AssistantPairDirs: %v", err)
	}
	defer pair.Close()

	if pair.TargetArch.Hidden != 8 || pair.TargetArch.Vocab != 8 {
		t.Fatalf("TargetArch = %+v, want hidden/vocab 8/8", pair.TargetArch)
	}
	if pair.Assistant == nil || pair.Assistant.NumLayers() != 2 {
		t.Fatalf("Assistant = %+v, want loaded two-layer assistant", pair.Assistant)
	}
}

func TestLoadGemma4AssistantPairDirsRejectsBackboneMismatch(t *testing.T) {
	targetDir := writeNativeAssistantTargetDir(t, 12, []string{"sliding_attention", "full_attention"})
	assistantDir := writeNativeAssistantDir(t, nativeAssistantTinyTensors(true))

	pair, err := LoadGemma4AssistantPairDirs(targetDir, assistantDir)
	if pair != nil {
		t.Fatalf("LoadGemma4AssistantPairDirs pair = %v, want nil on mismatch", pair)
	}
	if err == nil {
		t.Fatal("LoadGemma4AssistantPairDirs error = nil, want backbone mismatch")
	}
	if !core.Contains(err.Error(), "backbone_hidden_size") {
		t.Fatalf("LoadGemma4AssistantPairDirs error = %v, want backbone_hidden_size", err)
	}
}

func TestLoadGemma4AssistantPairDirsLoadsGGUFDrafter(t *testing.T) {
	targetDir := writeNativeAssistantTargetDir(t, 8, []string{"sliding_attention", "full_attention"})
	writeNativeAssistantTokenizer(t, targetDir)
	ggufPath := writeNativeAssistantGGUF(t, nativeAssistantTinyTensors(false))

	pair, err := LoadGemma4AssistantPairDirs(targetDir, ggufPath)
	if err != nil {
		t.Fatalf("LoadGemma4AssistantPairDirs(gguf): %v", err)
	}
	defer pair.Close()

	if pair.Assistant.Tokenizer() == nil {
		t.Fatal("GGUF assistant tokenizer = nil, want borrowed target tokenizer")
	}
	if pair.Assistant.Arch.Vocab != 8 || pair.Assistant.Arch.Hidden != 4 {
		t.Fatalf("GGUF assistant arch = %+v, want vocab/hidden 8/4", pair.Assistant.Arch)
	}
	if tensor, ok := pair.Assistant.Tensor("model.embed_tokens.weight"); !ok || tensor.Dtype != "BF16" || len(tensor.Shape) != 2 {
		t.Fatalf("GGUF mapped embed tensor = %+v ok=%v, want BF16 rank-2", tensor, ok)
	}
	if _, ok := pair.Assistant.Tensor("model.layers.0.layer_scalar.weight"); !ok {
		t.Fatal("GGUF layer_output_scale was not mapped to layer_scalar.weight")
	}
}

func TestGemma4AssistantTargetKVByLayerTypeResolvesSharedOwners(t *testing.T) {
	assistant := nativeAssistantTinyLoaded(t, true)
	defer assistant.Close()
	pair := &Gemma4AssistantPair{
		TargetArch: model.Arch{Hidden: 8, Vocab: 8, Layer: []model.LayerSpec{
			{Attention: model.SlidingAttention, KVShareFrom: 0, CacheIndex: 0},
			{Attention: model.GlobalAttention, KVShareFrom: 1, CacheIndex: 1},
			{Attention: model.SlidingAttention, KVShareFrom: 0, CacheIndex: -1},
			{Attention: model.GlobalAttention, KVShareFrom: 1, CacheIndex: -1},
		}},
		Assistant: assistant,
	}

	streams, err := pair.TargetKVByLayerType([]Gemma4AssistantTargetKV{
		nativeAssistantTargetKVFixture(0x11),
		nativeAssistantTargetKVFixture(0x22),
	})
	if err != nil {
		t.Fatalf("TargetKVByLayerType: %v", err)
	}

	sliding, ok := streams.Get("sliding_attention")
	if !ok || len(sliding.Key) == 0 || sliding.Key[0] != 0x11 {
		t.Fatalf("sliding stream = %+v, ok=%v; want cache 0", sliding, ok)
	}
	full, ok := streams.Get("full_attention")
	if !ok || len(full.Key) == 0 || full.Key[0] != 0x22 {
		t.Fatalf("full stream = %+v, ok=%v; want cache 1", full, ok)
	}
}

func TestGemma4AssistantTargetKVByLayerTypeRejectsMissingAssistantStream(t *testing.T) {
	assistant := nativeAssistantTinyLoaded(t, true)
	defer assistant.Close()
	pair := &Gemma4AssistantPair{
		TargetArch: model.Arch{Hidden: 8, Vocab: 8, Layer: []model.LayerSpec{
			{Attention: model.SlidingAttention, KVShareFrom: 0, CacheIndex: 0},
		}},
		Assistant: assistant,
	}

	_, err := pair.TargetKVByLayerType([]Gemma4AssistantTargetKV{nativeAssistantTargetKVFixture(0x11)})
	if err == nil {
		t.Fatal("TargetKVByLayerType error = nil, want missing full_attention stream")
	}
	if !core.Contains(err.Error(), "full_attention") {
		t.Fatalf("TargetKVByLayerType error = %v, want full_attention", err)
	}
}

func TestGemma4AssistantTargetKVByLayerTypeLastOwnerWins(t *testing.T) {
	assistant := nativeAssistantTinyLoaded(t, false)
	defer assistant.Close()
	pair := &Gemma4AssistantPair{
		TargetArch: model.Arch{Hidden: 8, Vocab: 8, Layer: []model.LayerSpec{
			{Attention: model.SlidingAttention, KVShareFrom: 0, CacheIndex: 0},
			{Attention: model.SlidingAttention, KVShareFrom: 1, CacheIndex: 1},
		}},
		Assistant: assistant,
	}
	pair.Assistant.Config.TextConfig.LayerTypes = []string{"sliding_attention", "sliding_attention"}

	streams, err := pair.TargetKVByLayerType([]Gemma4AssistantTargetKV{
		nativeAssistantTargetKVFixture(0x11),
		nativeAssistantTargetKVFixture(0x33),
	})
	if err != nil {
		t.Fatalf("TargetKVByLayerType: %v", err)
	}

	sliding, ok := streams.Get("sliding_attention")
	if !ok || len(sliding.Key) == 0 || sliding.Key[0] != 0x33 {
		t.Fatalf("sliding stream = %+v, ok=%v; want last owner cache 1", sliding, ok)
	}
}

func TestGemma4AssistantPairTargetKVByLayerTypeFromSessionTransposesResidentRows(t *testing.T) {
	assistant := nativeAssistantTinyLoaded(t, true)
	defer assistant.Close()

	arch := nativeAssistantSessionTargetArchForTest()
	rowBytes := 2 * 2 * bf16Size
	slidingKey := nativeAssistantSessionKVRowsForTest(4, 2, 2, 0x10)
	slidingValue := nativeAssistantSessionKVRowsForTest(4, 2, 2, 0x20)
	fullKey := nativeAssistantSessionKVRowsForTest(4, 2, 2, 0x30)
	fullValue := nativeAssistantSessionKVRowsForTest(4, 2, 2, 0x40)
	session := &ArchSession{
		arch: arch,
		state: archDecodeState{
			specs: arch.Layer,
		},
		stateBlockViews: []sessionStateLayerView{
			{
				layer: 0, kvHeads: 2, headDim: 2, rowBytes: rowBytes, cacheIndex: 0,
				cacheMode: nativeStateCacheModeFixed, cacheRows: 4, keyBytes: slidingKey, valueBytes: slidingValue,
			},
			{
				layer: 1, kvHeads: 2, headDim: 2, rowBytes: rowBytes, cacheIndex: 1,
				cacheMode: nativeStateCacheModeFixed, cacheRows: 4, keyBytes: fullKey, valueBytes: fullValue,
			},
		},
		pos:    3,
		maxLen: 4,
	}
	pair := &Gemma4AssistantPair{TargetArch: arch, Assistant: assistant}

	streams, err := pair.TargetKVByLayerTypeFromSession(session)
	if err != nil {
		t.Fatalf("TargetKVByLayerTypeFromSession: %v", err)
	}
	sliding, ok := streams.Get("sliding_attention")
	if !ok {
		t.Fatal("sliding_attention stream missing")
	}
	if sliding.Offset != 0 || sliding.Length != 3 || sliding.KVHeads != 2 || sliding.HeadDim != 2 {
		t.Fatalf("sliding stream = %+v, want offset 0 length 3 2x2 geometry", sliding)
	}
	if len(sliding.Key) != 3*rowBytes || len(sliding.Value) != 3*rowBytes {
		t.Fatalf("sliding stream bytes = %d/%d, want %d", len(sliding.Key), len(sliding.Value), 3*rowBytes)
	}
	if got := sliding.Key[0]; got != 0x10 {
		t.Fatalf("sliding head0 seq0 key = %#x, want token0/head0", got)
	}
	if got := sliding.Key[1*2*bf16Size]; got != 0x20 {
		t.Fatalf("sliding head0 seq1 key = %#x, want token1/head0", got)
	}
	if got := sliding.Key[3*2*bf16Size]; got != 0x11 {
		t.Fatalf("sliding head1 seq0 key = %#x, want token0/head1", got)
	}
	full, ok := streams.Get("full_attention")
	if !ok {
		t.Fatal("full_attention stream missing")
	}
	if full.Key[0] != 0x30 || full.Value[0] != 0x40 || full.Key[3*2*bf16Size] != 0x31 {
		t.Fatalf("full stream head-major bytes = %#x/%#x/%#x, want cache-index 1 rows transposed", full.Key[0], full.Value[0], full.Key[3*2*bf16Size])
	}
}

func TestGemma4AssistantPairTargetKVByLayerTypeFromSessionUsesSlidingWindowOffset(t *testing.T) {
	assistant := nativeAssistantTinyLoaded(t, true)
	defer assistant.Close()

	arch := nativeAssistantSessionTargetArchForTest()
	rowBytes := 2 * 2 * bf16Size
	slidingKey := make([]byte, 4*rowBytes)
	slidingValue := make([]byte, 4*rowBytes)
	for token := 2; token < 6; token++ {
		slot := token % 4
		slidingKey[slot*rowBytes] = byte(token)
		slidingValue[slot*rowBytes] = byte(token + 0x10)
	}
	session := &ArchSession{
		arch: arch,
		state: archDecodeState{
			specs: arch.Layer,
		},
		stateBlockViews: []sessionStateLayerView{
			{
				layer: 0, kvHeads: 2, headDim: 2, rowBytes: rowBytes, cacheIndex: 0,
				cacheMode: nativeStateCacheModeFixed, maxSize: 4, cacheRows: 4, keyBytes: slidingKey, valueBytes: slidingValue,
			},
			{
				layer: 1, kvHeads: 2, headDim: 2, rowBytes: rowBytes, cacheIndex: 1,
				cacheMode: nativeStateCacheModeFixed, cacheRows: 8,
				keyBytes:   nativeAssistantSessionRowsForTest(8, rowBytes, 0x30),
				valueBytes: nativeAssistantSessionRowsForTest(8, rowBytes, 0x40),
			},
		},
		pos:    6,
		maxLen: 8,
	}
	pair := &Gemma4AssistantPair{TargetArch: arch, Assistant: assistant}

	streams, err := pair.TargetKVByLayerTypeFromSession(session)
	if err != nil {
		t.Fatalf("TargetKVByLayerTypeFromSession: %v", err)
	}
	sliding, ok := streams.Get("sliding_attention")
	if !ok {
		t.Fatal("sliding_attention stream missing")
	}
	if sliding.Offset != 2 || sliding.Length != 4 {
		t.Fatalf("sliding stream offset/length = %d/%d, want 2/4", sliding.Offset, sliding.Length)
	}
	for row, want := range []byte{2, 3, 4, 5} {
		if got := sliding.Key[row*2*bf16Size]; got != want {
			t.Fatalf("sliding key head0 seq %d starts %#x, want token %#x", row, got, want)
		}
		if got := sliding.Value[row*2*bf16Size]; got != want+0x10 {
			t.Fatalf("sliding value head0 seq %d starts %#x, want token %#x", row, got, want+0x10)
		}
	}
}

func TestGemma4AssistantDraftInputProjectionMatchesReference(t *testing.T) {
	requireNativeRuntime(t)

	tensors := nativeAssistantTinyTensors(true)
	preW := nativeAssistantProjectionFixture(4, 16)
	tensors["pre_projection.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{4, 16}, Data: toBF16Bytes(preW)}
	dir := writeNativeAssistantDir(t, tensors)

	assistant, err := LoadGemma4AssistantDir(dir)
	if err != nil {
		t.Fatalf("LoadGemma4AssistantDir: %v", err)
	}
	defer assistant.Close()

	tokenEmbedding := toBF16Bytes([]float32{1, 2, -1, 0.5, 0.25, -0.5, 1.5, -2})
	previousHidden := toBF16Bytes([]float32{0.5, -1.5, 2, 1, -0.25, 0.75, -1, 0.125})
	got, err := assistant.DraftInputProjection(tokenEmbedding, previousHidden)
	if err != nil {
		t.Fatalf("DraftInputProjection: %v", err)
	}

	combined := append(append([]byte{}, tokenEmbedding...), previousHidden...)
	want := nativeAssistantMatMulBF16NTReference(combined, toBF16Bytes(preW), 1, 16, 4)
	assertFloat32Near(t, "draft input projection", bf16Floats(got), want, 0.02)
}

func TestGemma4AssistantPairDraftInputProjectionForTokenUsesScaledTargetEmbedding(t *testing.T) {
	requireNativeRuntime(t)

	targetDir := writeNativeAssistantTargetDir(t, 8, []string{"sliding_attention", "full_attention"})
	tensors := nativeAssistantTinyTensors(true)
	preW := nativeAssistantProjectionFixture(4, 16)
	tensors["pre_projection.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{4, 16}, Data: toBF16Bytes(preW)}
	assistantDir := writeNativeAssistantDir(t, tensors)

	pair, err := LoadGemma4AssistantPairDirs(targetDir, assistantDir)
	if err != nil {
		t.Fatalf("LoadGemma4AssistantPairDirs: %v", err)
	}
	defer pair.Close()

	targetEmbed := toBF16Bytes([]float32{
		0, 0, 0, 0, 0, 0, 0, 0,
		1, -0.5, 0.25, 2, -1, 0.75, 1.5, -2,
		0.5, 1, -1.5, 0, 0.125, -0.25, 2, -0.75,
		-1, 1.25, 0.5, -0.5, 2, 0, -2, 0.25,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
	})
	previousHidden := toBF16Bytes([]float32{0.5, -1.5, 2, 1, -0.25, 0.75, -1, 0.125})

	got, err := pair.DraftInputProjectionForToken(targetEmbed, 1, previousHidden)
	if err != nil {
		t.Fatalf("DraftInputProjectionForToken: %v", err)
	}

	embedding, err := EmbedTokensBF16(targetEmbed, []int32{1}, pair.TargetArch.Vocab, pair.TargetArch.Hidden, nativeGemma4EmbeddingScale(pair.TargetArch))
	if err != nil {
		t.Fatalf("EmbedTokensBF16 reference: %v", err)
	}
	combined := append(append([]byte{}, embedding[0]...), previousHidden...)
	want := nativeAssistantMatMulBF16NTReference(combined, toBF16Bytes(preW), 1, 16, 4)
	assertFloat32Near(t, "pair draft input projection for token", bf16Floats(got), want, 0.02)
}

func TestGemma4AssistantPairDraftInputProjectionForQuantTokenUsesScaledTargetEmbedding(t *testing.T) {
	requireNativeRuntime(t)

	targetDir := writeNativeAssistantTargetDir(t, 8, []string{"sliding_attention", "full_attention"})
	tensors := nativeAssistantTinyTensors(true)
	preW := nativeAssistantProjectionFixture(4, 16)
	tensors["pre_projection.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{4, 16}, Data: toBF16Bytes(preW)}
	assistantDir := writeNativeAssistantDir(t, tensors)

	pair, err := LoadGemma4AssistantPairDirs(targetDir, assistantDir)
	if err != nil {
		t.Fatalf("LoadGemma4AssistantPairDirs: %v", err)
	}
	defer pair.Close()

	const groupSize, bits = 4, 4
	packed, scales, biases := nativeAssistantQuantEmbeddingFixture(8, 8, groupSize)
	previousHidden := toBF16Bytes([]float32{0.5, -1.5, 2, 1, -0.25, 0.75, -1, 0.125})

	got, err := pair.DraftInputProjectionForTokenQuant(packed, scales, biases, groupSize, bits, 3, previousHidden)
	if err != nil {
		t.Fatalf("DraftInputProjectionForTokenQuant: %v", err)
	}

	embedding, err := EmbedTokensQuant(packed, scales, biases, []int32{3}, pair.TargetArch.Vocab, pair.TargetArch.Hidden, groupSize, bits, nativeGemma4EmbeddingScale(pair.TargetArch))
	if err != nil {
		t.Fatalf("EmbedTokensQuant reference: %v", err)
	}
	combined := append(append([]byte{}, embedding[0]...), previousHidden...)
	want := nativeAssistantMatMulBF16NTReference(combined, toBF16Bytes(preW), 1, 16, 4)
	assertFloat32Near(t, "pair draft input projection for quant token", bf16Floats(got), want, 0.02)
}

func TestGemma4AssistantDraftOutputProjectionMatchesReference(t *testing.T) {
	requireNativeRuntime(t)

	tensors := nativeAssistantTinyTensors(true)
	postW := nativeAssistantProjectionFixture(8, 4)
	tensors["post_projection.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{8, 4}, Data: toBF16Bytes(postW)}
	dir := writeNativeAssistantDir(t, tensors)

	assistant, err := LoadGemma4AssistantDir(dir)
	if err != nil {
		t.Fatalf("LoadGemma4AssistantDir: %v", err)
	}
	defer assistant.Close()

	assistantHidden := toBF16Bytes([]float32{1, -0.5, 0.25, 2})
	got, err := assistant.DraftOutputProjection(assistantHidden)
	if err != nil {
		t.Fatalf("DraftOutputProjection: %v", err)
	}

	want := nativeAssistantMatMulBF16NTReference(assistantHidden, toBF16Bytes(postW), 1, 4, 8)
	assertFloat32Near(t, "draft output projection", bf16Floats(got), want, 0.02)
}

func TestGemma4AssistantDraftFinalNormMatchesRMSNorm(t *testing.T) {
	requireNativeRuntime(t)

	tensors := nativeAssistantTinyTensors(true)
	normW := []float32{1, 0.75, 1.25, 0.5}
	tensors["model.norm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{4}, Data: toBF16Bytes(normW)}
	dir := writeNativeAssistantDir(t, tensors)

	assistant, err := LoadGemma4AssistantDir(dir)
	if err != nil {
		t.Fatalf("LoadGemma4AssistantDir: %v", err)
	}
	defer assistant.Close()

	hidden := toBF16Bytes([]float32{1, -0.5, 0.25, 2})
	got, err := assistant.DraftFinalNorm(hidden)
	if err != nil {
		t.Fatalf("DraftFinalNorm: %v", err)
	}
	want, err := RMSNormBF16(hidden, toBF16Bytes(normW), 1, 4, assistant.Arch.Eps)
	if err != nil {
		t.Fatalf("RMSNormBF16 reference: %v", err)
	}
	assertFloat32Near(t, "draft final norm", bf16Floats(got), bf16Floats(want), 0)
}

func TestGemma4AssistantDraftAttentionMatchesTargetKVPrimitivePath(t *testing.T) {
	requireNativeRuntime(t)

	const hidden, nHeads, kvHeads, headDim, kvLen = 128, 2, 2, 64, 3
	tensors := nativeAssistantAttentionTensors()
	qW := nativeAssistantProjectionFixture(nHeads*headDim, hidden)
	oW := nativeAssistantProjectionFixture(hidden, nHeads*headDim)
	qNorm := syntheticFloat32(headDim, 9)
	tensors["model.layers.0.self_attn.q_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{nHeads * headDim, hidden}, Data: toBF16Bytes(qW)}
	tensors["model.layers.0.self_attn.o_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden, nHeads * headDim}, Data: toBF16Bytes(oW)}
	tensors["model.layers.0.self_attn.q_norm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{headDim}, Data: toBF16Bytes(qNorm)}
	dir := writeNativeAssistantAttentionDir(t, tensors)

	assistant, err := LoadGemma4AssistantDir(dir)
	if err != nil {
		t.Fatalf("LoadGemma4AssistantDir: %v", err)
	}
	defer assistant.Close()

	x := toBF16Bytes(syntheticFloat32(hidden, 3))
	targetKV := Gemma4AssistantTargetKV{
		Key:     toBF16Bytes(syntheticFloat32(kvHeads*kvLen*headDim, 5)),
		Value:   toBF16Bytes(syntheticFloat32(kvHeads*kvLen*headDim, 7)),
		Offset:  2,
		Length:  kvLen,
		KVHeads: kvHeads,
		HeadDim: headDim,
	}

	got, err := assistant.DraftAttention(0, x, targetKV)
	if err != nil {
		t.Fatalf("DraftAttention: %v", err)
	}

	q, err := MatVecBF16(toBF16Bytes(qW), x, nHeads*headDim, hidden)
	if err != nil {
		t.Fatalf("MatVecBF16 q reference: %v", err)
	}
	q, err = RMSNormBF16(q, toBF16Bytes(qNorm), nHeads, headDim, assistant.Arch.Eps)
	if err != nil {
		t.Fatalf("RMSNormBF16 q reference: %v", err)
	}
	q, err = RoPEDimsBF16(q, 1, nHeads, headDim, headDim, assistant.Arch.RopeLocalBase, 1, targetKV.Offset, false)
	if err != nil {
		t.Fatalf("RoPEDimsBF16 q reference: %v", err)
	}
	attn, err := SDPA(q, targetKV.Key, targetKV.Value, 1, nHeads, kvHeads, headDim, targetKV.Length, nativeGemma4AssistantAttentionScale(assistant))
	if err != nil {
		t.Fatalf("SDPA reference: %v", err)
	}
	want, err := MatVecBF16(toBF16Bytes(oW), attn, hidden, nHeads*headDim)
	if err != nil {
		t.Fatalf("MatVecBF16 o reference: %v", err)
	}
	assertFloat32Near(t, "draft attention target kv path", bf16Floats(got), bf16Floats(want), 0)
}

func TestGemma4AssistantDraftLayerMatchesComposedPrimitivePath(t *testing.T) {
	requireNativeRuntime(t)

	const hidden, nHeads, kvHeads, headDim, kvLen, dFF = 128, 2, 2, 64, 3, 256
	tensors := nativeAssistantAttentionTensors()
	inputNorm := syntheticFloat32(hidden, 11)
	postAttnNorm := syntheticFloat32(hidden, 13)
	preFFNorm := syntheticFloat32(hidden, 17)
	postFFNorm := syntheticFloat32(hidden, 19)
	qW := nativeAssistantProjectionFixture(nHeads*headDim, hidden)
	oW := nativeAssistantProjectionFixture(hidden, nHeads*headDim)
	qNorm := syntheticFloat32(headDim, 23)
	gateW := nativeAssistantProjectionFixture(dFF, hidden)
	upW := nativeAssistantProjectionFixture(dFF, hidden)
	downW := nativeAssistantProjectionFixture(hidden, dFF)
	scalar := []float32{0.75}
	p := "model.layers.0"
	tensors[p+".input_layernorm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(inputNorm)}
	tensors[p+".post_attention_layernorm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(postAttnNorm)}
	tensors[p+".pre_feedforward_layernorm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(preFFNorm)}
	tensors[p+".post_feedforward_layernorm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(postFFNorm)}
	tensors[p+".layer_scalar"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{1}, Data: toBF16Bytes(scalar)}
	tensors[p+".self_attn.q_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{nHeads * headDim, hidden}, Data: toBF16Bytes(qW)}
	tensors[p+".self_attn.o_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden, nHeads * headDim}, Data: toBF16Bytes(oW)}
	tensors[p+".self_attn.q_norm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{headDim}, Data: toBF16Bytes(qNorm)}
	tensors[p+".mlp.gate_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{dFF, hidden}, Data: toBF16Bytes(gateW)}
	tensors[p+".mlp.up_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{dFF, hidden}, Data: toBF16Bytes(upW)}
	tensors[p+".mlp.down_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden, dFF}, Data: toBF16Bytes(downW)}
	dir := writeNativeAssistantAttentionDir(t, tensors)

	assistant, err := LoadGemma4AssistantDir(dir)
	if err != nil {
		t.Fatalf("LoadGemma4AssistantDir: %v", err)
	}
	defer assistant.Close()

	x := toBF16Bytes(syntheticFloat32(hidden, 29))
	targetKV := Gemma4AssistantTargetKV{
		Key:     toBF16Bytes(syntheticFloat32(kvHeads*kvLen*headDim, 31)),
		Value:   toBF16Bytes(syntheticFloat32(kvHeads*kvLen*headDim, 37)),
		Offset:  4,
		Length:  kvLen,
		KVHeads: kvHeads,
		HeadDim: headDim,
	}
	got, err := assistant.DraftLayer(0, x, targetKV)
	if err != nil {
		t.Fatalf("DraftLayer: %v", err)
	}

	normed, err := RMSNormBF16(x, toBF16Bytes(inputNorm), 1, hidden, assistant.Arch.Eps)
	if err != nil {
		t.Fatalf("input RMSNormBF16 reference: %v", err)
	}
	attnOut, err := assistant.DraftAttention(0, normed, targetKV)
	if err != nil {
		t.Fatalf("DraftAttention reference: %v", err)
	}
	attnResidual, err := RMSNormBF16(attnOut, toBF16Bytes(postAttnNorm), 1, hidden, assistant.Arch.Eps)
	if err != nil {
		t.Fatalf("post-attention RMSNormBF16 reference: %v", err)
	}
	h, err := AddBF16(x, attnResidual)
	if err != nil {
		t.Fatalf("attention residual AddBF16 reference: %v", err)
	}
	ffIn, err := RMSNormBF16(h, toBF16Bytes(preFFNorm), 1, hidden, assistant.Arch.Eps)
	if err != nil {
		t.Fatalf("pre-FF RMSNormBF16 reference: %v", err)
	}
	gate, err := MatVecBF16(toBF16Bytes(gateW), ffIn, dFF, hidden)
	if err != nil {
		t.Fatalf("gate MatVecBF16 reference: %v", err)
	}
	up, err := MatVecBF16(toBF16Bytes(upW), ffIn, dFF, hidden)
	if err != nil {
		t.Fatalf("up MatVecBF16 reference: %v", err)
	}
	gated, err := GeluGateMulBF16(gate, up)
	if err != nil {
		t.Fatalf("GeluGateMulBF16 reference: %v", err)
	}
	ff, err := MatVecBF16(toBF16Bytes(downW), gated, hidden, dFF)
	if err != nil {
		t.Fatalf("down MatVecBF16 reference: %v", err)
	}
	ffResidual, err := RMSNormBF16(ff, toBF16Bytes(postFFNorm), 1, hidden, assistant.Arch.Eps)
	if err != nil {
		t.Fatalf("post-FF RMSNormBF16 reference: %v", err)
	}
	want, err := AddBF16(h, ffResidual)
	if err != nil {
		t.Fatalf("FF residual AddBF16 reference: %v", err)
	}
	want, err = MulScalarBF16(want, toBF16Bytes(scalar))
	if err != nil {
		t.Fatalf("MulScalarBF16 reference: %v", err)
	}
	assertFloat32Near(t, "draft layer primitive path", bf16Floats(got), bf16Floats(want), 0)
}

func TestGemma4AssistantDraftStepActivationsRunsLayerStackAndPostProjection(t *testing.T) {
	requireNativeRuntime(t)

	const hidden, backbone, nHeads, kvHeads, headDim, kvLen, dFF = 128, 8, 2, 2, 64, 3, 256
	tensors := nativeAssistantAttentionTensors()
	p := "model.layers.0"
	tensors["model.norm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(syntheticFloat32(hidden, 41))}
	tensors["post_projection.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{backbone, hidden}, Data: toBF16Bytes(nativeAssistantProjectionFixture(backbone, hidden))}
	tensors[p+".input_layernorm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(syntheticFloat32(hidden, 43))}
	tensors[p+".post_attention_layernorm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(syntheticFloat32(hidden, 47))}
	tensors[p+".pre_feedforward_layernorm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(syntheticFloat32(hidden, 53))}
	tensors[p+".post_feedforward_layernorm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(syntheticFloat32(hidden, 59))}
	tensors[p+".layer_scalar"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{1}, Data: toBF16Bytes([]float32{0.5})}
	tensors[p+".self_attn.q_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{nHeads * headDim, hidden}, Data: toBF16Bytes(nativeAssistantProjectionFixture(nHeads*headDim, hidden))}
	tensors[p+".self_attn.o_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden, nHeads * headDim}, Data: toBF16Bytes(nativeAssistantProjectionFixture(hidden, nHeads*headDim))}
	tensors[p+".self_attn.q_norm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{headDim}, Data: toBF16Bytes(syntheticFloat32(headDim, 61))}
	tensors[p+".mlp.gate_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{dFF, hidden}, Data: toBF16Bytes(nativeAssistantProjectionFixture(dFF, hidden))}
	tensors[p+".mlp.up_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{dFF, hidden}, Data: toBF16Bytes(nativeAssistantProjectionFixture(dFF, hidden))}
	tensors[p+".mlp.down_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden, dFF}, Data: toBF16Bytes(nativeAssistantProjectionFixture(hidden, dFF))}
	dir := writeNativeAssistantAttentionDir(t, tensors)

	assistant, err := LoadGemma4AssistantDir(dir)
	if err != nil {
		t.Fatalf("LoadGemma4AssistantDir: %v", err)
	}
	defer assistant.Close()

	projectedHidden := toBF16Bytes(syntheticFloat32(hidden, 67))
	targetKV := Gemma4AssistantTargetKV{
		Key:     toBF16Bytes(syntheticFloat32(kvHeads*kvLen*headDim, 71)),
		Value:   toBF16Bytes(syntheticFloat32(kvHeads*kvLen*headDim, 73)),
		Offset:  5,
		Length:  kvLen,
		KVHeads: kvHeads,
		HeadDim: headDim,
	}
	targetKVs := Gemma4AssistantTargetKVByType{}
	targetKVs.set("sliding_attention", targetKV)

	gotNormed, gotHidden, err := assistant.DraftStepActivations(projectedHidden, targetKVs)
	if err != nil {
		t.Fatalf("DraftStepActivations: %v", err)
	}

	layerOut, err := assistant.DraftLayer(0, projectedHidden, targetKV)
	if err != nil {
		t.Fatalf("DraftLayer reference: %v", err)
	}
	wantNormed, err := assistant.DraftFinalNorm(layerOut)
	if err != nil {
		t.Fatalf("DraftFinalNorm reference: %v", err)
	}
	wantHidden, err := assistant.DraftOutputProjection(wantNormed)
	if err != nil {
		t.Fatalf("DraftOutputProjection reference: %v", err)
	}
	assertFloat32Near(t, "draft step normed activations", bf16Floats(gotNormed), bf16Floats(wantNormed), 0)
	assertFloat32Near(t, "draft step target hidden", bf16Floats(gotHidden), bf16Floats(wantHidden), 0)
}

func TestGemma4AssistantPairDraftStepUsesTokenAndTargetKVPath(t *testing.T) {
	requireNativeRuntime(t)

	const hidden, backbone, nHeads, kvHeads, headDim, kvLen, dFF, vocab = 128, 8, 2, 2, 64, 3, 256, 8
	targetDir := writeNativeAssistantAttentionTargetDir(t)
	tensors := nativeAssistantAttentionTensors()
	p := "model.layers.0"
	tensors["model.embed_tokens.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{vocab, hidden}, Data: toBF16Bytes(nativeAssistantProjectionFixture(vocab, hidden))}
	tensors["model.norm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(syntheticFloat32(hidden, 83))}
	tensors["pre_projection.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden, backbone * 2}, Data: toBF16Bytes(nativeAssistantProjectionFixture(hidden, backbone*2))}
	tensors["post_projection.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{backbone, hidden}, Data: toBF16Bytes(nativeAssistantProjectionFixture(backbone, hidden))}
	tensors[p+".input_layernorm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(syntheticFloat32(hidden, 89))}
	tensors[p+".post_attention_layernorm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(syntheticFloat32(hidden, 97))}
	tensors[p+".pre_feedforward_layernorm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(syntheticFloat32(hidden, 101))}
	tensors[p+".post_feedforward_layernorm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(syntheticFloat32(hidden, 103))}
	tensors[p+".layer_scalar"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{1}, Data: toBF16Bytes([]float32{0.625})}
	tensors[p+".self_attn.q_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{nHeads * headDim, hidden}, Data: toBF16Bytes(nativeAssistantProjectionFixture(nHeads*headDim, hidden))}
	tensors[p+".self_attn.o_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden, nHeads * headDim}, Data: toBF16Bytes(nativeAssistantProjectionFixture(hidden, nHeads*headDim))}
	tensors[p+".self_attn.q_norm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{headDim}, Data: toBF16Bytes(syntheticFloat32(headDim, 107))}
	tensors[p+".mlp.gate_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{dFF, hidden}, Data: toBF16Bytes(nativeAssistantProjectionFixture(dFF, hidden))}
	tensors[p+".mlp.up_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{dFF, hidden}, Data: toBF16Bytes(nativeAssistantProjectionFixture(dFF, hidden))}
	tensors[p+".mlp.down_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden, dFF}, Data: toBF16Bytes(nativeAssistantProjectionFixture(hidden, dFF))}
	assistantDir := writeNativeAssistantAttentionDir(t, tensors)

	pair, err := LoadGemma4AssistantPairDirs(targetDir, assistantDir)
	if err != nil {
		t.Fatalf("LoadGemma4AssistantPairDirs: %v", err)
	}
	defer pair.Close()

	targetEmbed := toBF16Bytes(syntheticFloat32(vocab*backbone, 109))
	previousHidden := toBF16Bytes(syntheticFloat32(backbone, 113))
	targetKV := Gemma4AssistantTargetKV{
		Key:     toBF16Bytes(syntheticFloat32(kvHeads*kvLen*headDim, 127)),
		Value:   toBF16Bytes(syntheticFloat32(kvHeads*kvLen*headDim, 131)),
		Offset:  6,
		Length:  kvLen,
		KVHeads: kvHeads,
		HeadDim: headDim,
	}
	targetKVs := Gemma4AssistantTargetKVByType{}
	targetKVs.set("sliding_attention", targetKV)

	got, err := pair.DraftStep(targetEmbed, 3, previousHidden, targetKVs)
	if err != nil {
		t.Fatalf("DraftStep: %v", err)
	}

	projected, err := pair.DraftInputProjectionForToken(targetEmbed, 3, previousHidden)
	if err != nil {
		t.Fatalf("DraftInputProjectionForToken reference: %v", err)
	}
	normed, hiddenOut, err := pair.Assistant.DraftStepActivations(projected, targetKVs)
	if err != nil {
		t.Fatalf("DraftStepActivations reference: %v", err)
	}
	logits, err := pair.Assistant.DraftLogits(normed)
	if err != nil {
		t.Fatalf("DraftLogits reference: %v", err)
	}
	token, err := pair.Assistant.DraftGreedyToken(logits)
	if err != nil {
		t.Fatalf("DraftGreedyToken reference: %v", err)
	}
	if got.Token != token {
		t.Fatalf("DraftStep token = %d, want %d", got.Token, token)
	}
	assertFloat32Near(t, "draft step logits", bf16Floats(got.Logits), bf16Floats(logits), 0)
	assertFloat32Near(t, "draft step hidden", bf16Floats(got.Hidden), bf16Floats(hiddenOut), 0)
}

func TestGemma4AssistantPairDraftStepQuantUsesTokenAndTargetKVPath(t *testing.T) {
	requireNativeRuntime(t)

	const hidden, backbone, nHeads, kvHeads, headDim, kvLen, dFF, vocab = 128, 8, 2, 2, 64, 3, 256, 8
	targetDir := writeNativeAssistantAttentionTargetDir(t)
	tensors := nativeAssistantAttentionTensors()
	p := "model.layers.0"
	tensors["model.embed_tokens.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{vocab, hidden}, Data: toBF16Bytes(nativeAssistantProjectionFixture(vocab, hidden))}
	tensors["model.norm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(syntheticFloat32(hidden, 137))}
	tensors["pre_projection.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden, backbone * 2}, Data: toBF16Bytes(nativeAssistantProjectionFixture(hidden, backbone*2))}
	tensors["post_projection.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{backbone, hidden}, Data: toBF16Bytes(nativeAssistantProjectionFixture(backbone, hidden))}
	tensors[p+".input_layernorm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(syntheticFloat32(hidden, 139))}
	tensors[p+".post_attention_layernorm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(syntheticFloat32(hidden, 149))}
	tensors[p+".pre_feedforward_layernorm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(syntheticFloat32(hidden, 151))}
	tensors[p+".post_feedforward_layernorm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(syntheticFloat32(hidden, 157))}
	tensors[p+".layer_scalar"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{1}, Data: toBF16Bytes([]float32{0.875})}
	tensors[p+".self_attn.q_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{nHeads * headDim, hidden}, Data: toBF16Bytes(nativeAssistantProjectionFixture(nHeads*headDim, hidden))}
	tensors[p+".self_attn.o_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden, nHeads * headDim}, Data: toBF16Bytes(nativeAssistantProjectionFixture(hidden, nHeads*headDim))}
	tensors[p+".self_attn.q_norm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{headDim}, Data: toBF16Bytes(syntheticFloat32(headDim, 163))}
	tensors[p+".mlp.gate_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{dFF, hidden}, Data: toBF16Bytes(nativeAssistantProjectionFixture(dFF, hidden))}
	tensors[p+".mlp.up_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{dFF, hidden}, Data: toBF16Bytes(nativeAssistantProjectionFixture(dFF, hidden))}
	tensors[p+".mlp.down_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden, dFF}, Data: toBF16Bytes(nativeAssistantProjectionFixture(hidden, dFF))}
	assistantDir := writeNativeAssistantAttentionDir(t, tensors)

	pair, err := LoadGemma4AssistantPairDirs(targetDir, assistantDir)
	if err != nil {
		t.Fatalf("LoadGemma4AssistantPairDirs: %v", err)
	}
	defer pair.Close()

	const groupSize, bits = 4, 4
	packed, scales, biases := nativeAssistantQuantEmbeddingFixture(vocab, backbone, groupSize)
	previousHidden := toBF16Bytes(syntheticFloat32(backbone, 167))
	targetKV := Gemma4AssistantTargetKV{
		Key:     toBF16Bytes(syntheticFloat32(kvHeads*kvLen*headDim, 173)),
		Value:   toBF16Bytes(syntheticFloat32(kvHeads*kvLen*headDim, 179)),
		Offset:  7,
		Length:  kvLen,
		KVHeads: kvHeads,
		HeadDim: headDim,
	}
	targetKVs := Gemma4AssistantTargetKVByType{}
	targetKVs.set("sliding_attention", targetKV)

	got, err := pair.DraftStepQuant(packed, scales, biases, groupSize, bits, 4, previousHidden, targetKVs)
	if err != nil {
		t.Fatalf("DraftStepQuant: %v", err)
	}

	projected, err := pair.DraftInputProjectionForTokenQuant(packed, scales, biases, groupSize, bits, 4, previousHidden)
	if err != nil {
		t.Fatalf("DraftInputProjectionForTokenQuant reference: %v", err)
	}
	normed, hiddenOut, err := pair.Assistant.DraftStepActivations(projected, targetKVs)
	if err != nil {
		t.Fatalf("DraftStepActivations reference: %v", err)
	}
	logits, err := pair.Assistant.DraftLogits(normed)
	if err != nil {
		t.Fatalf("DraftLogits reference: %v", err)
	}
	token, err := pair.Assistant.DraftGreedyToken(logits)
	if err != nil {
		t.Fatalf("DraftGreedyToken reference: %v", err)
	}
	if got.Token != token {
		t.Fatalf("DraftStepQuant token = %d, want %d", got.Token, token)
	}
	assertFloat32Near(t, "draft step quant logits", bf16Floats(got.Logits), bf16Floats(logits), 0)
	assertFloat32Near(t, "draft step quant hidden", bf16Floats(got.Hidden), bf16Floats(hiddenOut), 0)
}

func TestGemma4AssistantPairDraftStepFromSessionMatchesExplicitPath(t *testing.T) {
	requireNativeRuntime(t)

	targetDir := writeNativeAssistantAttentionTargetDir(t)
	assistantDir := writeNativeAssistantAttentionDir(t, nativeAssistantAttentionTensors())
	pair, err := LoadGemma4AssistantPairDirs(targetDir, assistantDir)
	if err != nil {
		t.Fatalf("LoadGemma4AssistantPairDirs: %v", err)
	}
	defer pair.Close()

	arch := pair.TargetArch
	kvHeads := arch.Layer[0].KVHeads
	if kvHeads <= 0 {
		kvHeads = arch.KVHeads
	}
	headDim := arch.Layer[0].HeadDim
	if headDim <= 0 {
		headDim = arch.HeadDim
	}
	rowBytes := kvHeads * headDim * bf16Size
	tokenEmbedding := toBF16Bytes(syntheticFloat32(arch.Hidden, 83))
	retainedHidden := toBF16Bytes(syntheticFloat32(arch.Hidden, 89))
	finalNorm := toBF16Bytes(syntheticFloat32(arch.Hidden, 97))
	session := &ArchSession{
		arch: arch,
		state: archDecodeState{
			specs: arch.Layer,
		},
		stateBlockViews: []sessionStateLayerView{
			{
				layer: 0, kvHeads: kvHeads, headDim: headDim, rowBytes: rowBytes, cacheIndex: 0,
				cacheMode: nativeStateCacheModeFixed, cacheRows: 4,
				keyBytes:   nativeAssistantSessionKVRowsForTest(4, kvHeads, headDim, 0x10),
				valueBytes: nativeAssistantSessionKVRowsForTest(4, kvHeads, headDim, 0x20),
			},
		},
		pos:            3,
		maxLen:         4,
		retainedHidden: retainedHidden,
		finalNorm:      finalNorm,
	}
	session.embedInto = func(dst []byte, id int32) ([]byte, error) {
		if id != 5 {
			return nil, core.NewError("unexpected token id")
		}
		if len(dst) < len(tokenEmbedding) {
			return nil, core.NewError("short embedding destination")
		}
		copy(dst, tokenEmbedding)
		return dst[:len(tokenEmbedding)], nil
	}

	targetKVs, err := pair.TargetKVByLayerTypeFromSession(session)
	if err != nil {
		t.Fatalf("TargetKVByLayerTypeFromSession: %v", err)
	}
	previousHidden, err := RMSNormBF16(retainedHidden, finalNorm, 1, arch.Hidden, arch.Eps)
	if err != nil {
		t.Fatalf("RMSNormBF16 boundary reference: %v", err)
	}
	projected, err := pair.Assistant.DraftInputProjection(tokenEmbedding, previousHidden)
	if err != nil {
		t.Fatalf("DraftInputProjection reference: %v", err)
	}
	want, err := pair.draftStepFromProjected(projected, targetKVs)
	if err != nil {
		t.Fatalf("draftStepFromProjected reference: %v", err)
	}

	got, err := pair.DraftStepFromSession(session, 5)
	if err != nil {
		t.Fatalf("DraftStepFromSession: %v", err)
	}
	if got.Token != want.Token {
		t.Fatalf("DraftStepFromSession token = %d, want %d", got.Token, want.Token)
	}
	eqBytes(t, "DraftStepFromSession logits", got.Logits, want.Logits)
	eqBytes(t, "DraftStepFromSession hidden", got.Hidden, want.Hidden)
}

func TestGemma4AssistantPairDraftBlockFromSessionMatchesRepeatedSteps(t *testing.T) {
	requireNativeRuntime(t)

	targetDir := writeNativeAssistantAttentionTargetDir(t)
	assistantDir := writeNativeAssistantAttentionDir(t, nativeAssistantAttentionTensors())
	pair, err := LoadGemma4AssistantPairDirs(targetDir, assistantDir)
	if err != nil {
		t.Fatalf("LoadGemma4AssistantPairDirs: %v", err)
	}
	defer pair.Close()

	arch := pair.TargetArch
	kvHeads := arch.Layer[0].KVHeads
	if kvHeads <= 0 {
		kvHeads = arch.KVHeads
	}
	headDim := arch.Layer[0].HeadDim
	if headDim <= 0 {
		headDim = arch.HeadDim
	}
	rowBytes := kvHeads * headDim * bf16Size
	retainedHidden := toBF16Bytes(syntheticFloat32(arch.Hidden, 191))
	finalNorm := toBF16Bytes(syntheticFloat32(arch.Hidden, 193))
	session := &ArchSession{
		arch: arch,
		state: archDecodeState{
			specs: arch.Layer,
		},
		stateBlockViews: []sessionStateLayerView{
			{
				layer: 0, kvHeads: kvHeads, headDim: headDim, rowBytes: rowBytes, cacheIndex: 0,
				cacheMode: nativeStateCacheModeFixed, cacheRows: 4,
				keyBytes:   nativeAssistantSessionKVRowsForTest(4, kvHeads, headDim, 0x30),
				valueBytes: nativeAssistantSessionKVRowsForTest(4, kvHeads, headDim, 0x40),
			},
		},
		pos:            3,
		maxLen:         4,
		retainedHidden: retainedHidden,
		finalNorm:      finalNorm,
	}
	session.embedInto = func(dst []byte, id int32) ([]byte, error) {
		if len(dst) < arch.Hidden*bf16Size {
			return nil, core.NewError("short embedding destination")
		}
		embedding := toBF16Bytes(syntheticFloat32(arch.Hidden, int(197+id)))
		copy(dst, embedding)
		return dst[:len(embedding)], nil
	}

	got, err := pair.DraftBlockFromSession(session, 5, 2)
	if err != nil {
		t.Fatalf("DraftBlockFromSession: %v", err)
	}

	targetKVs, err := pair.TargetKVByLayerTypeFromSession(session)
	if err != nil {
		t.Fatalf("TargetKVByLayerTypeFromSession: %v", err)
	}
	currentHidden, err := session.BoundaryNormedHidden()
	if err != nil {
		t.Fatalf("BoundaryNormedHidden: %v", err)
	}
	currentToken := int32(5)
	wantTokens := make([]int32, 0, 2)
	for len(wantTokens) < 2 {
		tokenEmbedding, err := session.embedID(currentToken)
		if err != nil {
			t.Fatalf("embedID reference: %v", err)
		}
		projected, err := pair.Assistant.DraftInputProjection(tokenEmbedding, currentHidden)
		if err != nil {
			t.Fatalf("DraftInputProjection reference: %v", err)
		}
		step, err := pair.draftStepFromProjected(projected, targetKVs)
		if err != nil {
			t.Fatalf("draftStepFromProjected reference: %v", err)
		}
		wantTokens = append(wantTokens, step.Token)
		currentToken = step.Token
		currentHidden = step.Hidden
	}
	if !idsEqual(got.Tokens, wantTokens) {
		t.Fatalf("DraftBlockFromSession tokens = %v, want %v", got.Tokens, wantTokens)
	}
	eqBytes(t, "DraftBlockFromSession hidden", got.Hidden, currentHidden)
}

func TestGemma4AssistantPairVerifyDraftBlockFromSessionAcceptsFullBlock(t *testing.T) {
	requireNativeRuntime(t)

	mk := newMTPDecodeFixture(t)
	prompt := []int32{1, 5, 3}
	want, err := mk().Generate(prompt, 3, -1)
	if err != nil {
		t.Fatalf("reference Generate: %v", err)
	}
	target := mk()
	if err := target.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	pair := &Gemma4AssistantPair{TargetArch: target.arch}

	got, err := pair.VerifyDraftBlockFromSession(target, want[:2])
	if err != nil {
		t.Fatalf("VerifyDraftBlockFromSession: %v", err)
	}

	if !got.AllAccepted || got.AcceptedCount != 2 || got.RejectedCount != 0 {
		t.Fatalf("verify counts allAccepted=%v accepted=%d rejected=%d, want true/2/0", got.AllAccepted, got.AcceptedCount, got.RejectedCount)
	}
	if !idsEqual(got.DraftedTokens, want[:2]) || !idsEqual(got.AcceptedTokens, want[:2]) || len(got.RejectedTokens) != 0 {
		t.Fatalf("verify tokens drafted=%v accepted=%v rejected=%v, want accepted %v", got.DraftedTokens, got.AcceptedTokens, got.RejectedTokens, want[:2])
	}
	if !idsEqual(got.TargetTokens, []int32{want[0]}) {
		t.Fatalf("verify target tokens = %v, want [%d]", got.TargetTokens, want[0])
	}
	if target.Pos() != len(prompt)+2 {
		t.Fatalf("target Pos after verify = %d, want %d", target.Pos(), len(prompt)+2)
	}
	if got.ReplacementToken != 0 {
		t.Fatalf("ReplacementToken = %d, want 0 when all accepted", got.ReplacementToken)
	}
	if len(got.Hidden) != target.arch.Hidden*bf16Size {
		t.Fatalf("Hidden bytes = %d, want %d", len(got.Hidden), target.arch.Hidden*bf16Size)
	}
	if len(got.Logits) != target.arch.Vocab*bf16Size {
		t.Fatalf("Logits bytes = %d, want %d", len(got.Logits), target.arch.Vocab*bf16Size)
	}
}

func TestGemma4AssistantPairVerifyDraftBlockFromSessionRejectsSuffixAndRestoresAcceptedBoundary(t *testing.T) {
	requireNativeRuntime(t)

	mk := newMTPDecodeFixture(t)
	prompt := []int32{1, 5, 3}
	want, err := mk().Generate(prompt, 4, -1)
	if err != nil {
		t.Fatalf("reference Generate: %v", err)
	}
	badSecond := nativeAssistantWrongToken(want[1])
	target := mk()
	if err := target.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	pair := &Gemma4AssistantPair{TargetArch: target.arch}

	got, err := pair.VerifyDraftBlockFromSession(target, []int32{want[0], badSecond})
	if err != nil {
		t.Fatalf("VerifyDraftBlockFromSession: %v", err)
	}

	if got.AllAccepted || got.AcceptedCount != 1 || got.RejectedCount != 1 {
		t.Fatalf("verify counts allAccepted=%v accepted=%d rejected=%d, want false/1/1", got.AllAccepted, got.AcceptedCount, got.RejectedCount)
	}
	if !idsEqual(got.AcceptedTokens, []int32{want[0]}) || !idsEqual(got.RejectedTokens, []int32{badSecond}) {
		t.Fatalf("verify accepted=%v rejected=%v, want [%d]/[%d]", got.AcceptedTokens, got.RejectedTokens, want[0], badSecond)
	}
	if got.ReplacementToken != want[1] {
		t.Fatalf("ReplacementToken = %d, want %d", got.ReplacementToken, want[1])
	}
	if target.Pos() != len(prompt)+1 {
		t.Fatalf("target Pos after verify = %d, want %d", target.Pos(), len(prompt)+1)
	}
	if len(got.Hidden) != target.arch.Hidden*bf16Size {
		t.Fatalf("Hidden bytes = %d, want %d", len(got.Hidden), target.arch.Hidden*bf16Size)
	}
	continued, err := target.GenerateFromCache(2, -1)
	if err != nil {
		t.Fatalf("GenerateFromCache after verify: %v", err)
	}
	wantContinued, err := mk().Generate(append(append([]int32{}, prompt...), want[0]), 2, -1)
	if err != nil {
		t.Fatalf("reference continuation: %v", err)
	}
	if !idsEqual(continued, wantContinued) {
		t.Fatalf("continuation after rollback = %v, want %v", continued, wantContinued)
	}
}

func TestGemma4AssistantPairVerifyDraftBlockFromSessionRejectsFirstTokenAndRestoresPromptBoundary(t *testing.T) {
	requireNativeRuntime(t)

	mk := newMTPDecodeFixture(t)
	prompt := []int32{1, 5, 3}
	want, err := mk().Generate(prompt, 3, -1)
	if err != nil {
		t.Fatalf("reference Generate: %v", err)
	}
	badFirst := nativeAssistantWrongToken(want[0])
	target := mk()
	if err := target.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	pair := &Gemma4AssistantPair{TargetArch: target.arch}

	got, err := pair.VerifyDraftBlockFromSession(target, []int32{badFirst})
	if err != nil {
		t.Fatalf("VerifyDraftBlockFromSession: %v", err)
	}

	if got.AllAccepted || got.AcceptedCount != 0 || got.RejectedCount != 1 {
		t.Fatalf("verify counts allAccepted=%v accepted=%d rejected=%d, want false/0/1", got.AllAccepted, got.AcceptedCount, got.RejectedCount)
	}
	if len(got.AcceptedTokens) != 0 || !idsEqual(got.RejectedTokens, []int32{badFirst}) {
		t.Fatalf("verify accepted=%v rejected=%v, want none/[%d]", got.AcceptedTokens, got.RejectedTokens, badFirst)
	}
	if got.ReplacementToken != want[0] {
		t.Fatalf("ReplacementToken = %d, want %d", got.ReplacementToken, want[0])
	}
	if target.Pos() != len(prompt) {
		t.Fatalf("target Pos after verify = %d, want %d", target.Pos(), len(prompt))
	}
	if len(got.Hidden) != 0 {
		t.Fatalf("Hidden bytes = %d, want 0 when no draft token is accepted", len(got.Hidden))
	}
	continued, err := target.GenerateFromCache(2, -1)
	if err != nil {
		t.Fatalf("GenerateFromCache after verify: %v", err)
	}
	wantContinued, err := mk().Generate(prompt, 2, -1)
	if err != nil {
		t.Fatalf("reference continuation: %v", err)
	}
	if !idsEqual(continued, wantContinued) {
		t.Fatalf("continuation after full rollback = %v, want %v", continued, wantContinued)
	}
}

func TestGemma4AssistantPairVerifyDraftBlockSampledFromSessionUsesTargetSampler(t *testing.T) {
	requireNativeRuntime(t)

	pair, mk := newNativeAssistantGenerateFixture(t)
	defer pair.Close()
	params := model.SampleParams{Temperature: 1.5}
	prompt, seed, sampled, badDraft := nativeAssistantSampledVerifierRejectFixture(t, mk, params)
	target := mk()
	if err := target.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}

	got, err := pair.VerifyDraftBlockSampledFromSession(target, []int32{badDraft}, model.NewSampler(seed), params, false)
	if err != nil {
		t.Fatalf("VerifyDraftBlockSampledFromSession: %v", err)
	}

	if got.AllAccepted || got.AcceptedCount != 0 || got.RejectedCount != 1 {
		t.Fatalf("sampled verify counts allAccepted=%v accepted=%d rejected=%d, want false/0/1", got.AllAccepted, got.AcceptedCount, got.RejectedCount)
	}
	if got.ReplacementToken != sampled {
		t.Fatalf("sampled replacement = %d, want target sampled token %d", got.ReplacementToken, sampled)
	}
	if !idsEqual(got.TargetTokens, []int32{sampled}) {
		t.Fatalf("sampled target tokens = %v, want [%d]", got.TargetTokens, sampled)
	}
	if target.Pos() != len(prompt) {
		t.Fatalf("target Pos after sampled reject = %d, want %d", target.Pos(), len(prompt))
	}
	if len(got.Hidden) != 0 {
		t.Fatalf("sampled reject hidden bytes = %d, want 0 when no draft token is accepted", len(got.Hidden))
	}
	if len(got.Logits) != target.arch.Vocab*bf16Size {
		t.Fatalf("sampled reject logits bytes = %d, want %d", len(got.Logits), target.arch.Vocab*bf16Size)
	}
}

func TestGemma4AssistantPairGenerateFromSessionMatchesTargetGenerate(t *testing.T) {
	requireNativeRuntime(t)

	pair, mk := newNativeAssistantGenerateFixture(t)
	defer pair.Close()
	prompt := []int32{1, 5, 3}
	maxNew := 4
	want, err := mk().Generate(prompt, maxNew, -1)
	if err != nil {
		t.Fatalf("reference Generate: %v", err)
	}
	target := mk()

	got, err := pair.GenerateFromSession(target, prompt, maxNew, -1, 2, nil)
	if err != nil {
		t.Fatalf("GenerateFromSession: %v", err)
	}

	if !idsEqual(got.Tokens, want) {
		t.Fatalf("GenerateFromSession tokens = %v, want %v", got.Tokens, want)
	}
	if target.Pos() != len(prompt)+len(want) {
		t.Fatalf("target Pos after GenerateFromSession = %d, want %d", target.Pos(), len(prompt)+len(want))
	}
	if got.PromptTokens != len(prompt) || got.TargetTokens != len(want) {
		t.Fatalf("generate token counts prompt=%d target=%d, want %d/%d", got.PromptTokens, got.TargetTokens, len(prompt), len(want))
	}
	if got.DraftCalls == 0 || got.TargetVerifyCalls == 0 || got.DraftTokens == 0 {
		t.Fatalf("generate counters draftCalls=%d verifyCalls=%d draftTokens=%d, want non-zero speculative path", got.DraftCalls, got.TargetVerifyCalls, got.DraftTokens)
	}
	for _, n := range got.DraftTokenSchedule {
		if n <= 0 || n > 2 {
			t.Fatalf("draft schedule entry = %d, want 1..2", n)
		}
	}
}

func TestGemma4AssistantPairGenerateFromSessionUsesExactWarmPromptCache(t *testing.T) {
	requireNativeRuntime(t)

	pair, mk := newNativeAssistantGenerateFixture(t)
	defer pair.Close()
	prompt := []int32{1, 5, 3}
	maxNew := 4
	want, err := mk().Generate(prompt, maxNew, -1)
	if err != nil {
		t.Fatalf("reference Generate: %v", err)
	}
	target := mk()
	if err := target.WarmPromptCache(prompt); err != nil {
		t.Fatalf("WarmPromptCache: %v", err)
	}
	if hit := target.CachedPrefixLen(prompt); hit != len(prompt) {
		t.Fatalf("warm CachedPrefixLen = %d, want exact prompt hit %d", hit, len(prompt))
	}

	got, err := pair.GenerateFromSession(target, prompt, maxNew, -1, 2, nil)
	if err != nil {
		t.Fatalf("GenerateFromSession after WarmPromptCache: %v", err)
	}

	if !idsEqual(got.Tokens, want) {
		t.Fatalf("GenerateFromSession warm tokens = %v, want %v", got.Tokens, want)
	}
	if hit := target.CachedPrefixLen(prompt); hit != len(prompt) {
		t.Fatalf("CachedPrefixLen after assistant generate = %d, want exact prompt hit %d retained", hit, len(prompt))
	}
}

func TestGemma4AssistantPairGenerateFromSessionUsesWarmPromptPrefix(t *testing.T) {
	requireNativeRuntime(t)

	pair, mk := newNativeAssistantGenerateFixture(t)
	defer pair.Close()
	shared := []int32{1, 5}
	prompt := []int32{1, 5, 3}
	maxNew := 4
	want, err := mk().Generate(prompt, maxNew, -1)
	if err != nil {
		t.Fatalf("reference Generate: %v", err)
	}
	target := mk()
	if err := target.WarmPromptCache(shared); err != nil {
		t.Fatalf("WarmPromptCache(shared): %v", err)
	}
	if hit := target.CachedPrefixLen(prompt); hit != len(shared) {
		t.Fatalf("warm CachedPrefixLen(full prompt) = %d, want shared prefix hit %d", hit, len(shared))
	}

	got, err := pair.GenerateFromSession(target, prompt, maxNew, -1, 2, nil)
	if err != nil {
		t.Fatalf("GenerateFromSession after shared WarmPromptCache: %v", err)
	}

	if !idsEqual(got.Tokens, want) {
		t.Fatalf("GenerateFromSession shared-prefix tokens = %v, want %v", got.Tokens, want)
	}
	if hit := target.CachedPrefixLen(prompt); hit != len(prompt) {
		t.Fatalf("CachedPrefixLen after shared-prefix assistant generate = %d, want exact prompt hit %d retained", hit, len(prompt))
	}
}

func TestGemma4AssistantPairGenerateFromSessionStopsWhenYieldReturnsFalse(t *testing.T) {
	requireNativeRuntime(t)

	pair, mk := newNativeAssistantGenerateFixture(t)
	defer pair.Close()
	prompt := nativeAssistantPromptWhoseFirstTargetTokenIsNot(t, mk, 0)
	target := mk()
	var yielded []int32

	got, err := pair.GenerateFromSessionEach(target, prompt, 4, -1, 2, nil, func(id int32) bool {
		yielded = append(yielded, id)
		return false
	})
	if err != nil {
		t.Fatalf("GenerateFromSessionEach: %v", err)
	}

	if len(got.Tokens) != 1 || len(yielded) != 1 || got.Tokens[0] != yielded[0] {
		t.Fatalf("yield stop tokens got=%v yielded=%v, want one matching token", got.Tokens, yielded)
	}
	if target.Pos() != len(prompt) {
		t.Fatalf("target Pos after replacement yield stop = %d, want unforwarded carry position %d", target.Pos(), len(prompt))
	}
}

func TestGemma4AssistantPairGenerateSampledFromSessionMatchesTargetGenerateSampled(t *testing.T) {
	requireNativeRuntime(t)

	pair, mk := newNativeAssistantGenerateFixture(t)
	defer pair.Close()
	params := model.SampleParams{Temperature: 1.5}
	prompt, seed, _, _ := nativeAssistantSampledVerifierRejectFixture(t, mk, params)
	maxNew := 4
	want, err := mk().GenerateSampledEach(prompt, maxNew, nil, model.NewSampler(seed), params, nil, nil)
	if err != nil {
		t.Fatalf("reference GenerateSampledEach: %v", err)
	}
	target := mk()

	got, err := pair.GenerateSampledFromSession(target, prompt, maxNew, nil, model.NewSampler(seed), params, 2)
	if err != nil {
		t.Fatalf("GenerateSampledFromSession: %v", err)
	}

	if !idsEqual(got.Tokens, want) {
		t.Fatalf("GenerateSampledFromSession tokens = %v, want %v", got.Tokens, want)
	}
	if target.Pos() != len(prompt)+len(want) {
		t.Fatalf("target Pos after GenerateSampledFromSession = %d, want %d", target.Pos(), len(prompt)+len(want))
	}
	if got.DraftCalls == 0 || got.TargetVerifyCalls == 0 || got.DraftTokens == 0 {
		t.Fatalf("sampled counters draftCalls=%d verifyCalls=%d draftTokens=%d, want non-zero speculative path", got.DraftCalls, got.TargetVerifyCalls, got.DraftTokens)
	}
}

func TestGemma4AssistantPairGenerateSampledFromSessionEachKeepsDraftBlockWhileStreaming(t *testing.T) {
	requireNativeRuntime(t)

	pair, mk := newNativeAssistantGenerateFixture(t)
	defer pair.Close()
	params := model.SampleParams{Temperature: 1.5}
	prompt, seed, _, _ := nativeAssistantSampledVerifierRejectFixture(t, mk, params)
	maxNew := 4
	want, err := mk().GenerateSampledEach(prompt, maxNew, nil, model.NewSampler(seed), params, nil, nil)
	if err != nil {
		t.Fatalf("reference GenerateSampledEach: %v", err)
	}
	target := mk()
	var yielded []int32

	got, err := pair.GenerateSampledFromSessionEach(target, prompt, maxNew, nil, model.NewSampler(seed), params, 2, func(id int32) bool {
		yielded = append(yielded, id)
		return true
	})
	if err != nil {
		t.Fatalf("GenerateSampledFromSessionEach: %v", err)
	}

	if !idsEqual(got.Tokens, want) {
		t.Fatalf("streaming sampled assistant tokens = %v, want %v", got.Tokens, want)
	}
	if !idsEqual(yielded, got.Tokens) {
		t.Fatalf("streaming sampled assistant yielded %v, want result tokens %v", yielded, got.Tokens)
	}
	hasBlock := false
	for _, n := range got.DraftTokenSchedule {
		if n > 1 {
			hasBlock = true
			break
		}
	}
	if !hasBlock {
		t.Fatalf("streaming sampled assistant draft schedule = %v, want a multi-token verify block", got.DraftTokenSchedule)
	}
}

func TestGemma4AssistantDraftInputProjectionRejectsBadHidden(t *testing.T) {
	tensors := nativeAssistantTinyTensors(true)
	dir := writeNativeAssistantDir(t, tensors)

	assistant, err := LoadGemma4AssistantDir(dir)
	if err != nil {
		t.Fatalf("LoadGemma4AssistantDir: %v", err)
	}
	defer assistant.Close()

	_, err = assistant.DraftInputProjection(make([]byte, 8*bf16Size), make([]byte, 7*bf16Size))
	if err == nil {
		t.Fatal("DraftInputProjection error = nil, want previous hidden length error")
	}
	if !core.Contains(err.Error(), "previous hidden") {
		t.Fatalf("DraftInputProjection error = %v, want previous hidden", err)
	}
}

func TestGemma4AssistantDraftOutputProjectionRejectsBadHidden(t *testing.T) {
	tensors := nativeAssistantTinyTensors(true)
	dir := writeNativeAssistantDir(t, tensors)

	assistant, err := LoadGemma4AssistantDir(dir)
	if err != nil {
		t.Fatalf("LoadGemma4AssistantDir: %v", err)
	}
	defer assistant.Close()

	_, err = assistant.DraftOutputProjection(make([]byte, 3*bf16Size))
	if err == nil {
		t.Fatal("DraftOutputProjection error = nil, want assistant hidden length error")
	}
	if !core.Contains(err.Error(), "assistant hidden") {
		t.Fatalf("DraftOutputProjection error = %v, want assistant hidden", err)
	}
}

func TestGemma4AssistantDraftLogitsDenseMatchesReference(t *testing.T) {
	requireNativeRuntime(t)

	tensors := nativeAssistantTinyTensors(false)
	embedW := nativeAssistantProjectionFixture(8, 4)
	tensors["model.embed_tokens.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{8, 4}, Data: toBF16Bytes(embedW)}
	dir := writeNativeAssistantDirWithOrdered(t, tensors, false)

	assistant, err := LoadGemma4AssistantDir(dir)
	if err != nil {
		t.Fatalf("LoadGemma4AssistantDir: %v", err)
	}
	defer assistant.Close()

	hidden := toBF16Bytes([]float32{1, -0.5, 0.25, 2})
	got, err := assistant.DraftLogits(hidden)
	if err != nil {
		t.Fatalf("DraftLogits dense: %v", err)
	}

	want := nativeAssistantMatMulBF16NTReference(hidden, toBF16Bytes(embedW), 1, 4, 8)
	assertFloat32Near(t, "dense draft logits", bf16Floats(got), want, 0.02)
}

func TestGemma4AssistantDraftLogitsOrderedMasksNonCandidates(t *testing.T) {
	tensors := nativeAssistantTinyTensors(true)
	embedW := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
		-1, 0, 0, 0,
		0, -1, 0, 0,
		0, 0, -1, 0,
		0, 0, 0, -1,
	}
	centroids := []float32{
		1, 0, 0, 0,
		-1, 0, 0, 0,
	}
	tensors["model.embed_tokens.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{8, 4}, Data: toBF16Bytes(embedW)}
	tensors["masked_embedding.centroids.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{2, 4}, Data: toBF16Bytes(centroids)}
	tensors["masked_embedding.token_ordering"] = safetensors.Tensor{Dtype: "I64", Shape: []int{2, 4}, Data: nativeAssistantI64Tensor(0, 1, 2, 3, 4, 5, 6, 7)}
	dir := writeNativeAssistantDir(t, tensors)

	assistant, err := LoadGemma4AssistantDir(dir)
	if err != nil {
		t.Fatalf("LoadGemma4AssistantDir: %v", err)
	}
	defer assistant.Close()

	hidden := toBF16Bytes([]float32{1, 0.5, -0.25, 2})
	got, err := assistant.DraftLogits(hidden)
	if err != nil {
		t.Fatalf("DraftLogits ordered: %v", err)
	}

	floor := nativeAssistantBF16Float(nativeAssistantLogitsFloorForTest)
	want := []float32{1, 0.5, -0.25, 2, floor, floor, floor, floor}
	assertFloat32Near(t, "ordered draft logits", bf16Floats(got), want, 0.02)
}

func TestGemma4AssistantDraftGreedyTokenSelectsArgmax(t *testing.T) {
	tensors := nativeAssistantTinyTensors(false)
	dir := writeNativeAssistantDirWithOrdered(t, tensors, false)

	assistant, err := LoadGemma4AssistantDir(dir)
	if err != nil {
		t.Fatalf("LoadGemma4AssistantDir: %v", err)
	}
	defer assistant.Close()

	got, err := assistant.DraftGreedyToken(toBF16Bytes([]float32{-1, 0.5, 3, 2.75, -0.25, 1, 0, 2}))
	if err != nil {
		t.Fatalf("DraftGreedyToken: %v", err)
	}
	if got != 2 {
		t.Fatalf("DraftGreedyToken = %d, want 2", got)
	}
}

func TestGemma4AssistantDraftGreedyTokenSuppressesIDs(t *testing.T) {
	tensors := nativeAssistantTinyTensors(false)
	dir := writeNativeAssistantDirWithOrdered(t, tensors, false)

	assistant, err := LoadGemma4AssistantDir(dir)
	if err != nil {
		t.Fatalf("LoadGemma4AssistantDir: %v", err)
	}
	defer assistant.Close()

	got, err := assistant.DraftGreedyToken(toBF16Bytes([]float32{-1, 0.5, 3, 2.75, -0.25, 1, 0, 2}), []int32{2, -1})
	if err != nil {
		t.Fatalf("DraftGreedyToken suppressed: %v", err)
	}
	if got != 3 {
		t.Fatalf("DraftGreedyToken suppressed = %d, want 3", got)
	}
}

func nativeAssistantProjectionFixture(out, in int) []float32 {
	weights := make([]float32, out*in)
	palette := []float32{-0.5, -0.25, 0, 0.25, 0.5}
	for o := 0; o < out; o++ {
		for k := 0; k < in; k++ {
			weights[o*in+k] = palette[(o*3+k*2)%len(palette)]
		}
	}
	return weights
}

func nativeAssistantMatMulBF16NTReference(a, w []byte, m, k, n int) []float32 {
	af, wf := bf16Floats(a), bf16Floats(w)
	out := make([]float32, m*n)
	for row := 0; row < m; row++ {
		for col := 0; col < n; col++ {
			var sum float32
			for inner := 0; inner < k; inner++ {
				sum += af[row*k+inner] * wf[col*k+inner]
			}
			h := f32ToBF16(sum)
			out[row*n+col] = bf16ToF32(byte(h), byte(h>>8))
		}
	}
	return out
}

const nativeAssistantLogitsFloorForTest = -3.4028234663852886e38

func nativeAssistantBF16Float(v float32) float32 {
	h := f32ToBF16(v)
	return bf16ToF32(byte(h), byte(h>>8))
}

func nativeAssistantI64Tensor(values ...int64) []byte {
	out := make([]byte, len(values)*8)
	for i, v := range values {
		binary.LittleEndian.PutUint64(out[i*8:], uint64(v))
	}
	return out
}

func nativeAssistantWrongToken(want int32) int32 {
	return (want + 1) % int32(mtpFixtureVocab)
}

func nativeAssistantQuantEmbeddingFixture(vocab, dModel, groupSize int) ([]byte, []byte, []byte) {
	packed := make([]byte, vocab*dModel/2)
	for row := 0; row < vocab; row++ {
		for col := 0; col < dModel; col += 2 {
			lo := byte((row + col) & 0x0F)
			hi := byte((row + col + 1) & 0x0F)
			packed[row*dModel/2+col/2] = lo | hi<<4
		}
	}
	groups := dModel / groupSize
	scales := make([]float32, vocab*groups)
	biases := make([]float32, vocab*groups)
	for i := range scales {
		scales[i] = 0.25
		biases[i] = -1
	}
	return packed, toBF16Bytes(scales), toBF16Bytes(biases)
}

func nativeAssistantTinyLoaded(t *testing.T, ordered bool) *Gemma4AssistantModel {
	t.Helper()
	tensors := nativeAssistantTinyTensors(ordered)
	dir := writeNativeAssistantDirWithOrdered(t, tensors, ordered)
	assistant, err := LoadGemma4AssistantDir(dir)
	if err != nil {
		t.Fatalf("LoadGemma4AssistantDir: %v", err)
	}
	return assistant
}

func nativeAssistantTargetKVFixture(seed byte) Gemma4AssistantTargetKV {
	return Gemma4AssistantTargetKV{
		Key:    []byte{seed, seed + 1, seed + 2, seed + 3},
		Value:  []byte{seed + 4, seed + 5, seed + 6, seed + 7},
		Offset: 1,
		Length: 2,
	}
}

func writeNativeAssistantDir(t *testing.T, tensors map[string]safetensors.Tensor) string {
	return writeNativeAssistantDirWithOrdered(t, tensors, true)
}

func writeNativeAssistantDirWithOrdered(t *testing.T, tensors map[string]safetensors.Tensor, ordered bool) string {
	return writeNativeAssistantDirWithModelType(t, tensors, ordered, "gemma4_assistant")
}

func writeNativeAssistantDirWithModelType(t *testing.T, tensors map[string]safetensors.Tensor, ordered bool, modelType string) string {
	t.Helper()
	dir := t.TempDir()
	cfg := []byte(core.Sprintf(`{
		"model_type": %q,
		"backbone_hidden_size": 8,
		"num_centroids": 2,
		"centroid_intermediate_top_k": 1,
		"use_ordered_embeddings": %v,
		"text_config": {
			"model_type": "gemma4_assistant",
			"hidden_size": 4,
			"num_hidden_layers": 2,
			"intermediate_size": 8,
			"num_attention_heads": 2,
			"num_key_value_heads": 2,
			"head_dim": 2,
			"vocab_size": 8,
			"rms_norm_eps": 0.000001,
			"max_position_embeddings": 16,
			"layer_types": ["sliding_attention", "full_attention"],
			"rope_parameters": {
				"sliding_attention": {"rope_theta": 10000, "partial_rotary_factor": 1.0},
				"full_attention": {"rope_theta": 1000000, "partial_rotary_factor": 1.0}
			}
		}
	}`, modelType, ordered))
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), string(cfg)); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeNativeAssistantTokenizer(t, dir)
	blob, err := safetensors.Encode(tensors)
	if err != nil {
		t.Fatalf("Encode assistant tensors: %v", err)
	}
	if err := coreio.Local.Write(core.PathJoin(dir, "model.safetensors"), string(blob)); err != nil {
		t.Fatalf("write model.safetensors: %v", err)
	}
	return dir
}

func writeNativeAssistantFlatDir(t *testing.T, tensors map[string]safetensors.Tensor, ordered bool) string {
	t.Helper()
	dir := t.TempDir()
	cfg := []byte(core.Sprintf(`{
		"model_type": "gemma4_assistant",
		"backbone_hidden_size": 8,
		"num_centroids": 2,
		"centroid_intermediate_top_k": 1,
		"use_ordered_embeddings": %v,
		"hidden_size": 4,
		"num_hidden_layers": 2,
		"intermediate_size": 8,
		"num_attention_heads": 2,
		"num_key_value_heads": 2,
		"head_dim": 2,
		"vocab_size": 8,
		"rms_norm_eps": 0.000001,
		"max_position_embeddings": 16,
		"layer_types": ["sliding_attention", "full_attention"],
		"rope_parameters": {
			"sliding_attention": {"rope_theta": 10000, "partial_rotary_factor": 1.0},
			"full_attention": {"rope_theta": 1000000, "partial_rotary_factor": 1.0}
		}
	}`, ordered))
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), string(cfg)); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeNativeAssistantTokenizer(t, dir)
	blob, err := safetensors.Encode(tensors)
	if err != nil {
		t.Fatalf("Encode assistant tensors: %v", err)
	}
	if err := coreio.Local.Write(core.PathJoin(dir, "model.safetensors"), string(blob)); err != nil {
		t.Fatalf("write model.safetensors: %v", err)
	}
	return dir
}

func writeNativeAssistantTokenizer(t testing.TB, dir string) {
	t.Helper()
	const body = `{
  "model": {
    "type": "BPE",
    "vocab": {"h": 1, "e": 2, "l": 3, "o": 4},
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<bos>", "special": true},
    {"id": 5, "content": "<eos>", "special": true}
  ]
}`
	if err := coreio.Local.Write(core.PathJoin(dir, "tokenizer.json"), body); err != nil {
		t.Fatalf("write tokenizer.json: %v", err)
	}
}

func writeNativeAssistantTargetDir(t *testing.T, hidden int, layerTypes []string) string {
	t.Helper()
	dir := t.TempDir()
	layerTypesJSON := core.JSONMarshal(layerTypes)
	if !layerTypesJSON.OK {
		t.Fatalf("marshal layer types: %s", layerTypesJSON.Error())
	}
	cfg := []byte(core.Sprintf(`{
		"model_type": "gemma4_text",
		"hidden_size": %d,
		"num_hidden_layers": %d,
		"intermediate_size": 16,
		"num_attention_heads": 2,
		"num_key_value_heads": 2,
		"head_dim": 2,
		"vocab_size": 8,
		"rms_norm_eps": 0.000001,
		"sliding_window": 16,
		"max_position_embeddings": 16,
		"layer_types": %s,
		"rope_parameters": {
			"sliding_attention": {"rope_theta": 10000, "partial_rotary_factor": 1.0},
			"full_attention": {"rope_theta": 1000000, "partial_rotary_factor": 1.0}
		}
	}`, hidden, len(layerTypes), string(layerTypesJSON.Value.([]byte))))
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), string(cfg)); err != nil {
		t.Fatalf("write target config.json: %v", err)
	}
	return dir
}

func nativeAssistantTinyTensors(includeOrdered bool) map[string]safetensors.Tensor {
	tensors := map[string]safetensors.Tensor{}
	bf := func(name string, shape ...int) {
		elems := 1
		for _, dim := range shape {
			elems *= dim
		}
		tensors[name] = safetensors.Tensor{Dtype: "BF16", Shape: shape, Data: make([]byte, elems*2)}
	}
	bf("model.embed_tokens.weight", 8, 4)
	bf("model.norm.weight", 4)
	bf("pre_projection.weight", 4, 16)
	bf("post_projection.weight", 8, 4)
	if includeOrdered {
		bf("masked_embedding.centroids.weight", 2, 4)
		tensors["masked_embedding.token_ordering"] = safetensors.Tensor{Dtype: "I64", Shape: []int{8}, Data: make([]byte, 8*8)}
	}
	for i := 0; i < 2; i++ {
		p := core.Sprintf("model.layers.%d", i)
		bf(p+".input_layernorm.weight", 4)
		bf(p+".post_attention_layernorm.weight", 4)
		bf(p+".pre_feedforward_layernorm.weight", 4)
		bf(p+".post_feedforward_layernorm.weight", 4)
		bf(p+".layer_scalar", 4)
		bf(p+".self_attn.q_proj.weight", 4, 4)
		bf(p+".self_attn.o_proj.weight", 4, 4)
		bf(p+".self_attn.q_norm.weight", 2)
		bf(p+".mlp.gate_proj.weight", 8, 4)
		bf(p+".mlp.up_proj.weight", 8, 4)
		bf(p+".mlp.down_proj.weight", 4, 8)
	}
	return tensors
}

func nativeAssistantAttentionTensors() map[string]safetensors.Tensor {
	return nativeAssistantAttentionTensorsForBackbone(8)
}

func nativeAssistantAttentionTensorsForBackbone(backbone int) map[string]safetensors.Tensor {
	const hidden, headDim, nHeads, intermediate, vocab = 128, 64, 2, 256, 8
	tensors := map[string]safetensors.Tensor{}
	bf := func(name string, shape ...int) {
		elems := 1
		for _, dim := range shape {
			elems *= dim
		}
		tensors[name] = safetensors.Tensor{Dtype: "BF16", Shape: shape, Data: make([]byte, elems*bf16Size)}
	}
	bf("model.embed_tokens.weight", vocab, hidden)
	bf("model.norm.weight", hidden)
	bf("pre_projection.weight", hidden, backbone*2)
	bf("post_projection.weight", backbone, hidden)
	p := "model.layers.0"
	bf(p+".input_layernorm.weight", hidden)
	bf(p+".post_attention_layernorm.weight", hidden)
	bf(p+".pre_feedforward_layernorm.weight", hidden)
	bf(p+".post_feedforward_layernorm.weight", hidden)
	bf(p+".layer_scalar", hidden)
	bf(p+".self_attn.q_proj.weight", nHeads*headDim, hidden)
	bf(p+".self_attn.o_proj.weight", hidden, nHeads*headDim)
	bf(p+".self_attn.q_norm.weight", headDim)
	bf(p+".mlp.gate_proj.weight", intermediate, hidden)
	bf(p+".mlp.up_proj.weight", intermediate, hidden)
	bf(p+".mlp.down_proj.weight", hidden, intermediate)
	return tensors
}

func writeNativeAssistantAttentionDir(t testing.TB, tensors map[string]safetensors.Tensor) string {
	return writeNativeAssistantAttentionDirForBackbone(t, tensors, 8)
}

func writeNativeAssistantAttentionDirForBackbone(t testing.TB, tensors map[string]safetensors.Tensor, backbone int) string {
	t.Helper()
	dir := t.TempDir()
	cfg := []byte(core.Sprintf(`{
			"model_type": "gemma4_assistant",
			"backbone_hidden_size": %d,
			"num_centroids": 0,
			"centroid_intermediate_top_k": 0,
			"use_ordered_embeddings": false,
		"text_config": {
			"model_type": "gemma4_assistant",
			"hidden_size": 128,
			"num_hidden_layers": 1,
			"intermediate_size": 256,
			"num_attention_heads": 2,
			"num_key_value_heads": 2,
			"head_dim": 64,
			"vocab_size": 8,
			"rms_norm_eps": 0.000001,
			"max_position_embeddings": 16,
			"layer_types": ["sliding_attention"],
			"rope_parameters": {
				"sliding_attention": {"rope_theta": 10000, "partial_rotary_factor": 1.0}
				}
			}
		}`, backbone))
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), string(cfg)); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeNativeAssistantTokenizer(t, dir)
	blob, err := safetensors.Encode(tensors)
	if err != nil {
		t.Fatalf("Encode assistant tensors: %v", err)
	}
	if err := coreio.Local.Write(core.PathJoin(dir, "model.safetensors"), string(blob)); err != nil {
		t.Fatalf("write model.safetensors: %v", err)
	}
	return dir
}

func writeNativeAssistantAttentionTargetDir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	cfg := []byte(`{
		"model_type": "gemma4_text",
		"hidden_size": 8,
		"num_hidden_layers": 1,
		"intermediate_size": 256,
		"num_attention_heads": 2,
		"num_key_value_heads": 2,
		"head_dim": 64,
		"vocab_size": 8,
		"rms_norm_eps": 0.000001,
		"sliding_window": 16,
		"max_position_embeddings": 16,
		"layer_types": ["sliding_attention"],
		"rope_parameters": {
			"sliding_attention": {"rope_theta": 10000, "partial_rotary_factor": 1.0}
		}
	}`)
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), string(cfg)); err != nil {
		t.Fatalf("write target config.json: %v", err)
	}
	return dir
}

const nativeTestGGUFTensorTypeBF16 = 30

type nativeTestGGUFTensor struct {
	Name string
	Type uint32
	Dims []uint64
	Data []byte
}

func writeNativeAssistantGGUF(t *testing.T, tensors map[string]safetensors.Tensor) string {
	t.Helper()
	path := core.PathJoin(t.TempDir(), "mtp-tiny.gguf")
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		if nativeAssistantGGUFNameForTest(t, name) != "" {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	payloads := make([]nativeTestGGUFTensor, 0, len(names))
	for _, name := range names {
		tensor := tensors[name]
		dims := make([]uint64, len(tensor.Shape))
		for i, dim := range tensor.Shape {
			dims[i] = uint64(dim)
		}
		payloads = append(payloads, nativeTestGGUFTensor{
			Name: nativeAssistantGGUFNameForTest(t, name),
			Type: nativeTestGGUFTensorTypeBF16,
			Dims: dims,
			Data: tensor.Data,
		})
	}
	writeNativeTestGGUF(t, path, nativeAssistantGGUFMetadata(), payloads)
	return path
}

func nativeAssistantGGUFMetadata() []nativeTestGGUFMeta {
	const p = nativeGemma4AssistantGGUFArch + "."
	return []nativeTestGGUFMeta{
		{Key: "general.architecture", ValueType: gguf.ValueTypeString, Value: nativeGemma4AssistantGGUFArch},
		{Key: "general.alignment", ValueType: gguf.ValueTypeUint32, Value: uint32(32)},
		{Key: p + "block_count", ValueType: gguf.ValueTypeUint32, Value: uint32(2)},
		{Key: p + "embedding_length", ValueType: gguf.ValueTypeUint32, Value: uint32(4)},
		{Key: p + "embedding_length_out", ValueType: gguf.ValueTypeUint32, Value: uint32(8)},
		{Key: p + "attention.head_count", ValueType: gguf.ValueTypeUint32, Value: uint32(2)},
		{Key: p + "attention.head_count_kv", ValueType: gguf.ValueTypeUint32, Value: uint32(2)},
		{Key: p + "attention.key_length", ValueType: gguf.ValueTypeUint32, Value: uint32(2)},
		{Key: p + "attention.sliding_window_pattern", ValueType: gguf.ValueTypeUint32, Value: uint32(2)},
		{Key: p + "attention.sliding_window", ValueType: gguf.ValueTypeUint32, Value: uint32(16)},
		{Key: p + "attention.shared_kv_layers", ValueType: gguf.ValueTypeUint32, Value: uint32(0)},
		{Key: p + "feed_forward_length", ValueType: gguf.ValueTypeUint32, Value: uint32(8)},
		{Key: p + "context_length", ValueType: gguf.ValueTypeUint32, Value: uint32(16)},
	}
}

func nativeAssistantGGUFNameForTest(t *testing.T, hf string) string {
	t.Helper()
	base := []string{
		"token_embd.weight",
		"output_norm.weight",
		"nextn.pre_projection.weight",
		"nextn.post_projection.weight",
	}
	for _, name := range base {
		if nativeGemma4AssistantGGUFWeightName(name) == hf {
			return name
		}
	}
	leaves := []string{
		"attn_norm.weight",
		"post_attention_norm.weight",
		"ffn_norm.weight",
		"post_ffw_norm.weight",
		"attn_q.weight",
		"attn_q_norm.weight",
		"attn_output.weight",
		"ffn_gate.weight",
		"ffn_up.weight",
		"ffn_down.weight",
		"layer_output_scale.weight",
	}
	for layer := 0; layer < 4; layer++ {
		for _, leaf := range leaves {
			name := core.Sprintf("blk.%d.%s", layer, leaf)
			mapped := nativeGemma4AssistantGGUFWeightName(name)
			if mapped == hf || (leaf == "layer_output_scale.weight" && mapped == hf+".weight") {
				return name
			}
		}
	}
	return ""
}

type nativeTestGGUFMeta struct {
	Key       string
	ValueType uint32
	Value     any
}

func writeNativeTestGGUF(t *testing.T, path string, metadata []nativeTestGGUFMeta, tensors []nativeTestGGUFTensor) {
	t.Helper()
	created := core.Create(path)
	if !created.OK {
		t.Fatalf("create gguf: %v", created.Value)
	}
	file := created.Value.(*core.OSFile)
	defer file.Close()
	writeNativeTestGGUFScalar(t, file, uint32(0x46554747))
	writeNativeTestGGUFScalar(t, file, uint32(3))
	writeNativeTestGGUFScalar(t, file, uint64(len(tensors)))
	writeNativeTestGGUFScalar(t, file, uint64(len(metadata)))
	for _, entry := range metadata {
		writeNativeTestGGUFString(t, file, entry.Key)
		writeNativeTestGGUFScalar(t, file, entry.ValueType)
		writeNativeTestGGUFValue(t, file, entry)
	}
	var offset uint64
	offsets := make([]uint64, len(tensors))
	for i, tensor := range tensors {
		offset += nativeTestGGUFAlignPadding(offset, 32)
		offsets[i] = offset
		offset += uint64(len(tensor.Data))
	}
	for i, tensor := range tensors {
		writeNativeTestGGUFString(t, file, tensor.Name)
		writeNativeTestGGUFScalar(t, file, uint32(len(tensor.Dims)))
		for _, dim := range tensor.Dims {
			writeNativeTestGGUFScalar(t, file, dim)
		}
		writeNativeTestGGUFScalar(t, file, tensor.Type)
		writeNativeTestGGUFScalar(t, file, offsets[i])
	}
	position, err := file.Seek(0, 1)
	if err != nil {
		t.Fatalf("seek gguf: %v", err)
	}
	writeNativeTestGGUFPadding(t, file, nativeTestGGUFAlignPadding(uint64(position), 32))
	var written uint64
	for i, tensor := range tensors {
		writeNativeTestGGUFPadding(t, file, offsets[i]-written)
		if _, err := file.Write(tensor.Data); err != nil {
			t.Fatalf("write gguf tensor: %v", err)
		}
		written = offsets[i] + uint64(len(tensor.Data))
	}
}

func writeNativeTestGGUFValue(t *testing.T, file *core.OSFile, entry nativeTestGGUFMeta) {
	t.Helper()
	switch entry.ValueType {
	case gguf.ValueTypeString:
		value, ok := entry.Value.(string)
		if !ok {
			t.Fatalf("metadata %s = %T, want string", entry.Key, entry.Value)
		}
		writeNativeTestGGUFString(t, file, value)
	case gguf.ValueTypeUint32:
		value, ok := entry.Value.(uint32)
		if !ok {
			t.Fatalf("metadata %s = %T, want uint32", entry.Key, entry.Value)
		}
		writeNativeTestGGUFScalar(t, file, value)
	default:
		t.Fatalf("unsupported native test gguf metadata type %d", entry.ValueType)
	}
}

func writeNativeTestGGUFString(t *testing.T, file *core.OSFile, value string) {
	t.Helper()
	writeNativeTestGGUFScalar(t, file, uint64(len(value)))
	if _, err := file.Write([]byte(value)); err != nil {
		t.Fatalf("write gguf string: %v", err)
	}
}

func writeNativeTestGGUFScalar(t *testing.T, file *core.OSFile, value any) {
	t.Helper()
	if err := binary.Write(file, binary.LittleEndian, value); err != nil {
		t.Fatalf("write gguf scalar: %v", err)
	}
}

func writeNativeTestGGUFPadding(t *testing.T, file *core.OSFile, n uint64) {
	t.Helper()
	if n == 0 {
		return
	}
	padding := make([]byte, int(n))
	if _, err := file.Write(padding); err != nil {
		t.Fatalf("write gguf padding: %v", err)
	}
}

func nativeTestGGUFAlignPadding(offset, alignment uint64) uint64 {
	if alignment == 0 {
		return 0
	}
	return (alignment - (offset % alignment)) % alignment
}

func newNativeAssistantGenerateFixture(t testing.TB) (*Gemma4AssistantPair, func() *ArchSession) {
	t.Helper()
	const hidden, heads, kvHeads, headDim, ff, vocab = 128, 2, 2, 64, 256, 8
	layers := []DecodeLayerWeights{forwardLayer(hidden, heads, kvHeads, headDim, ff, 701)}
	embed := toBF16Bytes(syntheticFloat32(vocab*hidden, 703))
	g := &BF16Model{
		Layers:    layers,
		Embed:     embed,
		FinalNorm: toBF16Bytes(syntheticFloat32(hidden, 707)),
		LMHead:    embed,
		Tied:      true,
	}
	arch := model.Arch{
		Hidden: hidden, Heads: heads, KVHeads: kvHeads, HeadDim: headDim, FF: ff, Vocab: vocab,
		GlobalHeadDim: headDim, GlobalKVHeads: kvHeads,
		Eps: 1e-5, AttnScale: 0.125, RopeBase: 10000, RopeScale: 1, RopeLocalBase: 10000,
		RotaryDim: headDim, RotaryDimLocal: headDim, SlidingWindow: 16,
		Layer: model.DeriveLayers([]string{"sliding_attention"}, 0),
	}
	assistantDir := writeNativeAssistantAttentionDirForBackbone(t, nativeAssistantAttentionTensorsForBackbone(hidden), hidden)
	assistant, err := LoadGemma4AssistantDir(assistantDir)
	if err != nil {
		t.Fatalf("LoadGemma4AssistantDir: %v", err)
	}
	pair := &Gemma4AssistantPair{TargetArch: arch, Assistant: assistant}
	if err := validateNativeGemma4AssistantPair(pair); err != nil {
		_ = pair.Close()
		t.Fatalf("validateNativeGemma4AssistantPair: %v", err)
	}
	mk := func() *ArchSession {
		s, err := NewArchSession(g, arch, 64)
		if err != nil {
			t.Fatalf("NewArchSession: %v", err)
		}
		head := &headEncoder{
			finalNorm: copyView(g.FinalNorm),
			weight:    copyView(g.LMHead),
			dModel:    arch.Hidden,
			vocab:     arch.Vocab,
			eps:       arch.Eps,
			softCap:   arch.SoftCap,
		}
		s.headEnc = head
		s.head = func(hidden []byte, skipSoftcap bool) ([]byte, error) {
			return head.encode(hidden, skipSoftcap)
		}
		s.greedy = func(hidden []byte, suppress []int32) (int32, bool, error) {
			return head.greedyInPool(hidden, suppress)
		}
		s.markDefaultHeadFunc()
		s.markDefaultGreedyFunc()
		return s
	}
	return pair, mk
}

func nativeAssistantPromptWhoseFirstTargetTokenIsNot(t testing.TB, mk func() *ArchSession, excluded int32) []int32 {
	t.Helper()
	candidates := [][]int32{
		{1, 5, 3},
		{2, 4, 6},
		{3, 1, 7},
		{4, 2, 5},
		{5, 3, 1},
		{6, 7, 2},
	}
	for _, prompt := range candidates {
		got, err := mk().Generate(prompt, 1, -1)
		if err != nil {
			t.Fatalf("reference Generate(%v): %v", prompt, err)
		}
		if len(got) == 1 && got[0] != excluded {
			return prompt
		}
	}
	t.Fatalf("no prompt produced a first target token outside %d", excluded)
	return nil
}

func nativeAssistantSampledVerifierRejectFixture(t testing.TB, mk func() *ArchSession, params model.SampleParams) ([]int32, uint64, int32, int32) {
	t.Helper()
	candidates := [][]int32{
		{1, 5, 3},
		{2, 4, 6},
		{3, 1, 7},
		{4, 2, 5},
		{5, 3, 1},
		{6, 7, 2},
	}
	for _, prompt := range candidates {
		greedy, err := mk().Generate(prompt, 1, -1)
		if err != nil {
			t.Fatalf("reference Generate(%v): %v", prompt, err)
		}
		for seed := uint64(1); seed <= 64; seed++ {
			for draft := int32(0); draft < 8; draft++ {
				probe := mk()
				if err := probe.PrefillTokens(prompt); err != nil {
					t.Fatalf("PrefillTokens(%v): %v", prompt, err)
				}
				logits, err := probe.BoundaryLogits()
				if err != nil {
					t.Fatalf("BoundaryLogits(%v): %v", prompt, err)
				}
				sampled, err := model.NewSampler(seed).Sample(logits, probe.arch.Vocab, params)
				if err != nil {
					t.Fatalf("sample verifier logits(%v, seed %d, draft %d): %v", prompt, seed, draft, err)
				}
				if len(greedy) == 1 && sampled != greedy[0] && sampled != draft {
					return prompt, seed, sampled, draft
				}
			}
		}
	}
	t.Fatal("no sampled verifier fixture produced a reject token different from greedy")
	return nil, 0, 0, 0
}

func nativeAssistantSessionTargetArchForTest() model.Arch {
	return model.Arch{
		Hidden: 8, Vocab: 8, Heads: 2, KVHeads: 2, HeadDim: 2, SlidingWindow: 4,
		Layer: []model.LayerSpec{
			{Attention: model.SlidingAttention, KVShareFrom: 0, CacheIndex: 0, HeadDim: 2, KVHeads: 2},
			{Attention: model.GlobalAttention, KVShareFrom: 1, CacheIndex: 1, HeadDim: 2, KVHeads: 2},
			{Attention: model.SlidingAttention, KVShareFrom: 0, CacheIndex: -1, HeadDim: 2, KVHeads: 2},
			{Attention: model.GlobalAttention, KVShareFrom: 1, CacheIndex: -1, HeadDim: 2, KVHeads: 2},
		},
	}
}

func nativeAssistantSessionRowsForTest(rows, rowBytes int, seed byte) []byte {
	out := make([]byte, rows*rowBytes)
	for row := 0; row < rows; row++ {
		for col := 0; col < rowBytes; col++ {
			out[row*rowBytes+col] = seed + byte(row+col)
		}
	}
	return out
}

func nativeAssistantSessionKVRowsForTest(tokens, kvHeads, headDim int, seed byte) []byte {
	rowBytes := kvHeads * headDim * bf16Size
	out := make([]byte, tokens*rowBytes)
	for token := 0; token < tokens; token++ {
		for head := 0; head < kvHeads; head++ {
			out[token*rowBytes+head*headDim*bf16Size] = seed + byte(token*0x10+head)
		}
	}
	return out
}
