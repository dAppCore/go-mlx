// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"strings"
	"testing"

	core "dappco.re/go"
)

// TestHasASCIIInsensitiveSuffix_Arms_Good covers the guard + case-folding
// arms of hasASCIIInsensitiveSuffix: a string shorter than the suffix (->
// false), an uppercase-extension match (the A-Z normalisation of both sides),
// and a clean mismatch.
func TestHasASCIIInsensitiveSuffix_Arms_Good(t *testing.T) {
	if hasASCIIInsensitiveSuffix("x", ".gguf") {
		t.Fatal("hasASCIIInsensitiveSuffix(too short) = true, want false")
	}
	if !hasASCIIInsensitiveSuffix("MODEL.GGUF", ".gguf") {
		t.Fatal("hasASCIIInsensitiveSuffix(upper vs lower) = false, want true")
	}
	if !hasASCIIInsensitiveSuffix("model.GgUf", ".GGUF") {
		t.Fatal("hasASCIIInsensitiveSuffix(mixed case both sides) = false, want true")
	}
	if hasASCIIInsensitiveSuffix("model.bin", ".gguf") {
		t.Fatal("hasASCIIInsensitiveSuffix(mismatch) = true, want false")
	}
}

// TestReadModelConfig_BadJSON_Bad covers readModelConfig's unmarshal-failure
// arm: a config.json whose bytes are not valid JSON surfaces the unmarshal
// error.
func TestReadModelConfig_BadJSON_Bad(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), "{not valid json")
	if _, err := readModelConfig(dir); err == nil {
		t.Fatal("readModelConfig(bad json) error = nil")
	}
	// Missing config.json -> read-failure arm.
	if _, err := readModelConfig(t.TempDir()); err == nil {
		t.Fatal("readModelConfig(missing) error = nil")
	}
}

// TestModelConfigProbe_Architecture_Arms_Good drives the architecture()
// resolution arms: the bert_rerank fast path (an ForSequenceClassification
// transformers class), the generic transformers-name fallback, and the
// empty-probe -> "" tail.
func TestModelConfigProbe_Architecture_Arms_Good(t *testing.T) {
	// bert_rerank fast path: Architectures carries a sequence-classification
	// class, which ArchitectureFromTransformersName maps to "bert_rerank".
	rerank := &modelConfigProbe{Architectures: []string{"BertForSequenceClassification"}}
	if got := rerank.architecture(); got != "bert_rerank" {
		t.Fatalf("architecture(bert rerank) = %q, want bert_rerank", got)
	}

	// Empty probe: no model_type, no text_config.model_type, no recognisable
	// architecture -> "".
	empty := &modelConfigProbe{Architectures: []string{"TotallyUnknownArch"}}
	if got := empty.architecture(); got != "" {
		t.Fatalf("architecture(unknown) = %q, want empty", got)
	}

	// nil receiver -> "".
	var nilProbe *modelConfigProbe
	if got := nilProbe.architecture(); got != "" {
		t.Fatalf("architecture(nil) = %q, want empty", got)
	}
}

// TestMetadataIntForSuffix_Arms_Good drives metadataIntForSuffix's tail arms:
// a hit on the bare suffix (no prefix), and the over-long-key fallback where
// "<prefix>.<suffix>" exceeds the 128-byte stack scratch and the concat-
// allowed path is taken.
func TestMetadataIntForSuffix_Arms_Good(t *testing.T) {
	// Bare-suffix hit: neither "<arch>.n_embd" nor "general.n_embd" exist, but
	// a bare "n_embd" does, so the final loop returns it.
	metadata := map[string]any{"n_embd": uint32(2048)}
	if got := metadataIntForSuffix(metadata, "qwen3", "embedding_length", "n_embd"); got != 2048 {
		t.Fatalf("metadataIntForSuffix(bare suffix) = %d, want 2048", got)
	}

	// Over-long key: a >128-char architecture makes "<arch>.<suffix>" exceed
	// the scratch buffer, forcing the alloc-allowed concat fallback. The key
	// is present so that fallback returns it.
	longArch := strings.Repeat("a", 130)
	longMeta := map[string]any{longArch + ".block_count": uint32(48)}
	if got := metadataIntForSuffix(longMeta, longArch, "block_count"); got != 48 {
		t.Fatalf("metadataIntForSuffix(long key) = %d, want 48", got)
	}
}

// TestInferLayerCount_Arms_Good drives inferLayerCount's metadata hit and the
// tensor-name scan fallback (no block_count metadata -> derive from the max
// layer index in tensor names). The over-long-architecture concat fallback is
// exercised separately by TestInferLayerCount_OverLongArchitecture_Good.
func TestInferLayerCount_Arms_Good(t *testing.T) {
	// Metadata hit: "<arch>.block_count" present.
	if got := inferLayerCount(map[string]any{"qwen3.block_count": uint32(28)}, nil, "qwen3"); got != 28 {
		t.Fatalf("inferLayerCount(metadata) = %d, want 28", got)
	}

	// No layer-count metadata: fall back to scanning tensor names; the highest
	// layer index + 1 is the count.
	tensors := []ggufTensorInfo{
		{Name: "model.layers.0.attn.weight"},
		{Name: "model.layers.5.attn.weight"},
		{Name: "model.norm.weight"},
	}
	if got := inferLayerCount(map[string]any{}, tensors, "qwen3"); got != 6 {
		t.Fatalf("inferLayerCount(tensor scan) = %d, want 6", got)
	}

	// No metadata, no layer-indexed tensor names -> 0.
	if got := inferLayerCount(map[string]any{}, []ggufTensorInfo{{Name: "token_embd.weight"}}, "qwen3"); got != 0 {
		t.Fatalf("inferLayerCount(no layers) = %d, want 0", got)
	}
}

// TestMetadataIntForSuffix_OverLongKeyAbsent_Good drives the `continue`
// after the over-long-key fallback: a >128-char architecture forces the
// concat-allowed lookup, and because the key is ABSENT that lookup returns 0
// and the loop continues (rather than returning a value). The function then
// falls through to 0.
func TestMetadataIntForSuffix_OverLongKeyAbsent_Good(t *testing.T) {
	longArch := strings.Repeat("c", 130)
	// No "<longArch>.block_count" key present -> over-long lookup misses and
	// continues; final result is 0.
	if got := metadataIntForSuffix(map[string]any{"unrelated": uint32(5)}, longArch, "block_count"); got != 0 {
		t.Fatalf("metadataIntForSuffix(long key absent) = %d, want 0", got)
	}
}

// TestProbeDiscoveredModel_GGUFConfigParsesNoArch_Good drives
// probeDiscoveredModel's `modelType == "" && configErr == nil` assignment
// arm: a directory with a valid GGUF that has NO general.architecture AND a
// config.json that parses but yields no architecture either. The probe enters
// the fallback (configErr == nil) and assigns the still-empty config
// architecture, leaving ModelType "".
func TestProbeDiscoveredModel_GGUFConfigParsesNoArch_Good(t *testing.T) {
	dir := t.TempDir()
	// config.json parses cleanly but has no model_type / text_config /
	// architectures -> architecture() == "".
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{"hidden_size":64}`)
	writeTestGGUF(t, core.PathJoin(dir, "model.gguf"),
		[]ggufMetaSpec{{Key: "general.file_type", ValueType: ValueTypeUint32, Value: uint32(7)}},
		[]ggufTensorSpec{{Name: "blk.0.attn_q.weight", Type: TensorTypeQ8_0, Dims: []uint64{32, 2}}},
	)
	model, ok := probeDiscoveredModel(dir)
	if !ok {
		t.Fatal("probeDiscoveredModel(no-arch gguf + parseable config) ok = false, want true")
	}
	if model.ModelType != "" {
		t.Fatalf("probeDiscoveredModel ModelType = %q, want empty", model.ModelType)
	}
}

// TestReadInfo_QuantBitsReconcile_Good drives ReadInfo's
// `quantization.Bits == 0 -> quantization.Bits = quantBits` reconciliation: a
// GGUF with only an f32 tensor (not quantised), no file_type and no
// quantisation metadata, yields a zero-bit quantisation summary — but
// config.json declares a bit width, so the reconcile arm pulls it in.
func TestReadInfo_QuantBitsReconcile_Good(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{"model_type":"qwen3","hidden_size":64,"quantization":{"bits":4,"group_size":32}}`)
	// f32 tensor only: not quantised, so the quant-summary bits are 0. No
	// general.file_type, no quantization.* metadata.
	writeTestGGUF(t, core.PathJoin(dir, "model.gguf"),
		[]ggufMetaSpec{{Key: "general.architecture", ValueType: ValueTypeString, Value: "qwen3"}},
		[]ggufTensorSpec{{Name: "blk.0.attn_q.weight", Type: ggufTensorTypeF32, Dims: []uint64{64, 2}}},
	)
	info, err := ReadInfo(dir)
	if err != nil {
		t.Fatalf("ReadInfo(f32 + config bits) error = %v", err)
	}
	// config.json's bits=4 is reconciled into the quantisation summary.
	if info.Quantization.Bits != 4 {
		t.Fatalf("ReadInfo Quantization.Bits = %d, want 4 (reconciled from config)", info.Quantization.Bits)
	}
}

// TestExtractLayerIndex_MarkerNoDigit_Good covers extractLayerIndex's
// "marker present but no digit follows" arm: a name containing "blk." with a
// non-numeric remainder must skip that marker (end == start) and return -1
// when no marker yields a number.
func TestExtractLayerIndex_MarkerNoDigit_Good(t *testing.T) {
	if got := extractLayerIndex("blk.attn.weight"); got != -1 {
		t.Fatalf("extractLayerIndex(marker no digit) = %d, want -1", got)
	}
	// A real numeric marker still resolves.
	if got := extractLayerIndex("blk.7.attn.weight"); got != 7 {
		t.Fatalf("extractLayerIndex(numeric) = %d, want 7", got)
	}
	// No marker at all -> -1.
	if got := extractLayerIndex("token_embd.weight"); got != -1 {
		t.Fatalf("extractLayerIndex(no marker) = %d, want -1", got)
	}
}

// TestDiscoverModels_GGUFFileErrorPaths_Bad drives DiscoverModels' single-
// file arms: a path that ends in .gguf but is not a readable GGUF (ReadInfo
// fails) returns nil, and a non-gguf file returns nil.
func TestDiscoverModels_GGUFFileErrorPaths_Bad(t *testing.T) {
	dir := t.TempDir()

	// A .gguf file with garbage contents -> ReadInfo fails -> nil.
	bad := core.PathJoin(dir, "broken.gguf")
	if w := core.WriteFile(bad, []byte("not a gguf"), 0o644); !w.OK {
		t.Fatalf("write broken gguf: %v", w.Value)
	}
	if got := DiscoverModels(bad); got != nil {
		t.Fatalf("DiscoverModels(broken .gguf file) = %+v, want nil", got)
	}

	// A non-gguf regular file -> nil (suffix check fails).
	other := core.PathJoin(dir, "weights.bin")
	if w := core.WriteFile(other, []byte("xx"), 0o644); !w.OK {
		t.Fatalf("write bin: %v", w.Value)
	}
	if got := DiscoverModels(other); got != nil {
		t.Fatalf("DiscoverModels(non-gguf file) = %+v, want nil", got)
	}
}

// TestDiscoverModels_MissingPath_Bad drives the walk over a path that does
// not exist: PathWalkDir errors, so DiscoverModels returns nil.
func TestDiscoverModels_MissingPath_Bad(t *testing.T) {
	missing := core.PathJoin(t.TempDir(), "no", "such", "tree")
	if got := DiscoverModels(missing); got != nil {
		t.Fatalf("DiscoverModels(missing tree) = %+v, want nil", got)
	}
}

// TestProbeDiscoveredModel_GGUFEmptyArchFallback_Good drives
// probeDiscoveredModel's "gguf architecture empty -> fall back to config"
// arm: a directory holding a valid GGUF whose metadata carries no
// general.architecture, plus a config.json that supplies the model type.
func TestProbeDiscoveredModel_GGUFEmptyArchFallback_Good(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{"model_type":"qwen3"}`)
	// GGUF with NO general.architecture key -> info.Architecture == "".
	writeTestGGUF(t, core.PathJoin(dir, "model.gguf"),
		[]ggufMetaSpec{{Key: "general.file_type", ValueType: ValueTypeUint32, Value: uint32(7)}},
		[]ggufTensorSpec{{Name: "blk.0.attn_q.weight", Type: TensorTypeQ8_0, Dims: []uint64{32, 2}}},
	)
	model, ok := probeDiscoveredModel(dir)
	if !ok {
		t.Fatal("probeDiscoveredModel(empty-arch gguf) ok = false, want true")
	}
	if model.ModelType != "qwen3" {
		t.Fatalf("probeDiscoveredModel ModelType = %q, want qwen3 (from config fallback)", model.ModelType)
	}
}

// TestReadInfo_QuantBitsFallback_Good drives ReadInfo's
// `quantization.Bits == 0 -> quantBits` reconciliation: a GGUF whose tensors
// are all unquantised dense floats (f16) yields no inferred quant bits from
// the quantisation path, but config.json (or the f16 file_type) still pins a
// bit width that ReadInfo carries through.
func TestReadInfo_QuantBitsFallback_Good(t *testing.T) {
	dir := t.TempDir()
	// f16 weights: not "Quantized", so the quant-summary bits are 0 and the
	// reconciliation arm must source bits elsewhere.
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{"model_type":"qwen3","hidden_size":64}`)
	writeTestGGUF(t, core.PathJoin(dir, "model.gguf"),
		[]ggufMetaSpec{
			{Key: "general.architecture", ValueType: ValueTypeString, Value: "qwen3"},
			{Key: "general.file_type", ValueType: ValueTypeUint32, Value: uint32(1)}, // f16
		},
		[]ggufTensorSpec{{Name: "blk.0.attn_q.weight", Type: ggufTensorTypeF16, Dims: []uint64{64, 2}}},
	)
	info, err := ReadInfo(dir)
	if err != nil {
		t.Fatalf("ReadInfo(f16 model) error = %v", err)
	}
	if !info.Valid() {
		t.Fatalf("ReadInfo(f16) validation issues = %+v", info.ValidationIssues)
	}
	// f16 file_type pins 16 bits even though no tensor is "Quantized".
	if info.QuantBits != 16 {
		t.Fatalf("ReadInfo(f16) QuantBits = %d, want 16", info.QuantBits)
	}
}
