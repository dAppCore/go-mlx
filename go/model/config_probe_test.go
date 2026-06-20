// SPDX-Licence-Identifier: EUPL-1.2

// Branch tests for the modelConfigProbe accessor chain in config_probe.go.
//
// TestModelConfigProbe_AccessorsAfterWalker (config_probe_unmarshal_test.go)
// already covers the populated top-level path through the walker. The
// accessors carry two further branches the walker-driven fixtures never
// reach, so they live here against directly-constructed probes (no JSON, no
// file I/O — pure struct → scalar):
//
//   - Good:  top-level field set → returned verbatim.
//   - Bad:   nil receiver → defensive zero (the `if probe == nil` guard).
//   - Ugly:  top-level field absent/zero, only text_config carries the value
//            → the TextConfig fallback wins.
//
// Constructing the probe directly is deliberate: the accessor contract is a
// pure function of the struct fields, independent of how the struct was
// filled, so the fixture is the struct itself rather than a config.json body.

package model

import "testing"

// TestModelConfigProbe_Accessors_Good confirms every scalar accessor returns
// the top-level field when it is populated — the common, fully-specified
// config.json shape where nothing falls through to text_config.
func TestModelConfigProbe_Accessors_Good(t *testing.T) {
	probe := &modelConfigProbe{
		ModelType:             "qwen3",
		VocabSize:             151936,
		HiddenSize:            2048,
		NumHiddenLayers:       28,
		NumKeyValueHeads:      8,
		HeadDim:               128,
		MaxPositionEmbeddings: 40960,
	}
	if got := probe.numLayers(); got != 28 {
		t.Errorf("numLayers(): got %d want 28", got)
	}
	if got := probe.vocabSize(); got != 151936 {
		t.Errorf("vocabSize(): got %d want 151936", got)
	}
	if got := probe.hiddenSize(); got != 2048 {
		t.Errorf("hiddenSize(): got %d want 2048", got)
	}
	if got := probe.numKeyValueHeads(); got != 8 {
		t.Errorf("numKeyValueHeads(): got %d want 8", got)
	}
	if got := probe.headDim(); got != 128 {
		t.Errorf("headDim(): got %d want 128", got)
	}
	if got := probe.contextLength(); got != 40960 {
		t.Errorf("contextLength(): got %d want 40960", got)
	}
}

// TestModelConfigProbe_Accessors_Bad drives every accessor on a nil receiver.
// readModelConfigAt can return (nil, err) and a caller that ignores the error
// would dereference a nil probe; each accessor guards against that and reports
// a zero value rather than panicking. This is the `if probe == nil` branch the
// walker-built fixtures never exercise.
func TestModelConfigProbe_Accessors_Bad(t *testing.T) {
	var probe *modelConfigProbe // nil

	if got := probe.architecture(); got != "" {
		t.Errorf("architecture() on nil: got %q want empty", got)
	}
	if got := probe.numLayers(); got != 0 {
		t.Errorf("numLayers() on nil: got %d want 0", got)
	}
	if got := probe.vocabSize(); got != 0 {
		t.Errorf("vocabSize() on nil: got %d want 0", got)
	}
	if got := probe.hiddenSize(); got != 0 {
		t.Errorf("hiddenSize() on nil: got %d want 0", got)
	}
	if got := probe.numKeyValueHeads(); got != 0 {
		t.Errorf("numKeyValueHeads() on nil: got %d want 0", got)
	}
	if got := probe.headDim(); got != 0 {
		t.Errorf("headDim() on nil: got %d want 0", got)
	}
	if got := probe.contextLength(); got != 0 {
		t.Errorf("contextLength() on nil: got %d want 0", got)
	}
	if got := probe.quantBits(); got != 0 {
		t.Errorf("quantBits() on nil: got %d want 0", got)
	}
	if got := probe.quantGroup(); got != 0 {
		t.Errorf("quantGroup() on nil: got %d want 0", got)
	}
}

// TestModelConfigProbe_Accessors_Ugly leaves every top-level scalar zero and
// populates only text_config — the multimodal-wrapper shape where the text
// tower's dimensions live nested. Each accessor must fall through to the
// TextConfig field. quantGroup/quantBits have no text_config mirror, so this
// also pins their second-source (quantization_config) branch.
func TestModelConfigProbe_Accessors_Ugly(t *testing.T) {
	probe := &modelConfigProbe{}
	probe.TextConfig.VocabSize = 262144
	probe.TextConfig.HiddenSize = 2304
	probe.TextConfig.NumHiddenLayers = 26
	probe.TextConfig.NumKeyValueHeads = 4
	probe.TextConfig.HeadDim = 256
	probe.TextConfig.MaxPositionEmbeddings = 131072
	probe.QuantizationConfig = quantBlock{Present: true, Bits: 4, GroupSize: 64}

	if got := probe.numLayers(); got != 26 {
		t.Errorf("numLayers() fallback: got %d want 26", got)
	}
	if got := probe.vocabSize(); got != 262144 {
		t.Errorf("vocabSize() fallback: got %d want 262144", got)
	}
	if got := probe.hiddenSize(); got != 2304 {
		t.Errorf("hiddenSize() fallback: got %d want 2304", got)
	}
	if got := probe.numKeyValueHeads(); got != 4 {
		t.Errorf("numKeyValueHeads() fallback: got %d want 4", got)
	}
	if got := probe.headDim(); got != 256 {
		t.Errorf("headDim() fallback: got %d want 256", got)
	}
	if got := probe.contextLength(); got != 131072 {
		t.Errorf("contextLength() fallback: got %d want 131072", got)
	}
	if got := probe.quantBits(); got != 4 {
		t.Errorf("quantBits() from quantization_config: got %d want 4", got)
	}
	if got := probe.quantGroup(); got != 64 {
		t.Errorf("quantGroup() from quantization_config: got %d want 64", got)
	}
}
