// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/safetensors"
)

func TestLoadMistralBF16ParsesConfigAndWeights(t *testing.T) {
	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers = 64, 1, 1, 64, 128, 32, 2
	cfg, arch := mistralConfigFixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	configJSON := core.JSONMarshal(cfg)
	if !configJSON.OK {
		t.Fatalf("marshal config: %s", configJSON.Error())
	}
	blob, err := safetensors.Encode(mistralTensorFixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers))
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}

	g, gotArch, err := LoadMistralBF16(configJSON.Value.([]byte), blob)
	if err != nil {
		t.Fatalf("LoadMistralBF16: %v", err)
	}
	if gotArch.Hidden != arch.Hidden || gotArch.HeadDim != arch.HeadDim || gotArch.Vocab != arch.Vocab || len(gotArch.Layer) != len(arch.Layer) {
		t.Fatalf("arch round-trip mismatch: got %+v, want %+v", gotArch, arch)
	}
	if g == nil || !g.Tied || len(g.Layers) != nLayers {
		t.Fatalf("loaded model not assembled as tied dense Mistral: %#v", g)
	}
	if _, _, err := LoadMistralBF16([]byte("{"), blob); err == nil {
		t.Fatal("bad config JSON should fail")
	}
	if _, _, err := LoadMistralBF16(configJSON.Value.([]byte), []byte("not safetensors")); err == nil {
		t.Fatal("bad safetensors blob should fail")
	}
}
