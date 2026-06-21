// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	modelg4 "dappco.re/go/mlx/pkg/model/gemma4"
)

// TestConfigParseParity proves the literal copy of this package's parseGemma4Config into pkg/model/gemma4
// (the no-cgo backend's authoritative parser) stays identical to it — the anti-drift guarantee until
// pkg/metal is retired. It parses real-shape configs through BOTH parsers and asserts the arch fields
// agree (incl the multimodal wrapper-merge + the quant block), and that both REJECT the fail-loud cases
// (absent layer_types / max_position_embeddings) — the don't-guess discipline the earlier re-roll dropped.
func TestConfigParseParity(t *testing.T) {
	const textCfg = `{"model_type":"gemma4_text","hidden_size":64,"num_hidden_layers":4,"intermediate_size":128,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":16,"global_head_dim":32,"vocab_size":32,"rms_norm_eps":1e-6,"sliding_window":8,"max_position_embeddings":1024,"num_kv_shared_layers":2,"num_global_key_value_heads":2,"layer_types":["sliding_attention","sliding_attention","full_attention","sliding_attention"],"rope_parameters":{"full_attention":{"rope_theta":1000000,"rope_type":"proportional","partial_rotary_factor":0.25},"sliding_attention":{"rope_theta":10000,"rope_type":"default"}}}`
	const wrapperCfg = `{"model_type":"gemma4","architectures":["Gemma4ForConditionalGeneration"],"max_position_embeddings":2048,"quantization":{"group_size":32,"bits":4,"mode":"affine"},"text_config":{"model_type":"gemma4_text","hidden_size":80,"num_hidden_layers":2,"intermediate_size":160,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":20,"vocab_size":48,"sliding_window":16,"max_position_embeddings":2048,"layer_types":["full_attention","sliding_attention"],"rope_parameters":{"full_attention":{"rope_theta":1000000}}}}`
	const moeCfg = `{"model_type":"gemma4_text","hidden_size":64,"num_hidden_layers":2,"intermediate_size":128,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":16,"vocab_size":32,"sliding_window":8,"max_position_embeddings":512,"layer_types":["full_attention","full_attention"],"enable_moe_block":true,"num_experts":8,"top_k_experts":2,"moe_intermediate_size":64}`

	for _, tc := range []struct{ name, js string }{
		{"text", textCfg}, {"multimodal-wrapper", wrapperCfg}, {"moe", moeCfg},
	} {
		m, merr := parseGemma4Config([]byte(tc.js))
		n, nerr := modelg4.ParseConfig([]byte(tc.js))
		if merr != nil || nerr != nil {
			t.Fatalf("%s: metal err=%v, neutral err=%v (both should parse)", tc.name, merr, nerr)
		}
		if m.HiddenSize != n.HiddenSize || m.NumHiddenLayers != n.NumHiddenLayers ||
			m.IntermediateSize != n.IntermediateSize || m.NumAttentionHeads != n.NumAttentionHeads ||
			m.NumKeyValueHeads != n.NumKeyValueHeads || m.HeadDim != n.HeadDim || m.GlobalHeadDim != n.GlobalHeadDim ||
			m.VocabSize != n.VocabSize || m.RMSNormEps != n.RMSNormEps || m.MaxPositionEmbeddings != n.MaxPositionEmbeddings ||
			m.SlidingWindow != n.SlidingWindow || m.NumKVSharedLayers != n.NumKVSharedLayers ||
			m.EnableMoEBlock != n.EnableMoEBlock || len(m.LayerTypes) != len(n.LayerTypes) {
			t.Fatalf("%s: arch fields diverge between metal and neutral parse\n metal=%+v\n neutral=%+v", tc.name, m, n)
		}
		for i := range m.LayerTypes {
			if m.LayerTypes[i] != n.LayerTypes[i] {
				t.Fatalf("%s: layer_types[%d] %q != %q", tc.name, i, m.LayerTypes[i], n.LayerTypes[i])
			}
		}
		if (m.NumGlobalKeyValueHeads == nil) != (n.NumGlobalKeyValueHeads == nil) {
			t.Fatalf("%s: num_global_key_value_heads nil-ness diverges", tc.name)
		}
		if m.NumGlobalKeyValueHeads != nil && *m.NumGlobalKeyValueHeads != *n.NumGlobalKeyValueHeads {
			t.Fatalf("%s: num_global_key_value_heads %d != %d", tc.name, *m.NumGlobalKeyValueHeads, *n.NumGlobalKeyValueHeads)
		}
		if len(m.RopeParameters) != len(n.RopeParameters) {
			t.Fatalf("%s: rope_parameters count %d != %d", tc.name, len(m.RopeParameters), len(n.RopeParameters))
		}
		for k, mr := range m.RopeParameters {
			nr, ok := n.RopeParameters[k]
			if !ok || mr.RopeTheta != nr.RopeTheta || mr.PartialRotaryFactor != nr.PartialRotaryFactor || mr.RopeType != nr.RopeType {
				t.Fatalf("%s: rope_parameters[%q] diverges (metal %+v vs neutral %+v)", tc.name, k, mr, nr)
			}
		}
		if (m.Quantization == nil) != (n.Quantization == nil) {
			t.Fatalf("%s: quantization nil-ness diverges", tc.name)
		}
		if m.Quantization != nil && (m.Quantization.GroupSize != n.Quantization.GroupSize ||
			m.Quantization.Bits != n.Quantization.Bits || m.Quantization.Mode != n.Quantization.Mode) {
			t.Fatalf("%s: quantization diverges (metal %+v vs neutral %+v)", tc.name, m.Quantization, n.Quantization)
		}
	}

	// fail-loud parity: the don't-guess rejects must agree (the re-roll silently accepted these).
	for _, fc := range []struct{ name, js string }{
		{"absent layer_types", `{"model_type":"gemma4_text","hidden_size":64,"num_hidden_layers":2,"intermediate_size":128,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":32,"sliding_window":8,"max_position_embeddings":512}`},
		{"absent max_position_embeddings", `{"model_type":"gemma4_text","hidden_size":64,"num_hidden_layers":1,"intermediate_size":128,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":32,"sliding_window":8,"layer_types":["full_attention"]}`},
	} {
		_, me := parseGemma4Config([]byte(fc.js))
		_, ne := modelg4.ParseConfig([]byte(fc.js))
		if me == nil || ne == nil {
			t.Fatalf("%s: both parsers must reject; metal err=%v, neutral err=%v", fc.name, me, ne)
		}
	}
	t.Logf("config parse parity: metal parseGemma4Config == pkg/model/gemma4.ParseConfig across text/wrapper/moe + fail-loud")
}
