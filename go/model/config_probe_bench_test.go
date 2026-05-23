// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the model/config_probe.go architecture-detection
// helpers. Per AX-11 — these fire on every Inspect call against a
// model directory. The HF class-name classifier in particular runs
// the full alternation chain on every architecture string we see —
// real workloads classify dozens of candidates while planning fits.
//
// Run:    go test -bench=Benchmark -benchmem -run='^$' ./go/model

package model

import (
	"testing"

	core "dappco.re/go"
)

// Sinks defeat compiler DCE.
var (
	probeSinkString string
	probeSinkInt    int
	probeSinkProbe  *modelConfigProbe
	probeSinkErr    error
)

// --- normalizeKnownArchitecture — switch hot loop ---

func BenchmarkModel_NormalizeKnownArchitecture_MiniMax(b *testing.B) {
	name := "MiniMax-M2"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		probeSinkString = normalizeKnownArchitecture(name)
	}
}

func BenchmarkModel_NormalizeKnownArchitecture_Qwen2_5(b *testing.B) {
	name := "qwen2.5"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		probeSinkString = normalizeKnownArchitecture(name)
	}
}

func BenchmarkModel_NormalizeKnownArchitecture_Qwen3_6(b *testing.B) {
	name := "qwen3_5_text"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		probeSinkString = normalizeKnownArchitecture(name)
	}
}

func BenchmarkModel_NormalizeKnownArchitecture_Passthrough(b *testing.B) {
	name := "qwen3"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		probeSinkString = normalizeKnownArchitecture(name)
	}
}

// --- architectureFromTransformersName — common HF class-name shapes ---

func BenchmarkModel_ArchitectureFromTransformersName_Qwen3(b *testing.B) {
	name := "Qwen3ForCausalLM"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		probeSinkString = architectureFromTransformersName(name)
	}
}

func BenchmarkModel_ArchitectureFromTransformersName_Qwen3MoE(b *testing.B) {
	name := "Qwen3MoeForCausalLM"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		probeSinkString = architectureFromTransformersName(name)
	}
}

func BenchmarkModel_ArchitectureFromTransformersName_Qwen3_6(b *testing.B) {
	name := "Qwen3_5ForConditionalGeneration"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		probeSinkString = architectureFromTransformersName(name)
	}
}

func BenchmarkModel_ArchitectureFromTransformersName_Gemma4(b *testing.B) {
	name := "Gemma4ForCausalLM"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		probeSinkString = architectureFromTransformersName(name)
	}
}

func BenchmarkModel_ArchitectureFromTransformersName_BertRerank(b *testing.B) {
	name := "BertForSequenceClassification"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		probeSinkString = architectureFromTransformersName(name)
	}
}

// Miss path — every contains check fires, returns "".
func BenchmarkModel_ArchitectureFromTransformersName_Unknown(b *testing.B) {
	name := "SomeFutureMythicalArchitectureForCausalLM"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		probeSinkString = architectureFromTransformersName(name)
	}
}

// --- compactArchitectureName — inner helper, fires before every classification ---

func BenchmarkModel_CompactArchitectureName_Short(b *testing.B) {
	name := "Qwen3MoeForCausalLM"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		probeSinkString = compactArchitectureName(name)
	}
}

func BenchmarkModel_CompactArchitectureName_Long(b *testing.B) {
	name := "XLMRobertaForSequenceClassification"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		probeSinkString = compactArchitectureName(name)
	}
}

// --- modelConfigProbe accessors — fire per-Inspect call ---

func benchProbe() *modelConfigProbe {
	return &modelConfigProbe{
		ModelType:             "qwen3",
		Architectures:         []string{"Qwen3ForCausalLM"},
		VocabSize:             151936,
		HiddenSize:            2048,
		NumHiddenLayers:       28,
		MaxPositionEmbeddings: 40960,
		QuantizationConfig: &struct {
			Bits      int `json:"bits"`
			GroupSize int `json:"group_size"`
		}{Bits: 4, GroupSize: 64},
	}
}

func benchProbeNestedText() *modelConfigProbe {
	probe := &modelConfigProbe{
		ModelType:     "qwen3_5",
		Architectures: []string{"Qwen3_5ForConditionalGeneration"},
	}
	probe.TextConfig.ModelType = "qwen3_5_text"
	probe.TextConfig.HiddenSize = 5120
	probe.TextConfig.NumHiddenLayers = 64
	probe.TextConfig.VocabSize = 248320
	probe.TextConfig.MaxPositionEmbeddings = 262144
	return probe
}

func BenchmarkModel_Probe_Architecture_Direct(b *testing.B) {
	probe := benchProbe()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		probeSinkString = probe.architecture()
	}
}

func BenchmarkModel_Probe_Architecture_NestedText(b *testing.B) {
	probe := benchProbeNestedText()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		probeSinkString = probe.architecture()
	}
}

func BenchmarkModel_Probe_NumLayers(b *testing.B) {
	probe := benchProbe()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		probeSinkInt = probe.numLayers()
	}
}

func BenchmarkModel_Probe_VocabSize(b *testing.B) {
	probe := benchProbe()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		probeSinkInt = probe.vocabSize()
	}
}

func BenchmarkModel_Probe_HiddenSize(b *testing.B) {
	probe := benchProbe()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		probeSinkInt = probe.hiddenSize()
	}
}

func BenchmarkModel_Probe_ContextLength(b *testing.B) {
	probe := benchProbe()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		probeSinkInt = probe.contextLength()
	}
}

func BenchmarkModel_Probe_QuantBits(b *testing.B) {
	probe := benchProbe()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		probeSinkInt = probe.quantBits()
	}
}

func BenchmarkModel_Probe_QuantGroup(b *testing.B) {
	probe := benchProbe()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		probeSinkInt = probe.quantGroup()
	}
}

// --- readModelConfig — disk read + JSON unmarshal of config.json ---

func BenchmarkModel_ReadModelConfig_Qwen3(b *testing.B) {
	dir := b.TempDir()
	if r := core.WriteFile(core.JoinPath(dir, "config.json"), []byte(`{
		"model_type": "qwen3",
		"architectures": ["Qwen3ForCausalLM"],
		"vocab_size": 151936,
		"hidden_size": 2048,
		"num_hidden_layers": 28,
		"max_position_embeddings": 40960,
		"quantization_config": {"bits": 4, "group_size": 64}
	}`), 0o644); !r.OK {
		b.Fatalf("WriteFile: %v", r.Value)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		probeSinkProbe, probeSinkErr = readModelConfig(dir)
	}
}

func BenchmarkModel_ReadModelConfig_NestedText(b *testing.B) {
	dir := b.TempDir()
	if r := core.WriteFile(core.JoinPath(dir, "config.json"), []byte(`{
		"model_type": "qwen3_5",
		"architectures": ["Qwen3_5ForConditionalGeneration"],
		"text_config": {
			"model_type": "qwen3_5_text",
			"vocab_size": 248320,
			"hidden_size": 5120,
			"num_hidden_layers": 64,
			"max_position_embeddings": 262144
		},
		"quantization": {"bits": 4, "group_size": 64}
	}`), 0o644); !r.OK {
		b.Fatalf("WriteFile: %v", r.Value)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		probeSinkProbe, probeSinkErr = readModelConfig(dir)
	}
}

// --- parseConfigProbe — pure JSON parse, no disk I/O ---
//
// Isolates the JSON unmarshal cost from file-system overhead so the
// W11-N hand-rolled walker payoff is visible without the b.TempDir +
// WriteFile + ReadFile floor. Three benches cover the canonical
// shapes the HF ecosystem ships: Qwen3 (dense LLM), Gemma3 (variant
// with text_config nest), Llama (long architectures slice).

var (
	configQwen3      = []byte(`{"model_type":"qwen3","architectures":["Qwen3ForCausalLM"],"vocab_size":151936,"hidden_size":2048,"num_hidden_layers":28,"max_position_embeddings":40960,"quantization_config":{"bits":4,"group_size":64}}`)
	configGemma3     = []byte(`{"model_type":"gemma3","architectures":["Gemma3ForCausalLM"],"text_config":{"model_type":"gemma3_text","vocab_size":262144,"hidden_size":2304,"num_hidden_layers":26,"max_position_embeddings":131072},"quantization":{"bits":4,"group_size":64}}`)
	configLlama      = []byte(`{"model_type":"llama","architectures":["LlamaForCausalLM"],"vocab_size":128256,"hidden_size":4096,"num_hidden_layers":32,"max_position_embeddings":8192}`)
	configBertRerank = []byte(`{"model_type":"bert","architectures":["BertForSequenceClassification"],"vocab_size":30522,"hidden_size":768,"num_hidden_layers":12,"max_position_embeddings":512,"num_labels":1}`)
)

func BenchmarkModel_ParseConfigProbe_Qwen3(b *testing.B) {
	data := configQwen3
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var probe modelConfigProbe
		if r := core.JSONUnmarshal(data, &probe); !r.OK {
			b.Fatalf("JSONUnmarshal: %v", r.Value)
		}
		probeSinkProbe = &probe
	}
}

func BenchmarkModel_ParseConfigProbe_Gemma3(b *testing.B) {
	data := configGemma3
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var probe modelConfigProbe
		if r := core.JSONUnmarshal(data, &probe); !r.OK {
			b.Fatalf("JSONUnmarshal: %v", r.Value)
		}
		probeSinkProbe = &probe
	}
}

func BenchmarkModel_ParseConfigProbe_Llama(b *testing.B) {
	data := configLlama
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var probe modelConfigProbe
		if r := core.JSONUnmarshal(data, &probe); !r.OK {
			b.Fatalf("JSONUnmarshal: %v", r.Value)
		}
		probeSinkProbe = &probe
	}
}

func BenchmarkModel_ParseConfigProbe_BertRerank(b *testing.B) {
	data := configBertRerank
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var probe modelConfigProbe
		if r := core.JSONUnmarshal(data, &probe); !r.OK {
			b.Fatalf("JSONUnmarshal: %v", r.Value)
		}
		probeSinkProbe = &probe
	}
}

// Multi-architecture variant — vision-text models often list 2-4
// architectures (e.g. Gemma4 with separate vision/text/audio heads).
// The pre-sized slice path saves the append growth here.
var configMultiArch = []byte(`{"model_type":"gemma4","architectures":["Gemma4ForCausalLM","Gemma4ForConditionalGeneration","Gemma4VisionModel","Gemma4ForAudio"],"vocab_size":262144,"hidden_size":2304}`)

func BenchmarkModel_ParseConfigProbe_MultiArch(b *testing.B) {
	data := configMultiArch
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var probe modelConfigProbe
		if r := core.JSONUnmarshal(data, &probe); !r.OK {
			b.Fatalf("JSONUnmarshal: %v", r.Value)
		}
		probeSinkProbe = &probe
	}
}
