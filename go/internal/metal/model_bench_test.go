// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

var benchmarkProbeModelTypeResult string

func BenchmarkModel_ProbeModelType_MetadataGuardFamilies(b *testing.B) {
	configs := [][]byte{
		[]byte(`{"architectures":["MixtralForCausalLM"],"model_type":"mixtral","hidden_size":1024}`),
		[]byte(`{"architectures":["DeepseekV3ForCausalLM"],"model_type":"deepseek_v3","hidden_size":1024}`),
		[]byte(`{"architectures":["GptOssForCausalLM"],"model_type":"gpt_oss","hidden_size":1024}`),
		[]byte(`{"architectures":["KimiForCausalLM"],"model_type":"kimi","hidden_size":1024}`),
		[]byte(`{"architectures":["BertModel"],"model_type":"bert","hidden_size":384}`),
		[]byte(`{"architectures":["BertForSequenceClassification"],"model_type":"bert","hidden_size":768}`),
	}
	b.ReportAllocs()
	for b.Loop() {
		for _, config := range configs {
			got, err := probeModelType(config)
			if err != nil {
				b.Fatalf("probeModelType() error = %v", err)
			}
			benchmarkProbeModelTypeResult = got
		}
	}
}
