// SPDX-Licence-Identifier: EUPL-1.2

package loraadapter

import "testing"

// BenchmarkParseConfig exercises the adapter_config.json parse + alias
// normalisation path. This package only parses/normalises metadata; the actual
// LoRA delta apply/merge lives in native Metal, so the flat allocator profile
// here is JSON decode only (AX-11 evidence, not a load-path hot loop).
func BenchmarkParseConfig(b *testing.B) {
	data := []byte(`{"r":8,"lora_alpha":16,"target_modules":["q_proj","k_proj","v_proj","o_proj"],"num_layers":32}`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := ParseConfig(data); err != nil {
			b.Fatalf("ParseConfig() error = %v", err)
		}
	}
}
