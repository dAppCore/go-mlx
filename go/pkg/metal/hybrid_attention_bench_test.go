// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

var hybridAttentionBenchPlanSink HybridAttentionCachePlan

func BenchmarkBuildHybridAttentionCachePlan_Qwen36_64Layers(b *testing.B) {
	layerTypes := []string{"linear_attention", "full_attention"}
	b.ReportAllocs()
	for b.Loop() {
		var err error
		hybridAttentionBenchPlanSink, err = BuildHybridAttentionCachePlan(64, layerTypes, 1024)
		if err != nil {
			b.Fatalf("BuildHybridAttentionCachePlan() error = %v", err)
		}
	}
}
