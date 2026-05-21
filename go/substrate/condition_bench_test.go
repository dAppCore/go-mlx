// SPDX-Licence-Identifier: EUPL-1.2

package substrate

import "testing"

func BenchmarkNormalize_ConditionAlias(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = Normalize("continuous-with-gap")
	}
}

func BenchmarkConditionTransition_FourConditions(b *testing.B) {
	conditions := All()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		for _, condition := range conditions {
			_ = condition.RequiresReplay()
			_ = condition.UsesContinuousState()
			_ = condition.RequiresArtificialGap()
			_ = condition.MeasuresPrefillGap()
		}
	}
}
