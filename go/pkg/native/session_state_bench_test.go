// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkSessionStateSerializeCachedPrefix(b *testing.B) {
	requireNativeRuntime(b)
	s := newSessionStateFixture(b)
	if _, err := s.GenerateCached([]int32{1, 2, 3, 4, 5}, 6, -1); err != nil {
		b.Fatalf("GenerateCached warmup: %v", err)
	}
	blob, err := s.SerializeState()
	if err != nil {
		b.Fatalf("SerializeState warmup: %v", err)
	}
	b.SetBytes(int64(len(blob)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := s.SerializeState(); err != nil {
			b.Fatalf("SerializeState: %v", err)
		}
	}
}

func BenchmarkSessionStateRestorePromptCacheEntry(b *testing.B) {
	requireNativeRuntime(b)
	saved := newSessionStateFixture(b)
	prompt := []int32{1, 2, 3, 4, 5}
	if err := saved.WarmPromptCache(prompt); err != nil {
		b.Fatalf("WarmPromptCache: %v", err)
	}
	blob, err := saved.SerializeState()
	if err != nil {
		b.Fatalf("SerializeState: %v", err)
	}
	restored := newSessionStateFixture(b)
	if err := restored.RestoreState(blob); err != nil {
		b.Fatalf("RestoreState warmup: %v", err)
	}
	if hit := restored.CachedPrefixLen(prompt); hit != len(prompt) {
		b.Fatalf("restored prompt-cache hit = %d, want %d", hit, len(prompt))
	}
	b.SetBytes(int64(len(blob)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := restored.RestoreState(blob); err != nil {
			b.Fatalf("RestoreState: %v", err)
		}
	}
}
