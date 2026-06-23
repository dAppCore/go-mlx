// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkNewArchSession(b *testing.B) {
	requireNativeRuntime(b)

	g, arch := gemma4BF16Fixture(b, 64, 1, 1, 64, 128, 32, 1)
	b.SetBytes(int64(len(g.Embed) + len(g.Layers[0].WGate)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sess, err := NewArchSession(g, arch, 4)
		if err != nil {
			b.Fatal(err)
		}
		_ = sess.Close()
	}
}

func BenchmarkArchSessionGenerateJoinedPrompt(b *testing.B) {
	requireNativeRuntime(b)

	g, arch := gemma4BF16Fixture(b, 128, 2, 1, 64, 256, 64, 2)
	prefix := []int32{1, 2, 3}
	suffix := []int32{4, 5}
	full := append(append([]int32{}, prefix...), suffix...)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sess, err := NewArchSession(g, arch, 24)
		if err != nil {
			b.Fatal(err)
		}
		if _, err := sess.Generate(full, 4, -1); err != nil {
			b.Fatalf("Generate: %v", err)
		}
		_ = sess.Close()
	}
}

func BenchmarkArchSessionPrefillAppendGenerateFromCache(b *testing.B) {
	requireNativeRuntime(b)

	g, arch := gemma4BF16Fixture(b, 128, 2, 1, 64, 256, 64, 2)
	prefix := []int32{1, 2, 3}
	suffix := []int32{4, 5}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sess, err := NewArchSession(g, arch, 24)
		if err != nil {
			b.Fatal(err)
		}
		if err := sess.PrefillTokens(prefix); err != nil {
			b.Fatalf("PrefillTokens: %v", err)
		}
		if err := sess.AppendTokens(suffix); err != nil {
			b.Fatalf("AppendTokens: %v", err)
		}
		if _, err := sess.GenerateFromCache(4, -1); err != nil {
			b.Fatalf("GenerateFromCache: %v", err)
		}
		_ = sess.Close()
	}
}

func BenchmarkArchSessionReplayFullPromptSecondTurn(b *testing.B) {
	requireNativeRuntime(b)

	g, arch := gemma4BF16Fixture(b, 128, 2, 1, 64, 256, 64, 2)
	full := []int32{1, 2, 3, 4, 5}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		sess, err := NewArchSession(g, arch, 24)
		if err != nil {
			b.Fatal(err)
		}
		b.StartTimer()
		if _, err := sess.Generate(full, 4, -1); err != nil {
			b.Fatalf("Generate: %v", err)
		}
		b.StopTimer()
		_ = sess.Close()
		b.StartTimer()
	}
}

func BenchmarkArchSessionAppendGenerateFromCacheSecondTurn(b *testing.B) {
	requireNativeRuntime(b)

	g, arch := gemma4BF16Fixture(b, 128, 2, 1, 64, 256, 64, 2)
	prefix := []int32{1, 2, 3}
	suffix := []int32{4, 5}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		sess, err := NewArchSession(g, arch, 24)
		if err != nil {
			b.Fatal(err)
		}
		if err := sess.PrefillTokens(prefix); err != nil {
			b.Fatalf("PrefillTokens: %v", err)
		}
		b.StartTimer()
		if err := sess.AppendTokens(suffix); err != nil {
			b.Fatalf("AppendTokens: %v", err)
		}
		if _, err := sess.GenerateFromCache(4, -1); err != nil {
			b.Fatalf("GenerateFromCache: %v", err)
		}
		b.StopTimer()
		_ = sess.Close()
		b.StartTimer()
	}
}
