// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkMTPDecodeDraftEqualsTarget(b *testing.B) {
	requireNativeRuntime(b)
	const K, maxNew = 4, 16
	prompt := []int32{1, 2, 3, 4, 5}
	mk := newMTPDecodeFixture(b)
	target := mk()
	draft := mk()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		target.pos = 0
		draft.pos = 0
		res, err := MTPDecode(target, draft, prompt, maxNew, -1, K)
		if err != nil {
			b.Fatalf("MTPDecode: %v", err)
		}
		if res.Accepted != res.Drafted {
			b.Fatalf("accepted %d drafted %d", res.Accepted, res.Drafted)
		}
	}
}

func BenchmarkMTPDecodeDensePromptPrefill(b *testing.B) {
	requireNativeRuntime(b)
	const K, maxNew = 4, 1
	prompt := []int32{1, 2, 3, 4, 5}
	mk := newMTPDecodeFixture(b)
	target := mk()
	draft := mk()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		target.pos = 0
		draft.pos = 0
		res, err := MTPDecode(target, draft, prompt, maxNew, -1, K)
		if err != nil {
			b.Fatalf("MTPDecode: %v", err)
		}
		if len(res.Tokens) != maxNew {
			b.Fatalf("tokens = %d, want %d", len(res.Tokens), maxNew)
		}
	}
}

func BenchmarkMTPDecodeSequentialFallback(b *testing.B) {
	requireNativeRuntime(b)
	const K, maxNew = 4, 12
	prompt := []int32{1, 2, 3, 4, 5}
	mk := newMTPDecodeFixture(b)
	target := mtpSequentialFallbackSession(mk())
	draft := mtpSequentialFallbackSession(mk())

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		target.pos = 0
		draft.pos = 0
		res, err := MTPDecode(target, draft, prompt, maxNew, -1, K)
		if err != nil {
			b.Fatalf("MTPDecode: %v", err)
		}
		if res.Accepted != res.Drafted {
			b.Fatalf("accepted %d drafted %d", res.Accepted, res.Drafted)
		}
	}
}
