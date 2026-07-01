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

func BenchmarkMTPVerifyBatchedFallbackReusedHiddenRows(b *testing.B) {
	requireNativeRuntime(b)
	mk := newMTPDecodeFixture(b)
	dense := mk()
	for _, id := range []int32{1, 2, 3} {
		if _, err := dense.stepID(id); err != nil {
			b.Fatalf("prefill dense stepID(%d): %v", id, err)
		}
	}
	dense.greedy = func(hidden []byte, suppress []int32) (int32, bool, error) {
		return dense.headEnc.greedyInPool(hidden, suppress)
	}
	ids := []int32{4, 5, 6, 7}
	greedys := make([]int32, len(ids))
	if _, ok, err := dense.verifyBatchedInto(ids, greedys); err != nil {
		b.Fatalf("verifyBatched warmup: %v", err)
	} else if !ok {
		b.Fatal("verifyBatched warmup ok = false")
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, ok, err := dense.verifyBatchedInto(ids, greedys); err != nil {
			b.Fatalf("verifyBatched: %v", err)
		} else if !ok {
			b.Fatal("verifyBatched ok = false")
		}
	}
}
