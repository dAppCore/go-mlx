// SPDX-Licence-Identifier: EUPL-1.2

// Benchmark for the root Model.NewSession constructor — the type
// assertion + Info() copy + session wrap. The session machinery itself
// is benched in the session package.
//
// Run:    go test -bench='BenchmarkSession_NewSession' -benchmem -run='^$' ./go

package mlx

import (
	"testing"

	"dappco.re/go/mlx/internal/sessionfake"
)

// Sinks defeat compiler DCE.
var (
	sessionBenchSinkErr     error
	sessionBenchSinkSession *ModelSession
)

func BenchmarkSession_NewSession(b *testing.B) {
	native := &sessionfake.Handle{}
	model := &Model{model: &fakeNativeModel{session: native}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sess, err := model.NewSession()
		sessionBenchSinkErr = err
		sessionBenchSinkSession = sess
	}
}

// --- Prefill / AppendPrompt — pure Go glue, fake native is a no-op ---
