// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// Coverage for the no-cgo nativeTextModel adapter's pure surface: the gemma
// chat formatter and the small state accessors (Err/Metrics/Close/ModelType/
// Info and the setErr/setMetrics mutators). The token-loop methods (Generate/
// Chat/stream/Classify/BatchGenerate) drive a model.TokenModel forward pass and
// are covered by the model-backed serve path; here we exercise everything that
// needs only a constructed struct.

package mlx

import (
	"testing"
	"time"

	"dappco.re/go/inference"
)

func TestNativeTextModel_FormatGemmaChat_Good(t *testing.T) {
	out := formatGemmaChat([]inference.Message{
		{Role: "system", Content: "be brief"},
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "hello"},
	})
	// gemma has no system role: system + user both render as "user"; assistant
	// renders as "model"; a trailing model turn opens the completion.
	want := "<bos>" +
		"<start_of_turn>user\nbe brief<end_of_turn>\n" +
		"<start_of_turn>user\nhi<end_of_turn>\n" +
		"<start_of_turn>model\nhello<end_of_turn>\n" +
		"<start_of_turn>model\n"
	if out != want {
		t.Fatalf("formatGemmaChat =\n%q\nwant\n%q", out, want)
	}
}

func TestNativeTextModel_FormatGemmaChat_Empty_Ugly(t *testing.T) {
	// No messages still opens a trailing model turn after <bos>.
	if got := formatGemmaChat(nil); got != "<bos><start_of_turn>model\n" {
		t.Fatalf("formatGemmaChat(nil) = %q", got)
	}
}

func TestNativeTextModel_TypeAndInfo_Good(t *testing.T) {
	m := &nativeTextModel{modelType: "gemma4", info: inference.ModelInfo{Architecture: "gemma4", VocabSize: 262144}}
	if m.ModelType() != "gemma4" {
		t.Fatalf("ModelType = %q, want gemma4", m.ModelType())
	}
	if info := m.Info(); info.Architecture != "gemma4" || info.VocabSize != 262144 {
		t.Fatalf("Info = %+v, want gemma4/262144", info)
	}
	if err := m.Close(); err != nil { // Close is a documented no-op.
		t.Fatalf("Close = %v, want nil", err)
	}
}

func TestNativeTextModel_ErrAndMetrics_Good(t *testing.T) {
	m := &nativeTextModel{}
	// Fresh model: no error, zero metrics.
	if m.Err() != nil {
		t.Fatalf("fresh Err = %v, want nil", m.Err())
	}
	if (m.Metrics() != inference.GenerateMetrics{}) {
		t.Fatalf("fresh Metrics = %+v, want zero", m.Metrics())
	}

	// setMetrics records a successful run and clears any prior error; the
	// tokens/sec is genTokens/total.
	m.setErr(coreError("boom"))
	if m.Err() == nil {
		t.Fatal("Err after setErr = nil, want error")
	}
	m.setMetrics(10, 20, time.Second)
	if m.Err() != nil {
		t.Fatalf("Err after setMetrics = %v, want cleared", m.Err())
	}
	got := m.Metrics()
	if got.PromptTokens != 10 || got.GeneratedTokens != 20 {
		t.Fatalf("Metrics tokens = %d/%d, want 10/20", got.PromptTokens, got.GeneratedTokens)
	}
	if got.DecodeTokensPerSec != 20 { // 20 tokens / 1s
		t.Fatalf("DecodeTokensPerSec = %v, want 20", got.DecodeTokensPerSec)
	}
}

func TestNativeTextModel_SetMetrics_ZeroDurationNoDivByZero_Ugly(t *testing.T) {
	m := &nativeTextModel{}
	m.setMetrics(5, 0, 0) // zero duration must not divide-by-zero
	if got := m.Metrics(); got.DecodeTokensPerSec != 0 {
		t.Fatalf("DecodeTokensPerSec on zero duration = %v, want 0", got.DecodeTokensPerSec)
	}
}

// coreError is a tiny error helper so the test does not import core just for one
// sentinel; the production setErr only stores and returns it.
type coreError string

func (e coreError) Error() string { return string(e) }
