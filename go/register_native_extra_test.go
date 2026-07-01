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
	"context"
	"testing"
	"time"

	"dappco.re/go/inference"
	"dappco.re/go/mlx/pkg/model"
	pkgtokenizer "dappco.re/go/mlx/pkg/tokenizer"
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

type nativeTextInfoTokenModel struct{}

func (nativeTextInfoTokenModel) Embed(int32) ([]byte, error) { return []byte{0}, nil }

func (nativeTextInfoTokenModel) DecodeForward([][]byte) ([][]byte, error) {
	return [][]byte{{0}}, nil
}

func (nativeTextInfoTokenModel) Head([]byte) ([]byte, error) { return make([]byte, 16), nil }

func (nativeTextInfoTokenModel) Vocab() int { return 8 }

func (nativeTextInfoTokenModel) NumLayers() int { return 3 }

func (nativeTextInfoTokenModel) HiddenSize() int { return 16 }

func (nativeTextInfoTokenModel) QuantBits() int { return 4 }

func (nativeTextInfoTokenModel) QuantGroup() int { return 64 }

func TestNativeTextModel_InfoUsesNativeTokenMetadata_Good(t *testing.T) {
	m := &nativeTextModel{tm: nativeTextInfoTokenModel{}, modelType: "gemma4", info: inference.ModelInfo{Architecture: "gemma4"}}
	info := m.Info()
	if info.VocabSize != 8 || info.NumLayers != 3 || info.HiddenSize != 16 {
		t.Fatalf("Info shape = %+v, want vocab/layers/hidden = 8/3/16", info)
	}
	if info.QuantBits != 4 || info.QuantGroup != 64 {
		t.Fatalf("Info quant = %d/%d, want 4/64", info.QuantBits, info.QuantGroup)
	}
	if layers := m.NumLayers(); layers != 3 {
		t.Fatalf("NumLayers = %d, want 3 from native metadata", layers)
	}
}

func TestNativeTextModel_FormatChatUsesSharedGemma4Template_Good(t *testing.T) {
	off := false
	m := &nativeTextModel{tm: nativeTextInfoTokenModel{}, modelType: "gemma4", info: inference.ModelInfo{Architecture: "gemma4_text"}}

	got := m.formatChat([]inference.Message{{Role: "user", Content: "hi"}}, inference.GenerateConfig{EnableThinking: &off})
	want := "<bos><|turn>user\nhi<turn|>\n<|turn>model\n"
	if got != want {
		t.Fatalf("native formatChat thinking-off = %q, want shared Gemma4 template %q", got, want)
	}

	on := true
	got = m.formatChat([]inference.Message{{Role: "user", Content: "hi"}}, inference.GenerateConfig{EnableThinking: &on})
	want = "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nhi<turn|>\n<|turn>model\n"
	if got != want {
		t.Fatalf("native formatChat thinking-on = %q, want shared Gemma4 template %q", got, want)
	}
}

func TestNativeTextModel_ApplyChatTemplateUsesSharedTemplate_Good(t *testing.T) {
	m := &nativeTextModel{tm: nativeTextInfoTokenModel{}, modelType: "gemma4", info: inference.ModelInfo{Architecture: "gemma4_text"}}

	got, err := m.ApplyChatTemplate([]inference.Message{{Role: "user", Content: "hi"}})
	if err != nil {
		t.Fatalf("ApplyChatTemplate error = %v", err)
	}
	want := m.formatChat([]inference.Message{{Role: "user", Content: "hi"}}, inference.GenerateConfig{})
	if got != want {
		t.Fatalf("ApplyChatTemplate = %q, want shared formatter output %q", got, want)
	}
}

func TestNativeTextModel_ApplyChatTemplateNil_Bad(t *testing.T) {
	var m *nativeTextModel
	if _, err := m.ApplyChatTemplate([]inference.Message{{Role: "user", Content: "hi"}}); err == nil {
		t.Fatal("ApplyChatTemplate(nil) error = nil, want errMLXModelNil")
	}
}

func TestNativeTextModel_TokenizerModelParity_Good(t *testing.T) {
	var _ inference.TokenizerModel = (*nativeTextModel)(nil)
	tok, err := pkgtokenizer.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	m := &nativeTextModel{tok: tok}

	ids := m.Encode("hello")
	if len(ids) != 2 || ids[0] != 0 || ids[1] != 10 {
		t.Fatalf("Encode(\"hello\") = %v, want tokenizer ids [0 10]", ids)
	}
	if got := m.Decode(ids); got != "hello" {
		t.Fatalf("Decode(%v) = %q, want hello", ids, got)
	}
}

func TestNativeTextModel_TokenizerModelNil_Bad(t *testing.T) {
	var m *nativeTextModel
	if ids := m.Encode("hello"); ids != nil {
		t.Fatalf("Encode(nil) = %v, want nil", ids)
	}
	if got := m.Decode([]int32{1, 2}); got != "" {
		t.Fatalf("Decode(nil) = %q, want empty string", got)
	}
}

func TestNativeTextModel_OutputParserParity_Good(t *testing.T) {
	var _ inference.ReasoningParser = (*nativeTextModel)(nil)
	var _ inference.ToolParser = (*nativeTextModel)(nil)
	m := &nativeTextModel{modelType: "gemma4", info: inference.ModelInfo{Architecture: "gemma4_text"}}

	reasoning, err := m.ParseReasoning(nil, "<think>scratch</think>answer")
	if err != nil {
		t.Fatalf("ParseReasoning error = %v", err)
	}
	if reasoning.VisibleText == "" {
		t.Fatalf("ParseReasoning = %+v, want visible text", reasoning)
	}
	tools, err := m.ParseTools(nil, "plain")
	if err != nil {
		t.Fatalf("ParseTools error = %v", err)
	}
	if len(tools.Calls) != 0 || tools.VisibleText != "plain" {
		t.Fatalf("ParseTools = %+v, want plain visible text and no calls", tools)
	}
}

func TestNativeTextModel_OutputParserNil_Good(t *testing.T) {
	var m *nativeTextModel
	reasoning, err := m.ParseReasoning(nil, "<think>scratch</think>answer")
	if err != nil {
		t.Fatalf("ParseReasoning(nil) error = %v", err)
	}
	if reasoning.VisibleText == "" {
		t.Fatalf("ParseReasoning(nil) = %+v, want visible text", reasoning)
	}
}

func TestNativeTextModel_CapabilitiesReportsActualNativeSurface_Good(t *testing.T) {
	var _ inference.CapabilityReporter = (*nativeTextModel)(nil)
	m := &nativeTextModel{modelType: "gemma4", info: inference.ModelInfo{Architecture: "gemma4_text", NumLayers: 3}}

	report := m.Capabilities()
	if report.Runtime.Backend != "native" || !report.Runtime.NativeRuntime {
		t.Fatalf("runtime = %+v, want native runtime", report.Runtime)
	}
	for _, id := range []inference.CapabilityID{
		inference.CapabilityGenerate,
		inference.CapabilityChat,
		inference.CapabilityTokenizer,
		inference.CapabilityChatTemplate,
		inference.CapabilityReasoningParse,
		inference.CapabilityToolParse,
		inference.CapabilityEvaluation,
		inference.CapabilityCacheWarm,
		inference.CapabilityAttentionProbe,
		inference.CapabilityProbeEvents,
		inference.CapabilityScheduler,
		inference.CapabilityRequestCancel,
		inference.CapabilityLoRAInference,
	} {
		if !report.Supports(id) {
			t.Fatalf("capabilities = %v, want %s", report.CapabilityIDs(), id)
		}
	}
}

func TestNativeTextModel_EvaluateDatasetStream_Good(t *testing.T) {
	var _ inference.Evaluator = (*nativeTextModel)(nil)
	tok, err := pkgtokenizer.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	tm := &nativeEvalTextTokenModel{
		vocab:   64,
		session: &nativeEvalTextSession{},
	}
	m := &nativeTextModel{
		tm:        tm,
		tok:       tok,
		modelType: "gemma4",
		info:      inference.ModelInfo{Architecture: "gemma4_text", VocabSize: 64},
	}
	stream := &inferenceContractDatasetStream{
		samples: []inference.DatasetSample{{Prompt: "hello", Response: "hello"}},
	}

	report, err := m.Evaluate(context.Background(), stream, inference.EvalConfig{BatchSize: 1, MaxSeqLen: 8})
	if err != nil {
		t.Fatalf("Evaluate error = %v", err)
	}
	if report == nil || report.Metrics.Samples != 1 || report.Metrics.Tokens == 0 {
		t.Fatalf("Evaluate report = %+v, want one sample with loss tokens", report)
	}
	if report.Model.Architecture != "gemma4_text" || report.Model.VocabSize != 64 {
		t.Fatalf("Evaluate model = %+v, want native model identity", report.Model)
	}
	if tm.decodeForwardCalls != 0 {
		t.Fatalf("DecodeForward calls = %d, want session eval path", tm.decodeForwardCalls)
	}
	if tm.session.stepWithIDCalls == 0 {
		t.Fatalf("StepWithID calls = %d, want id-aware native eval session path", tm.session.stepWithIDCalls)
	}
	if tm.headCalls != report.Metrics.Tokens {
		t.Fatalf("Head calls = %d, want one per loss token %d", tm.headCalls, report.Metrics.Tokens)
	}
}

type nativeEvalTextTokenModel struct {
	vocab              int
	headCalls          int
	decodeForwardCalls int
	session            *nativeEvalTextSession
}

func (m *nativeEvalTextTokenModel) Embed(id int32) ([]byte, error) { return []byte{byte(id)}, nil }

func (m *nativeEvalTextTokenModel) DecodeForward([][]byte) ([][]byte, error) {
	m.decodeForwardCalls++
	return nil, coreError("DecodeForward should not be used by session eval")
}

func (m *nativeEvalTextTokenModel) Head([]byte) ([]byte, error) {
	m.headCalls++
	return make([]byte, m.vocab*2), nil
}

func (m *nativeEvalTextTokenModel) Vocab() int { return m.vocab }

func (m *nativeEvalTextTokenModel) OpenSession() (model.DecodeStepper, error) {
	return m.session, nil
}

type nativeEvalTextSession struct {
	stepWithIDCalls int
	ids             []int32
}

func (s *nativeEvalTextSession) Step([]byte) ([]byte, error) {
	return nil, coreError("Step should not be used when StepWithID exists")
}

func (s *nativeEvalTextSession) StepWithID(id int32, emb []byte) ([]byte, error) {
	s.stepWithIDCalls++
	s.ids = append(s.ids, id)
	return emb, nil
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
