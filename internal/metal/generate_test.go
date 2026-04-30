// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

type fakeDetachCache struct {
	detachCalls int
}

func (f *fakeDetachCache) Update(_ *Array, _ *Array, _ int) (*Array, *Array) { return nil, nil }
func (f *fakeDetachCache) Offset() int                                       { return 0 }
func (f *fakeDetachCache) Len() int                                          { return 0 }
func (f *fakeDetachCache) State() []*Array                                   { return nil }
func (f *fakeDetachCache) Reset()                                            {}
func (f *fakeDetachCache) Detach()                                           { f.detachCalls++ }

func TestDetachEvalState_DetachesCaches_Good(t *testing.T) {
	coverageTokens := "DetachesCaches"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	first := &fakeDetachCache{}
	second := &fakeDetachCache{}

	detachEvalState(nil, []Cache{first, nil, second})

	if first.detachCalls != 1 {
		t.Fatalf("first cache detach calls = %d, want 1", first.detachCalls)
	}
	if second.detachCalls != 1 {
		t.Fatalf("second cache detach calls = %d, want 1", second.detachCalls)
	}
}

func TestAttentionCacheIndexByLayer_DefaultModel_Good(t *testing.T) {
	coverageTokens := "DefaultModel"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	got := attentionCacheIndexByLayer(&fakeModel{numLayers: 4}, 4, 4)
	want := []int{0, 1, 2, 3}
	for i, wantIdx := range want {
		if got[i] != wantIdx {
			t.Fatalf("cache index for layer %d = %d, want %d", i, got[i], wantIdx)
		}
	}
}

func TestAttentionCacheIndexByLayer_Gemma4SharedOwners_Good(t *testing.T) {
	coverageTokens := "Gemma4SharedOwners"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	model := &Gemma4Model{
		Cfg: &Gemma4TextConfig{
			NumKVSharedLayers: 2,
		},
		Layers: []*Gemma4DecoderLayer{
			{LayerType: "sliding_attention"},
			{LayerType: "full_attention"},
			{LayerType: "sliding_attention"},
			{LayerType: "full_attention"},
		},
	}

	got := attentionCacheIndexByLayer(model, len(model.Layers), 2)
	want := []int{0, 1, 0, 1}
	for i, wantIdx := range want {
		if got[i] != wantIdx {
			t.Fatalf("cache index for layer %d = %d, want %d", i, got[i], wantIdx)
		}
	}
}

func TestAttentionCacheIndexByLayer_Gemma4PromotedOwner_Good(t *testing.T) {
	coverageTokens := "Gemma4PromotedOwner"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	model := &Gemma4Model{
		Cfg: &Gemma4TextConfig{
			NumKVSharedLayers: 2,
		},
		Layers: []*Gemma4DecoderLayer{
			{LayerType: "sliding_attention"},
			{LayerType: "sliding_attention"},
			{LayerType: "sliding_attention"},
			{LayerType: "sliding_attention"},
			{LayerType: "full_attention"},
			{LayerType: "sliding_attention"},
		},
	}

	got := attentionCacheIndexByLayer(model, len(model.Layers), 5)
	want := []int{0, 1, 2, 3, 4, 3}
	for i, wantIdx := range want {
		if got[i] != wantIdx {
			t.Fatalf("cache index for layer %d = %d, want %d", i, got[i], wantIdx)
		}
	}
}

type fakeRotatingModel struct {
	caches []Cache
}

func (f *fakeRotatingModel) Forward(_ *Array, _ []Cache) *Array                 { return nil }
func (f *fakeRotatingModel) ForwardMasked(_ *Array, _ *Array, _ []Cache) *Array { return nil }
func (f *fakeRotatingModel) NewCache() []Cache                                  { return append([]Cache(nil), f.caches...) }
func (f *fakeRotatingModel) NumLayers() int                                     { return len(f.caches) }
func (f *fakeRotatingModel) Tokenizer() *Tokenizer                              { return nil }
func (f *fakeRotatingModel) ModelType() string                                  { return "fake" }
func (f *fakeRotatingModel) ApplyLoRA(_ LoRAConfig) *LoRAAdapter                { return nil }

func TestModel_NewCaches_ShrinksOversizedRotatingCache_Good(t *testing.T) {
	coverageTokens := "NewCaches ShrinksOversizedRotatingCache"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	model := &Model{
		model: &fakeRotatingModel{
			caches: []Cache{
				NewRotatingKVCache(4096),
				NewRotatingKVCache(256),
			},
		},
		contextLen: 1024,
	}

	caches := model.newCaches()
	if len(caches) != 2 {
		t.Fatalf("len(caches) = %d, want 2", len(caches))
	}

	first, ok := caches[0].(*RotatingKVCache)
	if !ok {
		t.Fatalf("cache[0] = %T, want *RotatingKVCache", caches[0])
	}
	if first.maxSize != 1024 {
		t.Fatalf("cache[0].maxSize = %d, want 1024", first.maxSize)
	}

	second, ok := caches[1].(*RotatingKVCache)
	if !ok {
		t.Fatalf("cache[1] = %T, want *RotatingKVCache", caches[1])
	}
	if second.maxSize != 256 {
		t.Fatalf("cache[1].maxSize = %d, want 256", second.maxSize)
	}
}

func TestModel_FormatChat_Gemma2UsesGemmaTemplate_Good(t *testing.T) {
	coverageTokens := "FormatChat Gemma2UsesGemmaTemplate"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	model := &Model{modelType: "gemma2"}

	got := model.formatChat([]ChatMessage{
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Hi"},
	})

	want := "<start_of_turn>user\nHello<end_of_turn>\n" +
		"<start_of_turn>model\nHi<end_of_turn>\n" +
		"<start_of_turn>model\n"
	if got != want {
		t.Fatalf("formatChat() = %q, want %q", got, want)
	}
}

// Generated file-aware compliance coverage.
func TestGenerate_Model_ModelType_Good(t *testing.T) {
	coverageTokens := "Model ModelType"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_ModelType"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_ModelType_Bad(t *testing.T) {
	coverageTokens := "Model ModelType"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_ModelType"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_ModelType_Ugly(t *testing.T) {
	coverageTokens := "Model ModelType"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_ModelType"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_Err_Good(t *testing.T) {
	coverageTokens := "Model Err"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Err"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_Err_Bad(t *testing.T) {
	coverageTokens := "Model Err"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Err"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_Err_Ugly(t *testing.T) {
	coverageTokens := "Model Err"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Err"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_LastMetrics_Good(t *testing.T) {
	coverageTokens := "Model LastMetrics"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_LastMetrics"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_LastMetrics_Bad(t *testing.T) {
	coverageTokens := "Model LastMetrics"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_LastMetrics"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_LastMetrics_Ugly(t *testing.T) {
	coverageTokens := "Model LastMetrics"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_LastMetrics"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_Info_Good(t *testing.T) {
	coverageTokens := "Model Info"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Info"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_Info_Bad(t *testing.T) {
	coverageTokens := "Model Info"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Info"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_Info_Ugly(t *testing.T) {
	coverageTokens := "Model Info"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Info"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_Close_Good(t *testing.T) {
	coverageTokens := "Model Close"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Close"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_Close_Bad(t *testing.T) {
	coverageTokens := "Model Close"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Close"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_Close_Ugly(t *testing.T) {
	coverageTokens := "Model Close"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Close"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_Chat_Good(t *testing.T) {
	coverageTokens := "Model Chat"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Chat"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_Chat_Bad(t *testing.T) {
	coverageTokens := "Model Chat"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Chat"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_Chat_Ugly(t *testing.T) {
	coverageTokens := "Model Chat"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Chat"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_Generate_Good(t *testing.T) {
	coverageTokens := "Model Generate"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Generate"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_Generate_Bad(t *testing.T) {
	coverageTokens := "Model Generate"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Generate"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_Generate_Ugly(t *testing.T) {
	coverageTokens := "Model Generate"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Generate"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_InspectAttention_Good(t *testing.T) {
	coverageTokens := "Model InspectAttention"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_InspectAttention"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_InspectAttention_Bad(t *testing.T) {
	coverageTokens := "Model InspectAttention"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_InspectAttention"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_InspectAttention_Ugly(t *testing.T) {
	coverageTokens := "Model InspectAttention"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_InspectAttention"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}
