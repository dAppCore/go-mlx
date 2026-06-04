// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

func TestGemma4AssistantDecode_DraftStep_Good(t *testing.T) {
	requireMetalRuntime(t)

	pair := loadTinyGemma4AssistantPair(t, false)
	defer pair.Close()

	caches := pair.Target.NewCache()
	defer metal.FreeCaches(caches)
	prefill := metal.FromValues([]int32{1, 2, 3}, 3)
	prefillInput := metal.Reshape(prefill, 1, 3)
	prefillLogits, previousHidden := pair.Target.ForwardLastTokenLogitsAndHidden(prefillInput, nil, caches)
	if err := metal.Eval(prefillLogits, previousHidden); err != nil {
		t.Fatalf("target prefill: %v", err)
	}
	metal.Free(prefill, prefillInput, prefillLogits)
	metal.DetachCaches(caches)
	defer metal.Free(previousHidden)
	result, err := pair.DraftStep(3, previousHidden, caches)
	if err != nil {
		t.Fatalf("DraftStep: %v", err)
	}
	defer result.Close()
	if err := metal.Eval(result.Logits, result.Token, result.Hidden); err != nil {
		t.Fatalf("Eval DraftStep result: %v", err)
	}
	assertShape(t, "logits", result.Logits, []int32{1, 1, 10})
	assertShape(t, "token", result.Token, []int32{1, 1})
	assertShape(t, "hidden", result.Hidden, []int32{1, 1, 8})
}

func TestGemma4AssistantDecode_DraftBlock_Good(t *testing.T) {
	requireMetalRuntime(t)

	pair := loadTinyGemma4AssistantPair(t, false)
	defer pair.Close()

	caches := pair.Target.NewCache()
	defer metal.FreeCaches(caches)
	prefill := metal.FromValues([]int32{1, 2, 3}, 3)
	prefillInput := metal.Reshape(prefill, 1, 3)
	prefillLogits, previousHidden := pair.Target.ForwardLastTokenLogitsAndHidden(prefillInput, nil, caches)
	if err := metal.Eval(prefillLogits, previousHidden); err != nil {
		t.Fatalf("target prefill: %v", err)
	}
	metal.Free(prefill, prefillInput, prefillLogits)
	metal.DetachCaches(caches)
	defer metal.Free(previousHidden)

	block, err := pair.DraftBlock(3, previousHidden, caches, 2)
	if err != nil {
		t.Fatalf("DraftBlock: %v", err)
	}
	defer block.Close()
	if len(block.Tokens) != 2 {
		t.Fatalf("DraftBlock tokens = %v, want 2 tokens", block.Tokens)
	}
	assertShape(t, "block hidden", block.Hidden, []int32{1, 1, 8})
}

func TestGemma4AssistantDecode_VerifyDraftBlock_Good(t *testing.T) {
	requireMetalRuntime(t)

	pair := loadTinyGemma4AssistantPair(t, false)
	defer pair.Close()
	caches := pair.Target.NewCache()
	defer metal.FreeCaches(caches)
	prefillLogits, previousHidden := prefillTinyGemma4AssistantTarget(t, pair, caches, []int32{1, 2, 3})
	defer metal.Free(prefillLogits, previousHidden)
	offsets := gemma4AssistantCacheOffsets(caches)
	targetToken, err := gemma4AssistantGreedyToken(prefillLogits)
	if err != nil {
		t.Fatalf("metal.Greedy target token: %v", err)
	}

	result, err := pair.VerifyDraftBlock(prefillLogits, []int32{targetToken}, caches)
	if err != nil {
		t.Fatalf("VerifyDraftBlock: %v", err)
	}
	defer result.Close()
	if !result.AllAccepted || result.AcceptedCount != 1 || result.RejectedCount != 0 {
		t.Fatalf("verify result = accepted %d rejected %d all %v", result.AcceptedCount, result.RejectedCount, result.AllAccepted)
	}
	if len(result.AcceptedTokens) != 1 || result.AcceptedTokens[0] != targetToken {
		t.Fatalf("accepted tokens = %v, want [%d]", result.AcceptedTokens, targetToken)
	}
	if result.ReplacementToken != 0 {
		t.Fatalf("replacement token = %d, want 0 on all-accepted path", result.ReplacementToken)
	}
	assertShape(t, "verify logits", result.Logits, []int32{1, 1, 10})
	assertShape(t, "verify hidden", result.Hidden, []int32{1, 1, 8})
	if got := gemma4AssistantCacheOffsets(caches); !gemma4AssistantIntSlicesEqual(got, offsets) {
		t.Fatalf("source cache offsets = %v, want unchanged %v", got, offsets)
	}
}

func TestGemma4AssistantDecode_VerifyDraftBlockRejectsBadToken_Good(t *testing.T) {
	requireMetalRuntime(t)

	pair := loadTinyGemma4AssistantPair(t, false)
	defer pair.Close()
	caches := pair.Target.NewCache()
	defer metal.FreeCaches(caches)
	prefillLogits, previousHidden := prefillTinyGemma4AssistantTarget(t, pair, caches, []int32{1, 2, 3})
	defer metal.Free(prefillLogits, previousHidden)
	targetToken, err := gemma4AssistantGreedyToken(prefillLogits)
	if err != nil {
		t.Fatalf("metal.Greedy target token: %v", err)
	}
	badToken := (targetToken + 1) % 10

	result, err := pair.VerifyDraftBlock(prefillLogits, []int32{badToken}, caches)
	if err != nil {
		t.Fatalf("VerifyDraftBlock: %v", err)
	}
	defer result.Close()
	if result.AllAccepted || result.AcceptedCount != 0 || result.RejectedCount != 1 {
		t.Fatalf("verify result = accepted %d rejected %d all %v", result.AcceptedCount, result.RejectedCount, result.AllAccepted)
	}
	if result.ReplacementToken != targetToken {
		t.Fatalf("replacement token = %d, want target token %d", result.ReplacementToken, targetToken)
	}
	if len(result.RejectedTokens) != 1 || result.RejectedTokens[0] != badToken {
		t.Fatalf("rejected tokens = %v, want [%d]", result.RejectedTokens, badToken)
	}
	assertShape(t, "reject logits", result.Logits, []int32{1, 1, 10})
	if result.Hidden != nil {
		t.Fatalf("reject hidden = %v, want nil before accepting any draft token", result.Hidden)
	}
}

func TestGemma4AssistantDecode_GreedyTokenSuppressesIDs_Good(t *testing.T) {
	requireMetalRuntime(t)

	logits := metal.FromValues([]float32{0.1, 9, 3, 2}, 1, 1, 4)
	defer metal.Free(logits)

	got, err := gemma4AssistantGreedyToken(logits, []int32{1})
	if err != nil {
		t.Fatalf("gemma4AssistantGreedyToken: %v", err)
	}
	if got != 2 {
		t.Fatalf("metal.Greedy token = %d, want unsuppressed token 2", got)
	}
}

func TestGemma4AssistantDecode_DraftStep_Bad(t *testing.T) {
	requireMetalRuntime(t)

	pair := loadTinyGemma4AssistantPair(t, false)
	defer pair.Close()
	previousHidden := seqArray(0.05, 1, 1, 8)
	defer metal.Free(previousHidden)
	_, err := pair.DraftStep(3, previousHidden, nil)
	if err == nil {
		t.Fatal("DraftStep() error = nil, want missing target caches")
	}
	if !core.Contains(err.Error(), "target caches") {
		t.Fatalf("DraftStep() error = %v, want target caches", err)
	}
}

func TestGemma4AssistantDecode_VerifyDraftBlock_Bad(t *testing.T) {
	pair := &Gemma4AssistantPair{}
	_, err := pair.VerifyDraftBlock(nil, []int32{1}, nil)
	if err == nil {
		t.Fatal("VerifyDraftBlock() error = nil, want target model error")
	}
	if !core.Contains(err.Error(), "target model") {
		t.Fatalf("VerifyDraftBlock() error = %v, want target model", err)
	}
}

func TestGemma4AssistantDecode_DraftBlock_Bad(t *testing.T) {
	pair := &Gemma4AssistantPair{}
	_, err := pair.DraftBlock(1, nil, nil, 0)
	if err == nil {
		t.Fatal("DraftBlock() error = nil, want maxDraftTokens error")
	}
	if !core.Contains(err.Error(), "maxDraftTokens") {
		t.Fatalf("DraftBlock() error = %v, want maxDraftTokens", err)
	}
}

func TestGemma4AssistantDecode_DraftStep_Ugly(t *testing.T) {
	requireMetalRuntime(t)

	pair := loadTinyGemma4AssistantPair(t, false)
	defer pair.Close()
	caches := pair.Target.NewCache()
	defer metal.FreeCaches(caches)
	prefill := metal.FromValues([]int32{1, 2}, 2)
	prefillInput := metal.Reshape(prefill, 1, 2)
	prefillLogits, previousHidden := pair.Target.ForwardLastTokenLogitsAndHidden(prefillInput, nil, caches)
	if err := metal.Eval(prefillLogits, previousHidden); err != nil {
		t.Fatalf("target prefill: %v", err)
	}
	metal.Free(prefill, prefillInput, prefillLogits, previousHidden)
	metal.DetachCaches(caches)

	wrongHidden := seqArray(0.05, 1, 1, 7)
	defer metal.Free(wrongHidden)
	_, err := pair.DraftStep(2, wrongHidden, caches)
	if err == nil {
		t.Fatal("DraftStep() error = nil, want hidden shape error")
	}
	if !core.Contains(err.Error(), "previous hidden shape") {
		t.Fatalf("DraftStep() error = %v, want previous hidden shape", err)
	}
}

func TestGemma4AssistantDecode_DraftStep_OrderedEmbeddingsGood(t *testing.T) {
	requireMetalRuntime(t)

	pair := loadTinyGemma4AssistantPair(t, true)
	defer pair.Close()
	caches := pair.Target.NewCache()
	defer metal.FreeCaches(caches)
	prefillLogits, previousHidden := prefillTinyGemma4AssistantTarget(t, pair, caches, []int32{1, 2, 3})
	defer metal.Free(prefillLogits, previousHidden)

	result, err := pair.DraftStep(3, previousHidden, caches)
	if err != nil {
		t.Fatalf("DraftStep() ordered embeddings: %v", err)
	}
	defer result.Close()
	if err := metal.Eval(result.Logits, result.Token, result.Hidden); err != nil {
		t.Fatalf("Eval ordered DraftStep result: %v", err)
	}
	assertShape(t, "ordered logits", result.Logits, []int32{1, 1, 10})
	assertShape(t, "ordered token", result.Token, []int32{1, 1})
	assertShape(t, "ordered hidden", result.Hidden, []int32{1, 1, 8})
	tokenValues := result.Token.DataInt32()
	if len(tokenValues) != 1 || tokenValues[0] < 0 || tokenValues[0] >= 10 {
		t.Fatalf("ordered token = %v, want one vocab token in [0,10)", tokenValues)
	}
}

func TestGemma4AssistantDecode_DraftStep_OrderedEmbeddingsBad(t *testing.T) {
	requireMetalRuntime(t)

	pair := loadTinyGemma4AssistantPair(t, true)
	defer pair.Close()
	metal.Free(pair.Assistant.TokenOrdering)
	pair.Assistant.TokenOrdering = metal.FromValues([]int32{0, 1, 2}, 3)
	caches := pair.Target.NewCache()
	defer metal.FreeCaches(caches)
	prefillLogits, previousHidden := prefillTinyGemma4AssistantTarget(t, pair, caches, []int32{1, 2, 3})
	defer metal.Free(prefillLogits, previousHidden)

	_, err := pair.DraftStep(3, previousHidden, caches)
	if err == nil {
		t.Fatal("DraftStep() error = nil, want token ordering layout error")
	}
	if !core.Contains(err.Error(), "token_ordering") {
		t.Fatalf("DraftStep() error = %v, want token_ordering", err)
	}
}

func TestGemma4AssistantDecode_LoadLocalAssistantPairDraftStep_Good(t *testing.T) {
	targetPath := core.Trim(core.Env("GO_MLX_GEMMA4_TARGET_MODEL"))
	assistantPath := core.Trim(core.Env("GO_MLX_GEMMA4_ASSISTANT_MODEL"))
	if targetPath == "" || assistantPath == "" {
		t.Skip("set GO_MLX_GEMMA4_TARGET_MODEL and GO_MLX_GEMMA4_ASSISTANT_MODEL to run the local draft-step smoke")
	}

	pair, err := LoadGemma4AssistantPair(targetPath, assistantPath)
	if err != nil {
		t.Fatalf("LoadGemma4AssistantPair(%s, %s): %v", targetPath, assistantPath, err)
	}
	defer pair.Close()

	caches := pair.Target.NewCache()
	defer metal.FreeCaches(caches)
	prefill := metal.FromValues([]int32{1, 2}, 2)
	prefillInput := metal.Reshape(prefill, 1, 2)
	prefillLogits, previousHidden := pair.Target.ForwardLastTokenLogitsAndHidden(prefillInput, nil, caches)
	if err := metal.Eval(prefillLogits, previousHidden); err != nil {
		t.Fatalf("target prefill: %v", err)
	}
	metal.Free(prefill, prefillInput)
	metal.DetachCaches(caches)

	defer metal.Free(prefillLogits, previousHidden)
	result, err := pair.DraftStep(2, previousHidden, caches)
	if err != nil {
		t.Fatalf("DraftStep(local): %v", err)
	}
	defer result.Close()
	if err := metal.Eval(result.Logits, result.Token, result.Hidden); err != nil {
		t.Fatalf("Eval local DraftStep result: %v", err)
	}
	assertShape(t, "local hidden", result.Hidden, []int32{1, 1, pair.Assistant.BackboneHiddenSize})

	targetToken, err := gemma4AssistantGreedyToken(prefillLogits)
	if err != nil {
		t.Fatalf("local metal.Greedy target token: %v", err)
	}
	verify, err := pair.VerifyDraftBlock(prefillLogits, []int32{targetToken}, caches)
	if err != nil {
		t.Fatalf("VerifyDraftBlock(local): %v", err)
	}
	defer verify.Close()
	if !verify.AllAccepted || verify.AcceptedCount != 1 {
		t.Fatalf("local verify accepted/all = %d/%v, want 1/true", verify.AcceptedCount, verify.AllAccepted)
	}
	assertShape(t, "local verify hidden", verify.Hidden, []int32{1, 1, pair.Assistant.BackboneHiddenSize})
}

func loadTinyGemma4AssistantPair(t testing.TB, ordered bool) *Gemma4AssistantPair {
	t.Helper()
	targetDir := t.TempDir()
	writeGemma4AssistantTargetConfig(t, targetDir)
	writeMinimalTokenizer(t, targetDir)
	if err := metal.SaveSafetensors(core.JoinPath(targetDir, "model.safetensors"), gemma4AssistantTargetTinyWeights()); err != nil {
		t.Fatalf("SaveSafetensors target: %v", err)
	}

	assistantDir := t.TempDir()
	writeGemma4AssistantConfig(t, assistantDir, ordered)
	writeMinimalTokenizer(t, assistantDir)
	if err := metal.SaveSafetensors(core.JoinPath(assistantDir, "model.safetensors"), gemma4AssistantTinyWeights(ordered)); err != nil {
		t.Fatalf("SaveSafetensors assistant: %v", err)
	}

	pair, err := LoadGemma4AssistantPair(targetDir, assistantDir)
	if err != nil {
		t.Fatalf("LoadGemma4AssistantPair: %v", err)
	}
	return pair
}

func prefillTinyGemma4AssistantTarget(t *testing.T, pair *Gemma4AssistantPair, caches []metal.Cache, tokens []int32) (*metal.Array, *metal.Array) {
	t.Helper()
	prefill := metal.FromValues(tokens, len(tokens))
	prefillInput := metal.Reshape(prefill, 1, int32(len(tokens)))
	prefillLogits, previousHidden := pair.Target.ForwardLastTokenLogitsAndHidden(prefillInput, nil, caches)
	if err := metal.Eval(prefillLogits, previousHidden); err != nil {
		metal.Free(prefill, prefillInput, prefillLogits, previousHidden)
		t.Fatalf("target prefill: %v", err)
	}
	metal.Free(prefill, prefillInput)
	metal.DetachCaches(caches)
	return prefillLogits, previousHidden
}

func gemma4AssistantCacheOffsets(caches []metal.Cache) []int {
	out := make([]int, len(caches))
	for i, cache := range caches {
		if cache != nil {
			out[i] = cache.Offset()
		}
	}
	return out
}

func gemma4AssistantIntSlicesEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func assertShape(t *testing.T, label string, array *metal.Array, want []int32) {
	t.Helper()
	if array == nil || !array.Valid() {
		t.Fatalf("%s array invalid", label)
	}
	got := array.Shape()
	if len(got) != len(want) {
		t.Fatalf("%s shape = %v, want %v", label, got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("%s shape = %v, want %v", label, got, want)
		}
	}
}
