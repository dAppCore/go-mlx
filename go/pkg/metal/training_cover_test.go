// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// training_cover_test.go covers the *Model training/adapter SDK surface in
// training.go the model-eval suites leave dark: adapter-info derivation (both the
// path and pathless hash arms), the nil-receiver guards on LoadLoRA/UnloadLoRA,
// the nil-adapter ApplyLoRA arm, the tokenizer pass-throughs, and the
// deviceInternalModel wrapper. fakeModel-backed — no weights.

func TestTraining_AdapterInfoFromLoRA_Branches_Good(t *testing.T) {
	if got := adapterInfoFromLoRA("", nil); got.Hash != "" || got.Name != "" || got.Rank != 0 {
		t.Fatalf("nil adapter → %+v, want zero AdapterInfo", got)
	}

	adapter := &LoRAAdapter{Config: LoRAConfig{
		Rank:       8,
		Alpha:      16,
		Scale:      2,
		TargetKeys: []string{"q_proj", "v_proj"},
	}}

	// Pathless form hashes only the numeric config (so an in-memory ApplyLoRA
	// and a saved-then-loaded adapter with the same geometry share a key). With
	// an empty path the derived Path stays empty (PathBase("") is "." for Name).
	pathless := adapterInfoFromLoRA("", adapter)
	if pathless.Path != "" {
		t.Errorf("pathless info carried a path: %+v", pathless)
	}
	if pathless.Rank != 8 || pathless.Hash == "" {
		t.Errorf("pathless info malformed: %+v", pathless)
	}

	// Path form folds the name/path into the hash, so it differs from pathless.
	withPath := adapterInfoFromLoRA("/tmp/adapters/my-lora", adapter)
	if withPath.Name != "my-lora" || withPath.Path != "/tmp/adapters/my-lora" {
		t.Errorf("path info name/path wrong: %+v", withPath)
	}
	if withPath.Hash == pathless.Hash {
		t.Error("path and pathless hashes collided; path must fold into the hash")
	}
}

func TestTraining_CloneMetalAdapterInfo_Good(t *testing.T) {
	src := AdapterInfo{Name: "a", TargetKeys: []string{"q_proj"}}
	clone := cloneMetalAdapterInfo(src)
	if clone.Name != "a" || len(clone.TargetKeys) != 1 {
		t.Fatalf("clone = %+v, want a copy of src", clone)
	}
	// Mutating the clone's slice must not touch the source — it is a fresh copy.
	clone.TargetKeys[0] = "v_proj"
	if src.TargetKeys[0] != "q_proj" {
		t.Error("cloneMetalAdapterInfo shared the TargetKeys backing array")
	}
}

func TestTraining_Adapter_NilAndEmpty_Good(t *testing.T) {
	if got := (*Model)(nil).Adapter(); got.Name != "" || got.Hash != "" || len(got.TargetKeys) != 0 {
		t.Errorf("nil model Adapter = %+v, want zero", got)
	}
	m := &Model{adapterInfo: AdapterInfo{Name: "x", TargetKeys: []string{"q_proj"}}}
	if got := m.Adapter(); got.Name != "x" {
		t.Errorf("Adapter = %+v, want name x", got)
	}
}

func TestTraining_LoadUnloadLoRA_NilGuards_Bad(t *testing.T) {
	// Nil inner model → descriptive errors.
	if _, err := (&Model{}).LoadLoRA("/tmp/x"); err == nil {
		t.Error("LoadLoRA on a model with no inner model returned nil error")
	}
	if err := (&Model{}).UnloadLoRA(); err == nil {
		t.Error("UnloadLoRA on a model with no inner model returned nil error")
	}
	// Inner model present but no active adapter → UnloadLoRA is a no-op success.
	m := &Model{model: &fakeModel{numLayers: 1}}
	if err := m.UnloadLoRA(); err != nil {
		t.Errorf("UnloadLoRA with no adapter returned error: %v", err)
	}
}

func TestTraining_ApplyLoRA_NilAdapterArm_Good(t *testing.T) {
	requireMetalRuntime(t)
	// fakeModel.ApplyLoRA returns nil, so the adapter-install block is skipped
	// and the method returns nil without mutating model state.
	m := &Model{model: &fakeModel{numLayers: 1}}
	if got := m.ApplyLoRA(DefaultLoRAConfig()); got != nil {
		t.Errorf("ApplyLoRA returned %v, want nil for the fake model", got)
	}
	if m.adapter != nil {
		t.Error("ApplyLoRA installed an adapter from a nil return")
	}
}

func TestTraining_TokenizerPassThroughs_Good(t *testing.T) {
	requireMetalRuntime(t)
	tok := NewForDecode(nil)
	m := &Model{model: &fakeModel{numLayers: 3}, tokenizer: tok}

	if m.Tokenizer() != tok {
		t.Error("Tokenizer() did not return the model tokenizer")
	}
	if m.NumLayers() != 3 {
		t.Errorf("NumLayers = %d, want 3", m.NumLayers())
	}
	// Encode/Decode round-trip through the tokenizer (empty/whitespace tolerated).
	ids := m.Encode("")
	_ = m.Decode(ids)
}

func TestTraining_DeviceInternalModelWrapper_Good(t *testing.T) {
	requireMetalRuntime(t)
	m := &Model{model: &fakeModel{numLayers: 2}}
	im := m.Internal()
	if im == nil {
		t.Fatal("Internal() returned nil")
	}
	// fakeModel returns nil from the forward methods; the wrapper just routes the
	// device-scoped call. Drive each entry point to cover the wrapper bodies.
	if im.NumLayers() != 2 {
		t.Errorf("wrapper NumLayers = %d, want 2", im.NumLayers())
	}
	if im.ModelType() != "fake" {
		t.Errorf("wrapper ModelType = %q, want fake", im.ModelType())
	}
	_ = im.Tokenizer()
	tokens := FromSingleInt32Matrix(1)
	defer Free(tokens)
	_ = im.Forward(tokens, nil)
	_ = im.ForwardMasked(tokens, nil, nil)
	_ = im.ApplyLoRA(DefaultLoRAConfig())
}
