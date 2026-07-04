// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// Gemma4AssistantPair is a validated target plus attached MTP assistant. The
// assistant is not a standalone text model; it is only valid beside the target
// Gemma 4 runtime whose hidden state and K/V cache streams it borrows.
type Gemma4AssistantPair struct {
	Target    *Gemma4Model
	Assistant *Gemma4AssistantModel

	ownsTarget    bool
	ownsAssistant bool

	// validated records that validateGemma4AssistantPair has already passed for
	// this Target+Assistant. The attachment constraints (shapes, tokenizer
	// equivalence, required tensors) are structural and invariant across decode
	// steps, but the check builds per-layer key strings + re-encodes tokenizer
	// probes — a heavy per-call allocation. Every pair built through
	// attachGemma4AssistantModels is validated at construction, so the per-step
	// re-validation in the draft hot path is redundant. Cached on first success
	// (construction, or first decode of a hand-built pair) and skipped after.
	validated bool
}

// LoadGemma4AssistantPair loads a Gemma 4 target and its assistant drafter,
// then validates the runtime attachment constraints.
func LoadGemma4AssistantPair(targetPath, assistantPath string) (*Gemma4AssistantPair, error) {
	if core.Trim(targetPath) == "" {
		return nil, core.NewError("gemma4.assistant pair target path is required")
	}
	if core.Trim(assistantPath) == "" {
		return nil, core.NewError("gemma4.assistant pair assistant path is required")
	}

	target, err := loadGemma4TextModel(targetPath)
	if err != nil {
		return nil, core.E("gemma4.assistant.Pair", "load target", err)
	}
	assistant, err := LoadGemma4Assistant(assistantPath)
	if err != nil {
		closeGemma4(target)
		metal.ClearCache()
		return nil, core.E("gemma4.assistant.Pair", "load assistant", err)
	}
	pair, err := attachGemma4AssistantModels(target, assistant)
	if err != nil {
		closeGemma4(target)
		if closeErr := assistant.Close(); closeErr != nil {
			err = core.ErrorJoin(err, closeErr)
		}
		return nil, core.E("gemma4.assistant.Pair", "validate attachment", err)
	}
	pair.ownsTarget = true
	pair.ownsAssistant = true
	return pair, nil
}

// attachGemma4AssistantModels validates an already loaded target and assistant.
func attachGemma4AssistantModels(target *Gemma4Model, assistant *Gemma4AssistantModel) (*Gemma4AssistantPair, error) {
	if err := validateGemma4AssistantPair(target, assistant); err != nil {
		return nil, err
	}
	return &Gemma4AssistantPair{Target: target, Assistant: assistant, validated: true}, nil
}

// AttachGemma4Assistant loads the assistant drafter at draftPath and validates
// it against the Gemma 4 model already loaded into the target runtime. The
// returned pair drives speculative decoding via (*Gemma4AssistantPair).Generate.
//
//	pair, err := gemma4.AttachGemma4Assistant(targetModel, "path/to/drafter")
func AttachGemma4Assistant(target *metal.Model, draftPath string) (*Gemma4AssistantPair, error) {
	if target == nil {
		return nil, core.NewError("gemma4.assistant pair target model is nil")
	}
	model, ok := target.UnderlyingModel().(*Gemma4Model)
	if !ok {
		return nil, core.NewError("gemma4.assistant pair requires a Gemma 4 target")
	}
	var assistant *Gemma4AssistantModel
	var err error
	if file, isGGUF := resolveGGUFDrafterFile(draftPath); isGGUF {
		// The unsloth single-file drafter lane (#92): config from gguf
		// metadata, tensors renamed, tokenizer borrowed from the target.
		assistant, err = loadGemma4AssistantFromGGUF(file, model.Tok)
	} else {
		assistant, err = LoadGemma4Assistant(draftPath)
	}
	if err != nil {
		return nil, err
	}
	pair, err := attachGemma4AssistantModels(model, assistant)
	if err != nil {
		if closeErr := assistant.Close(); closeErr != nil {
			err = core.ErrorJoin(err, closeErr)
		}
		return nil, err
	}
	pair.ownsAssistant = true
	return pair, nil
}

// Close releases models owned by a pair returned from LoadGemma4AssistantPair.
func (pair *Gemma4AssistantPair) Close() error {
	if pair == nil {
		return nil
	}
	var err error
	if pair.ownsAssistant && pair.Assistant != nil {
		err = core.ErrorJoin(err, pair.Assistant.Close())
	}
	if pair.ownsTarget && pair.Target != nil {
		closeGemma4(pair.Target)
		metal.ClearCache()
	}
	pair.Target = nil
	pair.Assistant = nil
	return err
}

func validateGemma4AssistantPair(target *Gemma4Model, assistant *Gemma4AssistantModel) error {
	if target == nil || target.Cfg == nil {
		return core.NewError("gemma4.assistant pair target is nil")
	}
	if assistant == nil || assistant.Cfg == nil {
		return core.NewError("gemma4.assistant pair assistant is nil")
	}
	if target.Cfg.HiddenSize <= 0 {
		return core.NewError("gemma4.assistant pair target hidden_size is invalid")
	}
	if assistant.BackboneHiddenSize != target.Cfg.HiddenSize {
		return core.NewError(core.Sprintf("gemma4.assistant backbone_hidden_size = %d, want target hidden_size %d", assistant.BackboneHiddenSize, target.Cfg.HiddenSize))
	}
	if target.Cfg.VocabSize > 0 && assistant.Cfg.VocabSize > 0 && target.Cfg.VocabSize != assistant.Cfg.VocabSize {
		return core.NewError(core.Sprintf("gemma4.assistant vocab_size = %d, want target vocab_size %d", assistant.Cfg.VocabSize, target.Cfg.VocabSize))
	}
	if target.Tok == nil || assistant.Tok == nil {
		return core.NewError("gemma4.assistant pair requires target and assistant tokenizers")
	}
	if err := validateGemma4AssistantTokenizerProbe(target.Tok, assistant.Tok); err != nil {
		return err
	}
	if err := validateGemma4AssistantTargetTypes(target, assistant); err != nil {
		return err
	}
	if err := validateGemma4AssistantModel(assistant); err != nil {
		return err
	}
	return nil
}

func validateGemma4AssistantTokenizerProbe(target, assistant *metal.Tokenizer) error {
	probes := []string{"hello", "The quick brown fox", "Answer in one short sentence."}
	for _, probe := range probes {
		targetTokens := target.Encode(probe)
		assistantTokens := assistant.Encode(probe)
		if !gemma4AssistantInt32SlicesEqual(targetTokens, assistantTokens) {
			return core.NewError("gemma4.assistant target and assistant tokenizers differ")
		}
	}
	return nil
}

func validateGemma4AssistantTargetTypes(target *Gemma4Model, assistant *Gemma4AssistantModel) error {
	targetTypes := gemma4TargetLayerTypes(target)
	if len(targetTypes) == 0 {
		return core.NewError("gemma4.assistant pair target layer types are unavailable")
	}
	for idx, layer := range assistant.Layers {
		if layer == nil {
			return core.NewError(core.Sprintf("gemma4.assistant layer %d is nil", idx))
		}
		if !targetTypes[layer.LayerType] {
			return core.NewError(core.Sprintf("gemma4.assistant layer %d type %q has no target K/V stream", idx, layer.LayerType))
		}
		if layer.Attention == nil {
			continue
		}
		wantHeadDim := gemma4TargetHeadDimForLayerType(target.Cfg, layer.LayerType)
		if wantHeadDim > 0 && layer.Attention.HeadDim != wantHeadDim {
			return core.NewError(core.Sprintf("gemma4.assistant layer %d head_dim = %d, want target %s head_dim %d", idx, layer.Attention.HeadDim, layer.LayerType, wantHeadDim))
		}
	}
	return nil
}

func gemma4TargetLayerTypes(target *Gemma4Model) map[string]bool {
	out := make(map[string]bool)
	if target == nil || target.Cfg == nil {
		return out
	}
	for _, layerType := range target.Cfg.LayerTypes {
		if layerType != "" {
			out[layerType] = true
		}
	}
	for _, layer := range target.Layers {
		if layer != nil && layer.LayerType != "" {
			out[layer.LayerType] = true
		}
	}
	return out
}

func gemma4TargetHeadDimForLayerType(cfg *Gemma4TextConfig, layerType string) int32 {
	if cfg == nil {
		return 0
	}
	if layerType == "full_attention" && cfg.GlobalHeadDim > 0 {
		return cfg.GlobalHeadDim
	}
	return cfg.HeadDim
}

func gemma4AssistantInt32SlicesEqual(a, b []int32) bool {
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
