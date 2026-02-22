//go:build darwin && arm64

package metal

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

// gemma3Path returns the path to a Gemma3-1B model, or skips the test.
func gemma3Path(t *testing.T) string {
	t.Helper()
	paths := []string{
		"/Volumes/Data/lem/gemma-3-1b-it-base",
		"/Volumes/Data/lem/safetensors/gemma-3/",
	}
	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	t.Skip("no Gemma3 model available")
	return ""
}

// TestLoRA_EndToEnd validates the full LoRA training pipeline:
// load base model → apply LoRA → train on small data → save adapter → reload.
func TestLoRA_EndToEnd(t *testing.T) {
	modelPath := gemma3Path(t)

	// Step 1: Load base model.
	model, err := loadModel(modelPath)
	if err != nil {
		t.Fatalf("loadModel: %v", err)
	}

	gemma := model.(*GemmaModel)
	tok := gemma.Tokenizer()

	// Step 2: Apply LoRA to Q and V projections.
	cfg := DefaultLoRAConfig() // rank=8, alpha=16, targets=[q_proj, v_proj]
	adapter := gemma.ApplyLoRA(cfg)

	numParams := adapter.TotalParams()
	t.Logf("LoRA applied: %d trainable parameters across %d layers",
		numParams, len(adapter.Layers))

	if numParams == 0 {
		t.Fatal("no trainable parameters")
	}

	// Step 3: Create a tiny training example.
	// Encode a short sequence, use shifted tokens as targets.
	inputIDs := tok.Encode("The capital of France is Paris")
	if len(inputIDs) < 3 {
		t.Fatalf("encoded too few tokens: %d", len(inputIDs))
	}
	t.Logf("Training tokens: %v (len=%d)", inputIDs, len(inputIDs))

	seqLen := len(inputIDs) - 1 // input is all but last, target is all but first
	inputTokens := FromValues(inputIDs[:seqLen], 1, seqLen)
	targetTokens := FromValues(inputIDs[1:], 1, seqLen)
	Materialize(inputTokens, targetTokens)

	// Step 4: Run a few training steps.
	params := adapter.AllTrainableParams()
	argnums := make([]int, len(params))
	for i := range argnums {
		argnums[i] = i
	}

	opt := NewAdamW(1e-4)
	var initialLoss, finalLoss float64
	const numSteps = 5

	for step := range numSteps {
		// Fresh caches each step (stateful — can't reuse across gradient calls).
		caches := gemma.NewCache()

		lossFn := func(inputs []*Array) []*Array {
			adapter.SetAllParams(inputs)
			logits := gemma.Forward(inputTokens, caches)
			// logits is [1, seqLen, vocab] — compute cross-entropy against targets
			loss := CrossEntropyLoss(logits, targetTokens)
			return []*Array{loss}
		}

		grad := ValueAndGrad(lossFn, argnums...)
		values, grads, err := grad.Apply(params...)
		grad.Free()
		if err != nil {
			t.Fatalf("step %d: ValueAndGrad failed: %v", step, err)
		}

		Materialize(append(values, grads...)...)

		loss := values[0].Float()
		t.Logf("step %d: loss = %.4f", step, loss)

		if step == 0 {
			initialLoss = loss
			if math.IsNaN(loss) || math.IsInf(loss, 0) {
				t.Fatalf("initial loss is %f", loss)
			}
		}
		finalLoss = loss

		// Update params.
		updated := opt.Step(params, grads)
		for i := range updated {
			Materialize(updated[i])
		}
		params = updated
		adapter.SetAllParams(params)
	}

	// Verify loss decreased.
	t.Logf("loss: %.4f → %.4f", initialLoss, finalLoss)
	if finalLoss >= initialLoss {
		t.Errorf("loss did not decrease: %.4f → %.4f", initialLoss, finalLoss)
	}

	// Step 5: Save adapter.
	savePath := filepath.Join(t.TempDir(), "adapter.safetensors")
	if err := adapter.Save(savePath); err != nil {
		t.Fatalf("adapter.Save: %v", err)
	}

	info, err := os.Stat(savePath)
	if err != nil {
		t.Fatalf("saved adapter not found: %v", err)
	}
	t.Logf("adapter saved: %s (%d bytes)", savePath, info.Size())

	// Step 6: Reload and verify weights match.
	loaded, err := LoadAllSafetensors(savePath)
	if err != nil {
		t.Fatalf("LoadAllSafetensors: %v", err)
	}

	for _, name := range adapter.SortedNames() {
		layer := adapter.Layers[name]
		aKey := name + ".lora_a"
		bKey := name + ".lora_b"

		loadedA, ok := loaded[aKey]
		if !ok {
			t.Errorf("missing %s in saved adapter", aKey)
			continue
		}
		loadedB, ok := loaded[bKey]
		if !ok {
			t.Errorf("missing %s in saved adapter", bKey)
			continue
		}

		Materialize(loadedA, loadedB)

		// Compare A weights.
		origA := layer.A.Floats()
		reloadA := loadedA.Floats()
		if len(origA) != len(reloadA) {
			t.Errorf("%s: A size mismatch: %d vs %d", name, len(origA), len(reloadA))
			continue
		}
		for i := range origA {
			if math.Abs(float64(origA[i]-reloadA[i])) > 1e-6 {
				t.Errorf("%s: A[%d] = %f, reloaded = %f", name, i, origA[i], reloadA[i])
				break
			}
		}

		// Compare B weights.
		origB := layer.B.Floats()
		reloadB := loadedB.Floats()
		if len(origB) != len(reloadB) {
			t.Errorf("%s: B size mismatch: %d vs %d", name, len(origB), len(reloadB))
			continue
		}
		for i := range origB {
			if math.Abs(float64(origB[i]-reloadB[i])) > 1e-6 {
				t.Errorf("%s: B[%d] = %f, reloaded = %f", name, i, origB[i], reloadB[i])
				break
			}
		}
	}

	t.Logf("all %d adapter layers verified after reload", len(adapter.Layers))

	ClearCache()
}

// TestLoRA_GradientCheckpointing validates that wrapping the forward pass in
// Checkpoint produces correct gradients (same loss decrease as non-checkpointed).
func TestLoRA_GradientCheckpointing(t *testing.T) {
	modelPath := gemma3Path(t)

	model, err := loadModel(modelPath)
	if err != nil {
		t.Fatalf("loadModel: %v", err)
	}

	gemma := model.(*GemmaModel)
	tok := gemma.Tokenizer()

	adapter := gemma.ApplyLoRA(DefaultLoRAConfig())
	t.Logf("LoRA: %d trainable params", adapter.TotalParams())

	inputIDs := tok.Encode("The capital of France is Paris")
	seqLen := len(inputIDs) - 1
	inputTokens := FromValues(inputIDs[:seqLen], 1, seqLen)
	targetTokens := FromValues(inputIDs[1:], 1, seqLen)
	Materialize(inputTokens, targetTokens)

	params := adapter.AllTrainableParams()
	argnums := make([]int, len(params))
	for i := range argnums {
		argnums[i] = i
	}

	opt := NewAdamW(1e-4)
	var initialLoss, finalLoss float64
	const numSteps = 3

	for step := range numSteps {
		caches := gemma.NewCache()

		// Wrap the model forward pass in Checkpoint to recompute activations
		// during backward instead of storing them.
		checkpointedForward := Checkpoint(func(inputs []*Array) []*Array {
			adapter.SetAllParams(inputs)
			logits := gemma.Forward(inputTokens, caches)
			return []*Array{logits}
		})

		lossFn := func(inputs []*Array) []*Array {
			logits := checkpointedForward(inputs)[0]
			loss := CrossEntropyLoss(logits, targetTokens)
			return []*Array{loss}
		}

		grad := ValueAndGrad(lossFn, argnums...)
		values, grads, err := grad.Apply(params...)
		grad.Free()
		if err != nil {
			t.Fatalf("step %d: ValueAndGrad failed: %v", step, err)
		}

		Materialize(append(values, grads...)...)

		loss := values[0].Float()
		t.Logf("step %d: loss = %.4f (checkpointed)", step, loss)

		if step == 0 {
			initialLoss = loss
			if math.IsNaN(loss) || math.IsInf(loss, 0) {
				t.Fatalf("initial loss is %f", loss)
			}
		}
		finalLoss = loss

		updated := opt.Step(params, grads)
		for i := range updated {
			Materialize(updated[i])
		}
		params = updated
		adapter.SetAllParams(params)
	}

	t.Logf("checkpointed loss: %.4f → %.4f", initialLoss, finalLoss)
	if finalLoss >= initialLoss {
		t.Errorf("loss did not decrease with checkpointing: %.4f → %.4f", initialLoss, finalLoss)
	}

	ClearCache()
}

// TestLoRA_MixedPrecision validates training with BFloat16 LoRA parameters.
// The base model stays in its native dtype; LoRA A/B are BFloat16.
// MLX auto-promotes for cross-dtype operations.
func TestLoRA_MixedPrecision(t *testing.T) {
	modelPath := gemma3Path(t)

	model, err := loadModel(modelPath)
	if err != nil {
		t.Fatalf("loadModel: %v", err)
	}

	gemma := model.(*GemmaModel)
	tok := gemma.Tokenizer()

	// Apply LoRA with BFloat16 parameters.
	cfg := DefaultLoRAConfig()
	cfg.DType = DTypeBFloat16
	adapter := gemma.ApplyLoRA(cfg)

	// Verify A/B are actually BFloat16.
	for name, layer := range adapter.Layers {
		if layer.A.Dtype() != DTypeBFloat16 {
			t.Errorf("%s: A dtype = %v, want bfloat16", name, layer.A.Dtype())
		}
		if layer.B.Dtype() != DTypeBFloat16 {
			t.Errorf("%s: B dtype = %v, want bfloat16", name, layer.B.Dtype())
		}
		break // just check first layer
	}

	t.Logf("LoRA BFloat16: %d trainable params (half memory vs Float32)",
		adapter.TotalParams())

	inputIDs := tok.Encode("The capital of France is Paris")
	seqLen := len(inputIDs) - 1
	inputTokens := FromValues(inputIDs[:seqLen], 1, seqLen)
	targetTokens := FromValues(inputIDs[1:], 1, seqLen)
	Materialize(inputTokens, targetTokens)

	params := adapter.AllTrainableParams()
	argnums := make([]int, len(params))
	for i := range argnums {
		argnums[i] = i
	}

	opt := NewAdamW(1e-4)
	var initialLoss, finalLoss float64
	const numSteps = 5

	for step := range numSteps {
		caches := gemma.NewCache()

		lossFn := func(inputs []*Array) []*Array {
			adapter.SetAllParams(inputs)
			logits := gemma.Forward(inputTokens, caches)
			loss := CrossEntropyLoss(logits, targetTokens)
			return []*Array{loss}
		}

		grad := ValueAndGrad(lossFn, argnums...)
		values, grads, err := grad.Apply(params...)
		grad.Free()
		if err != nil {
			t.Fatalf("step %d: ValueAndGrad failed: %v", step, err)
		}

		Materialize(append(values, grads...)...)

		loss := values[0].Float()
		t.Logf("step %d: loss = %.4f (bf16)", step, loss)

		if step == 0 {
			initialLoss = loss
			if math.IsNaN(loss) || math.IsInf(loss, 0) {
				t.Fatalf("initial loss is %f — bf16 may have caused NaN", loss)
			}
		}
		finalLoss = loss

		updated := opt.Step(params, grads)
		for i := range updated {
			Materialize(updated[i])
		}
		params = updated
		adapter.SetAllParams(params)
	}

	t.Logf("bf16 loss: %.4f → %.4f", initialLoss, finalLoss)
	if finalLoss >= initialLoss {
		t.Errorf("loss did not decrease with bf16: %.4f → %.4f", initialLoss, finalLoss)
	}

	ClearCache()
}
