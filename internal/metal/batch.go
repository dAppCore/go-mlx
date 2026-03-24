//go:build darwin && arm64

package metal

import (
	"cmp"
	"context"
	"fmt"
	"math"
	"slices"
	"time"

	coreerr "forge.lthn.ai/core/go-log"
)

// ClassifyResult holds the output for a single prompt in batch classification.
type ClassifyResult struct {
	Token  Token
	Logits []float32
}

// BatchResult holds the output for a single prompt in batch generation.
type BatchResult struct {
	Tokens []Token
	Err    error
}

// Classify runs batched prefill-only inference. Each prompt gets a single
// forward pass and the token at the last position is sampled.
func (m *Model) Classify(ctx context.Context, prompts []string, cfg GenerateConfig, returnLogits bool) ([]ClassifyResult, error) {
	m.lastMetrics = Metrics{}
	if len(prompts) == 0 {
		return nil, nil
	}

	totalStart := time.Now()
	ResetPeakMemory()

	// Tokenise all prompts.
	encoded := make([][]int32, len(prompts))
	lengths := make([]int, len(prompts))
	totalPromptTokens := 0
	for i, p := range prompts {
		encoded[i] = m.tokenizer.Encode(p)
		lengths[i] = len(encoded[i])
		totalPromptTokens += lengths[i]
	}

	// Sort by length descending for minimal padding. Track original indices.
	indices := make([]int, len(prompts))
	for i := range indices {
		indices[i] = i
	}
	slices.SortFunc(indices, func(a, b int) int {
		return cmp.Compare(lengths[b], lengths[a])
	})

	maxLen := lengths[indices[0]]
	N := int32(len(prompts))
	L := int32(maxLen)

	// Build padded token matrix [N, maxLen] and prompt lengths (sorted order).
	padded := make([]int32, int(N)*int(L))
	sortedLengths := make([]int32, N)
	for si, origIdx := range indices {
		sortedLengths[si] = int32(lengths[origIdx])
		copy(padded[si*int(L):], encoded[origIdx])
		// Remaining positions are already 0 (pad token)
	}

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	// Build attention mask [N, 1, L, L].
	mask := buildBatchMask(N, L, sortedLengths)

	// Single forward pass.
	tokens := FromValues(padded, int(N), int(L))
	logits := m.model.ForwardMasked(tokens, mask, m.newCachesN(int(N)))
	if err := Eval(logits); err != nil {
		Free(tokens, mask)
		return nil, coreerr.E("Model.Classify", "classify prefill", err)
	}

	// logits shape: [N, L, vocab]
	sampler := newSampler(cfg.Temperature, cfg.TopP, 0, cfg.TopK)

	// Gather logits at each prompt's last real token position and sample.
	sortedResults := make([]ClassifyResult, N)
	for si := range N {
		lastPos := sortedLengths[si] - 1

		// Extract [1, vocab] at position lastPos for this batch element.
		batchLogits := SliceAxis(logits, 0, si, si+1) // [1, L, vocab]
		posLogits := SliceAxis(batchLogits, 1, lastPos, lastPos+1)
		posLogitsReshaped := Reshape(posLogits, 1, int32(posLogits.Dim(2))) // [1, vocab]

		next := sampler.Sample(posLogitsReshaped)
		if err := Eval(next); err != nil {
			Free(batchLogits, posLogits, posLogitsReshaped)
			return nil, coreerr.E("Model.Classify", fmt.Sprintf("classify sample %d", si), err)
		}

		id := int32(next.Int())
		text := m.tokenizer.DecodeToken(id)
		sortedResults[si].Token = Token{ID: id, Text: text}

		if returnLogits {
			logitsFlat := Reshape(posLogitsReshaped, int32(posLogitsReshaped.Dim(1)))
			if err := Eval(logitsFlat); err != nil {
				Free(batchLogits, posLogits, posLogitsReshaped, next, logitsFlat)
				return nil, coreerr.E("Model.Classify", fmt.Sprintf("classify logits %d", si), err)
			}
			sortedResults[si].Logits = logitsFlat.Floats()
			Free(logitsFlat)
		}
		Free(batchLogits, posLogits, posLogitsReshaped, next)
	}
	Free(logits, tokens, mask)

	// Unsort results back to original prompt order.
	results := make([]ClassifyResult, N)
	for si, origIdx := range indices {
		results[origIdx] = sortedResults[si]
	}

	totalDur := time.Since(totalStart)
	m.lastMetrics = Metrics{
		PromptTokens:      totalPromptTokens,
		GeneratedTokens:   int(N), // One token sampled per prompt
		PrefillDuration:   totalDur,
		TotalDuration:     totalDur,
		PeakMemoryBytes:   GetPeakMemory(),
		ActiveMemoryBytes: GetActiveMemory(),
	}
	if totalDur > 0 {
		m.lastMetrics.PrefillTokensPerSec = float64(totalPromptTokens) / totalDur.Seconds()
	}

	return results, nil
}

// BatchGenerate runs batched autoregressive generation.
func (m *Model) BatchGenerate(ctx context.Context, prompts []string, cfg GenerateConfig) ([]BatchResult, error) {
	m.lastMetrics = Metrics{}
	if len(prompts) == 0 {
		return nil, nil
	}

	totalStart := time.Now()
	ResetPeakMemory()

	// Tokenise all prompts.
	encoded := make([][]int32, len(prompts))
	lengths := make([]int, len(prompts))
	totalPromptTokens := 0
	for i, p := range prompts {
		encoded[i] = m.tokenizer.Encode(p)
		lengths[i] = len(encoded[i])
		totalPromptTokens += lengths[i]
	}

	// Sort by length descending.
	indices := make([]int, len(prompts))
	for i := range indices {
		indices[i] = i
	}
	slices.SortFunc(indices, func(a, b int) int {
		return cmp.Compare(lengths[b], lengths[a])
	})

	N := int32(len(prompts))
	maxLen := lengths[indices[0]]
	L := int32(maxLen)

	// Build padded token matrix and lengths.
	padded := make([]int32, int(N)*int(L))
	sortedLengths := make([]int32, N)
	for si, origIdx := range indices {
		sortedLengths[si] = int32(lengths[origIdx])
		copy(padded[si*int(L):], encoded[origIdx])
	}

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	// Prefill with mask.
	prefillStart := time.Now()
	mask := buildBatchMask(N, L, sortedLengths)
	tokens := FromValues(padded, int(N), int(L))
	caches := m.newCachesN(int(N))
	logits := m.model.ForwardMasked(tokens, mask, caches)
	if err := Eval(logits); err != nil {
		Free(tokens, mask)
		return nil, coreerr.E("Model.BatchGenerate", "batch prefill", err)
	}
	Free(tokens, mask) // No longer needed after prefill
	prefillDur := time.Since(prefillStart)

	sampler := newSampler(cfg.Temperature, cfg.TopP, 0, cfg.TopK)
	eosID := m.tokenizer.EOSToken()

	// Per-sequence state.
	type seqState struct {
		tokens   []Token
		finished bool
	}
	states := make([]seqState, N)

	maxTokens := cfg.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 256
	}

	for step := range maxTokens {
		select {
		case <-ctx.Done():
			// Return partial results on cancellation.
			sortedResults := make([]BatchResult, N)
			for si := range states {
				sortedResults[si] = BatchResult{Tokens: states[si].tokens, Err: ctx.Err()}
			}
			results := make([]BatchResult, N)
			for si, origIdx := range indices {
				results[origIdx] = sortedResults[si]
			}
			return results, nil
		default:
		}

		// Sample next token for each sequence from last position logits.
		nextIDs := make([]int32, N)
		allFinished := true

		for si := range N {
			if states[si].finished {
				nextIDs[si] = 0 // pad
				continue
			}

			var posLogits *Array
			var batchL, posL *Array
			if step == 0 {
				// First step: gather from prefill at each prompt's last real position.
				lastPos := sortedLengths[si] - 1
				batchL = SliceAxis(logits, 0, si, si+1)
				posL = SliceAxis(batchL, 1, lastPos, lastPos+1)
			} else {
				// Subsequent steps: logits shape [N, 1, vocab].
				batchL = SliceAxis(logits, 0, si, si+1)
				posL = SliceAxis(batchL, 1, 0, 1)
			}
			posLogits = Reshape(posL, 1, int32(posL.Dim(posL.NumDims()-1)))

			next := sampler.Sample(posLogits)
			if err := Eval(next); err != nil {
				Free(batchL, posL, posLogits, next)
				return nil, coreerr.E("Model.BatchGenerate", fmt.Sprintf("batch sample step %d seq %d", step, si), err)
			}

			id := int32(next.Int())
			nextIDs[si] = id

			// Check stop conditions.
			if id == eosID {
				states[si].finished = true
			} else if slices.Contains(cfg.StopTokens, id) {
				states[si].finished = true
			}
			if !states[si].finished {
				text := m.tokenizer.DecodeToken(id)
				states[si].tokens = append(states[si].tokens, Token{ID: id, Text: text})
				allFinished = false
			}
			Free(batchL, posL, posLogits, next)
		}

		if allFinished {
			break
		}

		// Feed next tokens [N, 1] through the model.
		nextInput := FromValues(nextIDs, int(N), 1)
		oldLogits := logits
		logits = m.model.Forward(nextInput, caches)
		if err := Eval(logits); err != nil {
			Free(nextInput, oldLogits)
			return nil, coreerr.E("Model.BatchGenerate", fmt.Sprintf("batch decode step %d", step), err)
		}
		Free(nextInput, oldLogits)
	}
	Free(logits)

	// Unsort results back to original order.
	sortedResults := make([]BatchResult, N)
	totalGenerated := 0
	for si := range states {
		sortedResults[si] = BatchResult{Tokens: states[si].tokens}
		totalGenerated += len(states[si].tokens)
	}
	results := make([]BatchResult, N)
	for si, origIdx := range indices {
		results[origIdx] = sortedResults[si]
	}

	totalDur := time.Since(totalStart)
	decodeDur := totalDur - prefillDur
	m.lastMetrics = Metrics{
		PromptTokens:      totalPromptTokens,
		GeneratedTokens:   totalGenerated,
		PrefillDuration:   prefillDur,
		DecodeDuration:    decodeDur,
		TotalDuration:     totalDur,
		PeakMemoryBytes:   GetPeakMemory(),
		ActiveMemoryBytes: GetActiveMemory(),
	}
	if prefillDur > 0 {
		m.lastMetrics.PrefillTokensPerSec = float64(totalPromptTokens) / prefillDur.Seconds()
	}
	if decodeDur > 0 {
		m.lastMetrics.DecodeTokensPerSec = float64(totalGenerated) / decodeDur.Seconds()
	}

	return results, nil
}

// buildBatchMask constructs a combined causal + padding attention mask.
// Shape: [N, 1, L, L]. Values: 0.0 = attend, -inf = ignore.
// mask[b, 0, i, j] = 0.0 if j <= i AND j < promptLen[b], else -inf.
func buildBatchMask(N, L int32, promptLens []int32) *Array {
	negInf := float32(math.Inf(-1))
	data := make([]float32, int(N)*int(L)*int(L))

	for b := range N {
		pLen := promptLens[b]
		base := int(b) * int(L) * int(L)
		for i := range L {
			for j := range L {
				if j <= i && j < pLen {
					data[base+int(i)*int(L)+int(j)] = 0
				} else {
					data[base+int(i)*int(L)+int(j)] = negInf
				}
			}
		}
	}

	mask := FromValues(data, int(N), 1, int(L), int(L))
	return mask
}

// newCachesN creates N independent sets of per-layer caches for batched inference.
// Since our KV cache implementation handles batch dimension internally,
// we create a single set of caches (same as non-batched).
func (m *Model) newCachesN(n int) []Cache {
	// KV caches handle the batch dimension automatically via the key/value
	// array shapes. A single cache set works for any batch size.
	return m.newCaches()
}
