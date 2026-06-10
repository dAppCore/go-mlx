// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"context"
	"slices"
	"time"

	core "dappco.re/go"
)

// Pipelined decode — one-ahead token generation. The serial loop reads token
// N on the host before building token N+1's forward, so the graph encode
// (~4ms) and the GPU compute (~2.6ms) alternate with each side idle while the
// other works. The pipelined loop builds the next forward against the LAZY
// sampled token array (the forward never needs the token's value, only the
// text emission does), submits it, and only then reads the token — the encode
// of N+1 overlaps the GPU still running N, and the read is nearly free
// because the sample op sits at the head of the submitted batch.
//
// Speculation is made safe by the functional compiled-layer cache path:
// every cache is armed before the build, so the layer adoptions STAGE the
// updated K/V (FixedKVCache pending-commit) instead of swapping them in. If
// the token read shows EOS or a stop token, the staged forward is discarded
// and the cache state is exactly what the serial loop would have left — the
// speculated token's K/V never existed. Otherwise the stage commits and the
// loop continues. At most one forward is ever in flight.
//
// The loop requires every layer on the functional path: all caches are
// FixedKVCache, whole-layer compiled decode is on, and (for wide-head global
// layers) the 512-wide SDPA gate is held for the generation scope. A
// non-functional cache update while armed (a layer fell off the compiled
// path mid-generation) flags a violation; the loop commits coherently and
// hands the remaining steps back to the serial loop.

// pipelinedSubmitLogitsOnly is an in-code diagnostic (off by default, NEVER
// ambient env): submit only the logits root and let the staged K/V evaluate
// as the next forward's dependencies. Probes whether the wide root set costs
// scheduling time.
var pipelinedSubmitLogitsOnly = false

// pipelinedDecodeState is the slice of generateLocked's local state the
// pipelined loop shares with the serial loop it may hand back to.
type pipelinedDecodeState struct {
	cfg         GenerateConfig
	sampler     Sampler
	yield       func(Token) bool
	genCount    *int
	firstToken  *time.Duration
	totalStart  time.Time
	tokenPhases *[]TokenPhaseTrace
}

// pipelinedDecodeEligibleLocked reports whether this generation can run the
// one-ahead loop. Host-value control flow inside a step (repeat penalty,
// suppression, probes, early-stop suppression windows) keeps the serial loop.
func (s *ModelSession) pipelinedDecodeEligibleLocked(cfg GenerateConfig) bool {
	if !PipelinedDecodeEnabled() || !CompiledLayerDecodeEnabled() {
		return false
	}
	if cfg.RepeatPenalty > 1.0 || len(cfg.SuppressTokens) > 0 || cfg.MinTokensBeforeStop > 0 || cfg.ProbeSink != nil {
		return false
	}
	if s.logits == nil || !s.logits.Valid() || len(s.caches) == 0 {
		return false
	}
	for _, cache := range s.caches {
		fixed, ok := cache.(*FixedKVCache)
		if !ok || fixed == nil || fixed.MaxSize() <= 0 {
			return false
		}
	}
	return true
}

func (s *ModelSession) armPendingCachesLocked() {
	for _, cache := range s.caches {
		if fixed, ok := cache.(*FixedKVCache); ok {
			fixed.ArmPending()
		}
	}
}

func (s *ModelSession) commitPendingCachesLocked() {
	for _, cache := range s.caches {
		if fixed, ok := cache.(*FixedKVCache); ok {
			fixed.CommitPending()
		}
	}
}

func (s *ModelSession) discardPendingCachesLocked() {
	for _, cache := range s.caches {
		if fixed, ok := cache.(*FixedKVCache); ok {
			fixed.DiscardPending()
		}
	}
}

func (s *ModelSession) pendingViolationLocked() bool {
	for _, cache := range s.caches {
		if fixed, ok := cache.(*FixedKVCache); ok && fixed.PendingViolated() {
			return true
		}
	}
	return false
}

// pipelineTokenInput shapes the lazy sampled token into the [1,1] int32 input
// the serial loop feeds the forward (FromSingleInt32Matrix), keeping the
// traced decode graph identical between modes.
func pipelineTokenInput(next *Array) *Array {
	cast := AsType(next, DTypeInt32)
	if cast.NumDims() == 2 && cast.Dim(0) == 1 && cast.Dim(1) == 1 {
		return cast
	}
	reshaped := Reshape(cast, 1, 1)
	Free(cast)
	return reshaped
}

// runPipelinedDecodeLocked runs the one-ahead loop. It returns the step the
// serial loop should resume from and whether the generation already finished
// (EOS, stop token, consumer stop, max tokens, or error).
func (s *ModelSession) runPipelinedDecodeLocked(ctx context.Context, st pipelinedDecodeState) (resume int, finished bool) {
	cfg := st.cfg
	// Wide-head global layers compile in every mode now: the pre-cap
	// attention step is composed in-trace over a fill-band slice, so the
	// wide-SDPA gate (which guards the capacity-wide native call) is not
	// needed for the pipelined scope.

	for i := 0; i < cfg.MaxTokens; i++ {
		tracePhases := cfg.TraceTokenPhases
		var phaseStart, phaseLast time.Time
		var phase TokenPhaseTrace
		if tracePhases {
			phaseStart = time.Now()
			phaseLast = phaseStart
			phase = TokenPhaseTrace{Step: i}
		}
		select {
		case <-ctx.Done():
			s.err = ctx.Err()
			return i, true
		default:
		}

		// Lazy next token — the same op the serial loop samples, unevaluated.
		var next *Array
		if nativeGreedyDecodeAvailable(cfg, nil, s.logits) {
			var err error
			next, err = nativeGreedyDecodeToken(s.logits)
			if err != nil {
				s.err = core.E("ModelSession.Generate", core.Sprintf("pipelined greedy decode step %d", i), err)
				return i, true
			}
		} else {
			lastPos, err := lastTokenLogits(s.logits)
			if err != nil {
				s.err = core.E("ModelSession.Generate", core.Sprintf("pipelined last logits step %d", i), err)
				return i, true
			}
			next = st.sampler.Sample(lastPos)
			Free(lastPos)
		}
		if tracePhases {
			phase.LogitsDuration = time.Since(phaseLast)
			phaseLast = time.Now()
		}

		// Submit the sample on its own — a separate (tiny) command buffer
		// whose completion event fires in microseconds. If the sample rode
		// in the forward's batch, reading the token would wait on the whole
		// forward's buffer and re-serialise the loop.
		if err := EvalAsync(next); err != nil {
			Free(next)
			s.err = core.E("ModelSession.Generate", core.Sprintf("pipelined sample submit step %d", i), err)
			return i, true
		}

		// Build the speculated forward against the lazy token; the armed
		// caches stage their adoptions instead of committing them.
		s.armPendingCachesLocked()
		input := pipelineTokenInput(next)
		nextLogits, _ := s.model.forwardLastTokenLogits(input, nil, s.caches)
		Free(input)
		if nextLogits == nil || !nextLogits.Valid() {
			s.discardPendingCachesLocked()
			Free(next, nextLogits)
			if err := LastError(); err != nil {
				s.err = core.E("ModelSession.Generate", core.Sprintf("pipelined decode step %d", i), err)
			} else {
				s.err = core.E("ModelSession.Generate", core.Sprintf("pipelined decode step %d", i), errForwardNilLogits)
			}
			return i, true
		}
		violated := s.pendingViolationLocked()
		if tracePhases {
			phase.ForwardDuration = time.Since(phaseLast)
			phaseLast = time.Now()
		}

		// Submit the speculated forward: the next logits plus the staged
		// K/V. The GPU picks the batch up while this iteration's host work
		// continues.
		var outStack [80]*Array
		outputs := append(outStack[:0], nextLogits)
		if !pipelinedSubmitLogitsOnly {
			for _, cache := range s.caches {
				if fixed, ok := cache.(*FixedKVCache); ok {
					outputs = fixed.AppendPendingState(outputs)
				}
			}
		}
		if err := EvalAsync(outputs...); err != nil {
			s.discardPendingCachesLocked()
			Free(next, nextLogits)
			s.err = core.E("ModelSession.Generate", core.Sprintf("pipelined submit step %d", i), err)
			return i, true
		}
		if tracePhases {
			phase.PrefetchDuration = time.Since(phaseLast)
			phase.PrefetchLogitsDuration = phase.PrefetchDuration
			phaseLast = time.Now()
		}

		// Read the token — the sample op sits at the head of the submitted
		// batch, so this returns as soon as the GPU clears it, while the
		// speculated forward keeps running behind it.
		id := int32(next.Int())
		if tracePhases {
			phase.TokenReadDuration = time.Since(phaseLast)
			phaseLast = time.Now()
		}

		// The previous step's state is materialised by now; cut its graph
		// edges exactly as the serial loop does post-eval.
		detachEvalState(s.logits, s.caches)
		if tracePhases {
			phase.DetachDuration = time.Since(phaseLast)
			phaseLast = time.Now()
		}

		text := s.model.tokenizer.DecodeToken(id)
		if tracePhases {
			phase.TokenID = id
			if cfg.TraceTokenText {
				phase.TokenText = text
			}
			phase.DecodeTextDuration = time.Since(phaseLast)
			phaseLast = time.Now()
		}

		stop := s.model.tokenizer.HasEOSToken() && id == s.model.tokenizer.EOSToken()
		stop = stop || slices.Contains(cfg.StopTokens, id)
		if stop {
			// The speculated forward was for the stop token — drop it; the
			// cache state is exactly the serial loop's (no forward ran for
			// the stop token).
			if violated {
				core.Error("mlx: pipelined decode stop token after a non-functional cache update; cache keeps the speculated forward",
					"step", i)
				s.commitPendingCachesLocked()
			} else {
				s.discardPendingCachesLocked()
			}
			Free(next, nextLogits)
			if tracePhases {
				phase.FinalToken = true
				*st.tokenPhases = appendTokenPhaseTrace(*st.tokenPhases, phase, phaseStart)
			}
			return i, true
		}

		// Commit the speculated forward: the staged K/V become the cache
		// state and the logits advance, matching what the serial loop's
		// advance would have produced.
		s.commitPendingCachesLocked()
		oldLogits := s.logits
		s.logits = nextLogits
		Free(oldLogits, next)
		s.tokens = append(s.tokens, id)
		s.generated = append(s.generated, id)
		s.tokenOffset++
		*st.genCount++
		if *st.firstToken == 0 {
			*st.firstToken = time.Since(st.totalStart)
		}

		if violated {
			// A layer fell off the functional path; its storage mutated at
			// build time, so speculation is no longer safe. Hand the rest of
			// the generation to the serial loop.
			core.Error("mlx: pipelined decode degrading to serial after a non-functional cache update", "step", i)
			if !st.yield(Token{ID: id, Text: text}) {
				if tracePhases {
					phase.FinalToken = true
					*st.tokenPhases = appendTokenPhaseTrace(*st.tokenPhases, phase, phaseStart)
				}
				return i, true
			}
			if tracePhases {
				*st.tokenPhases = appendTokenPhaseTrace(*st.tokenPhases, phase, phaseStart)
			}
			return i + 1, false
		}

		if !st.yield(Token{ID: id, Text: text}) {
			if tracePhases {
				phase.FinalToken = true
				*st.tokenPhases = appendTokenPhaseTrace(*st.tokenPhases, phase, phaseStart)
			}
			return i, true
		}
		if tracePhases {
			phase.YieldDuration = time.Since(phaseLast)
			*st.tokenPhases = appendTokenPhaseTrace(*st.tokenPhases, phase, phaseStart)
		}
	}
	return cfg.MaxTokens, true
}
