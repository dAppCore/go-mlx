// SPDX-Licence-Identifier: EUPL-1.2

package m2

import (
	"math"
	"slices"

	core "dappco.re/go"
	"dappco.re/go/inference/probe"
)

// RouteTokens computes deterministic top-k router decisions for a
// batch of router scores. Scores are sigmoid-normalised by default and top-k
// weights are renormalised, matching the MiniMax M2 sparse routing contract.
func RouteTokens(cfg Config, scores [][]float32, bias []float32) ([]RouterDecision, error) {
	if cfg.NumLocalExperts <= 0 {
		return nil, core.NewError("mlx: MiniMax M2 routing requires local expert count")
	}
	topK := cfg.NumExpertsPerToken
	if topK <= 0 {
		topK = 1
	}
	if topK > cfg.NumLocalExperts {
		return nil, core.NewError("mlx: MiniMax M2 routing top-k exceeds expert count")
	}
	if len(bias) > 0 && len(bias) != cfg.NumLocalExperts {
		return nil, core.NewError("mlx: MiniMax M2 routing bias length does not match expert count")
	}
	decisions := make([]RouterDecision, 0, len(scores))
	hasBias := len(bias) > 0
	scoreFn := scoringFunc(cfg.ScoringFunc)
	// Reuse one scored buffer across tokens — was alloc-per-token before,
	// which dominated RouteTokens at 256 experts × 32 tokens (~128 KiB churn
	// per call). Buffer is call-local so no concurrency risk.
	scored := make(expertScoreSlice, cfg.NumLocalExperts)
	// Single arena slab for all per-token ExpertIDs + Weights slices. Was
	// make([]int, topK) + make([]float32, topK) per token = 2N allocs;
	// now 2 allocs total. Third-index cap = topK keeps any future append
	// from running into the next token's slot (we don't append today, but
	// the bound is the cheap insurance that lets us share the backing).
	expertIDArena := make([]int, len(scores)*topK)
	weightArena := make([]float32, len(scores)*topK)
	for tokenIndex, row := range scores {
		if len(row) != cfg.NumLocalExperts {
			return nil, core.NewError(core.Sprintf("mlx: MiniMax M2 routing row %d has %d scores, expected %d", tokenIndex, len(row), cfg.NumLocalExperts))
		}
		if hasBias {
			for expertID, raw := range row {
				scored[expertID] = expertScore{ID: expertID, Score: scoreFn(raw + bias[expertID])}
			}
		} else {
			for expertID, raw := range row {
				scored[expertID] = expertScore{ID: expertID, Score: scoreFn(raw)}
			}
		}
		// slices.SortFunc with a top-level cmp avoids the interface
		// boxing of sort.Sort(sort.Interface(expertScoreSlice)) which
		// (per pprof) charged one alloc per call to the interface
		// conversion. compareExpertScoresDesc is a package-level
		// function so no closure is created. Ordering matches the
		// sort.Interface impl: Score descending, ID tie-break.
		slices.SortFunc(scored, compareExpertScoresDesc)
		start := tokenIndex * topK
		end := start + topK
		expertIDs := expertIDArena[start:end:end]
		weights := weightArena[start:end:end]
		total := float32(0)
		for i := 0; i < topK; i++ {
			expertIDs[i] = scored[i].ID
			weights[i] = scored[i].Score
			total += scored[i].Score
		}
		if total > 0 {
			for i := range weights {
				weights[i] /= total
			}
		}
		decisions = append(decisions, RouterDecision{
			TokenIndex: tokenIndex,
			ExpertIDs:  expertIDs,
			Weights:    weights,
		})
	}
	return decisions, nil
}

// DispatchExperts applies fake expert functions and weighted routing.
func DispatchExperts(hidden [][]float32, decisions []RouterDecision, experts map[int]ExpertFunc) ([][]float32, error) {
	out := make([][]float32, len(hidden))
	// Defensive-copy arena. The contract hands every ExpertFunc its own copy
	// of the hidden row so an expert can't mutate the caller's hidden state.
	// Previously this cloned per (token × expert) — 8× the bytes and one heap
	// alloc per call (128 allocs / ~131 KiB at the M2 routing shape). The copy
	// only needs to be per token: all experts for a token read the same row, so
	// one window per decision protects the caller's hidden just as well while
	// collapsing the clones into a single arena allocation. First pass sums the
	// per-decision row footprint (and validates token index + lengths) so the
	// second pass slices non-overlapping windows from one backing slab.
	rowArenaLen := 0
	for d := range decisions {
		decision := &decisions[d]
		tokenIndex := decision.TokenIndex
		if tokenIndex < 0 || tokenIndex >= len(hidden) {
			return nil, core.NewError(core.Sprintf("mlx: MiniMax M2 dispatch token index %d out of range", tokenIndex))
		}
		if len(decision.ExpertIDs) != len(decision.Weights) {
			return nil, core.NewError("mlx: MiniMax M2 dispatch expert/weight length mismatch")
		}
		if len(decision.ExpertIDs) > 0 {
			rowArenaLen += len(hidden[tokenIndex])
		}
	}
	rowArena := make([]float32, rowArenaLen)
	rowCursor := 0
	// Index iteration: RouterDecision is 56 B, exceeding the value-copy
	// threshold where range-by-value bites under hot fan-out.
	for d := range decisions {
		decision := &decisions[d]
		tokenIndex := decision.TokenIndex
		expertIDs := decision.ExpertIDs
		weights := decision.Weights
		if len(expertIDs) == 0 {
			continue
		}
		hiddenRow := hidden[tokenIndex]
		// One defensive copy per token, carved from the shared arena. Third-index
		// cap = len(hiddenRow) keeps any append inside this window's backing.
		rowEnd := rowCursor + len(hiddenRow)
		rowCopy := rowArena[rowCursor:rowEnd:rowEnd]
		copy(rowCopy, hiddenRow)
		rowCursor = rowEnd
		for i, expertID := range expertIDs {
			expert := experts[expertID]
			if expert == nil {
				return nil, core.NewError(core.Sprintf("mlx: MiniMax M2 dispatch missing expert %d", expertID))
			}
			result := expert(rowCopy)
			outRow := out[tokenIndex]
			if outRow == nil {
				outRow = make([]float32, len(result))
				out[tokenIndex] = outRow
			}
			if len(result) != len(outRow) {
				return nil, core.NewError("mlx: MiniMax M2 dispatch expert output shape mismatch")
			}
			weight := weights[i]
			for j, value := range result {
				outRow[j] += weight * value
			}
		}
	}
	return out, nil
}

// ProjectRouterScores computes hidden @ router.weight.T.
func ProjectRouterScores(hidden [][]float32, router RouterWeights) ([][]float32, error) {
	numExperts := router.NumExperts
	hiddenSize := router.HiddenSize
	if numExperts <= 0 || hiddenSize <= 0 {
		return nil, core.NewError("mlx: MiniMax M2 router requires expert and hidden sizes")
	}
	weight := router.Weight
	if len(weight) != numExperts*hiddenSize {
		return nil, core.NewError(core.Sprintf("mlx: MiniMax M2 router weight length %d, expected %d", len(weight), numExperts*hiddenSize))
	}
	out := make([][]float32, len(hidden))
	// Single arena for all per-token scores rows. Was one alloc per
	// token (len(hidden) small allocs); now one bulk alloc backing all
	// rows with third-index cap = numExperts for safe per-row append.
	scoresArena := make([]float32, len(hidden)*numExperts)
	for tokenIndex, row := range hidden {
		if len(row) != hiddenSize {
			return nil, core.NewError(core.Sprintf("mlx: MiniMax M2 router hidden row %d has %d values, expected %d", tokenIndex, len(row), hiddenSize))
		}
		start := tokenIndex * numExperts
		end := start + numExperts
		scores := scoresArena[start:end:end]
		// Hint the compiler that row[:hiddenSize] is in bounds, eliminating
		// the per-multiply bounds check on row[i] inside the hot dot-product
		// loop (16 tokens × 256 experts × 3072 fma = 12M iters per call).
		hiddenRow := row[:hiddenSize:hiddenSize]
		base := 0
		// hiddenSize is invariant across experts; precompute the unroll
		// boundary once per token instead of recomputing per expert.
		// 4-way accumulator unroll helps the compiler issue back-to-back
		// FMAs on Apple Silicon (W8-A2 pattern); tail loop handles the
		// hiddenSize % 4 remainder.
		unrollEnd := hiddenSize - (hiddenSize % 4)
		for expertID := range numExperts {
			expertWeights := weight[base : base+hiddenSize : base+hiddenSize]
			var s0, s1, s2, s3 float32
			i := 0
			for ; i < unrollEnd; i += 4 {
				s0 += hiddenRow[i] * expertWeights[i]
				s1 += hiddenRow[i+1] * expertWeights[i+1]
				s2 += hiddenRow[i+2] * expertWeights[i+2]
				s3 += hiddenRow[i+3] * expertWeights[i+3]
			}
			sum := s0 + s1 + s2 + s3
			for ; i < hiddenSize; i++ {
				sum += hiddenRow[i] * expertWeights[i]
			}
			scores[expertID] = sum
			base += hiddenSize
		}
		out[tokenIndex] = scores
	}
	return out, nil
}

// RouterProbeEvents converts router decisions into typed probe events.
func RouterProbeEvents(layer int, tokenIDs []int32, decisions []RouterDecision) []probe.Event {
	// Index iteration: RouterDecision is 56 B, above the value-copy
	// threshold where range-by-value bites under hot per-token fan-out.
	events := make([]probe.Event, len(decisions))
	tokenIDLen := len(tokenIDs)
	// Two-pass arena: sum the ExpertIDs + Weights footprint up front
	// then allocate one []int + one []float32 backing the per-event
	// clones. Was 2 × len(decisions) small allocs; now 2 allocs total
	// for the clones plus one bulk RouterDecision struct alloc (see
	// below). Sums are taken independently so a decision with
	// mismatched ExpertIDs / Weights lengths still clones each
	// faithfully (the existing per-event SliceClone path made no
	// length-match assumption either).
	totalIDs, totalWeights := 0, 0
	for d := range decisions {
		totalIDs += len(decisions[d].ExpertIDs)
		totalWeights += len(decisions[d].Weights)
	}
	idArena := make([]int, totalIDs)
	weightArena := make([]float32, totalWeights)
	// Bulk-allocate the per-event probe.RouterDecision payloads so the
	// per-event &probe.RouterDecision{} doesn't trigger one heap alloc
	// per event. Each event still gets a unique pointer via index alias.
	payloads := make([]probe.RouterDecision, len(decisions))
	idCursor, weightCursor := 0, 0
	for d := range decisions {
		decision := &decisions[d]
		tokenIndex := decision.TokenIndex
		tokenID := int32(0)
		if tokenIndex >= 0 && tokenIndex < tokenIDLen {
			tokenID = tokenIDs[tokenIndex]
		}
		// Preserve nil-vs-empty distinction from core.SliceClone: nil
		// input → nil output, empty-non-nil input → empty-non-nil arena
		// slice. Recorders/exporters can rely on the same shape.
		var ids []int
		if decision.ExpertIDs != nil {
			nID := len(decision.ExpertIDs)
			idEnd := idCursor + nID
			ids = idArena[idCursor:idEnd:idEnd]
			copy(ids, decision.ExpertIDs)
			idCursor = idEnd
		}
		var weights []float32
		if decision.Weights != nil {
			nW := len(decision.Weights)
			wEnd := weightCursor + nW
			weights = weightArena[weightCursor:wEnd:wEnd]
			copy(weights, decision.Weights)
			weightCursor = wEnd
		}
		payloads[d] = probe.RouterDecision{
			Layer:     layer,
			TokenID:   tokenID,
			ExpertIDs: ids,
			Weights:   weights,
		}
		events[d] = probe.Event{
			Kind:           probe.KindRouterDecision,
			Step:           tokenIndex,
			RouterDecision: &payloads[d],
			Meta:           metaMinimaxM2,
		}
	}
	return events
}

type expertScore struct {
	ID    int
	Score float32
}

// expertScoreSlice is a typed []expertScore used by RouteTokens as the
// per-call scoring buffer; the sort happens via slices.SortFunc + the
// package-level compareExpertScoresDesc comparator below to avoid the
// per-call sort.Interface boxing of sort.Sort.
type expertScoreSlice []expertScore

// compareExpertScoresDesc orders expertScore values by Score descending
// with an ID-ascending tie-break. The ID tie-break gives a total order
// over unique expert IDs so the sort is intrinsically stable. Lifted to
// package level so slices.SortFunc can use a direct func pointer instead
// of a per-call closure.
//
//	slices.SortFunc(scored, compareExpertScoresDesc)
func compareExpertScoresDesc(a, b expertScore) int {
	if a.Score > b.Score {
		return -1
	}
	if a.Score < b.Score {
		return 1
	}
	if a.ID < b.ID {
		return -1
	}
	if a.ID > b.ID {
		return 1
	}
	return 0
}

// scoringFunc returns the per-value scoring closure selected once for a
// router pass, hoisting the core.Lower(name) string transform out of the
// per-token inner loop.
func scoringFunc(name string) func(float32) float32 {
	switch core.Lower(name) {
	case "", "sigmoid":
		return sigmoidScore
	default:
		return identityScore
	}
}

func sigmoidScore(value float32) float32 {
	return float32(1 / (1 + math.Exp(float64(-value))))
}

func identityScore(value float32) float32 {
	return value
}
