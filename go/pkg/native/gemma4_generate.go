// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/model"
	g4 "dappco.re/go/mlx/pkg/model/gemma4"
)

// GenerateGemma4BF16 is the autoregressive token loop on an assembled bf16 gemma4 — the
// whole chain end to end: embed the ids → DecodeForward (the norm-faithful arch decode,
// behind model.Backend) → LM head on the last hidden state → greedy argmax → append,
// until maxNew tokens or eosID (eosID < 0 disables early stop). Returns the generated ids
// (excluding the prompt).
//
// Whole-sequence today: each step re-decodes the full running sequence over a fresh cache
// (correct, but O(N²) — incremental single-token decode with a persistent cache is the
// efficiency follow-up the model.Backend doc flags). Greedy/deterministic — the right shape
// for a tok/s bench; a sampled variant can layer model.Sampler on the same logits. The
// embedding scale is √hidden, eps/softCap come from the arch.
func GenerateGemma4BF16(g *Gemma4BF16, arch g4.Arch, promptIDs []int32, maxNew, maxLen, eosID int) ([]int32, error) {
	if g == nil || len(g.Layers) != len(arch.Layer) {
		return nil, core.NewError("native.GenerateGemma4BF16: weights/arch layer count mismatch")
	}
	if len(promptIDs) == 0 {
		return nil, core.NewError("native.GenerateGemma4BF16: empty prompt")
	}
	if maxNew <= 0 {
		return nil, core.NewError("native.GenerateGemma4BF16: maxNew must be > 0")
	}
	if len(promptIDs)+maxNew > maxLen {
		return nil, core.NewError("native.GenerateGemma4BF16: prompt + maxNew exceeds maxLen cache rows")
	}
	backend, err := NewBF16Backend(arch, g.Layers, maxLen)
	if err != nil {
		return nil, err
	}
	embedScale := float32(math.Sqrt(float64(arch.Hidden)))

	ids := append([]int32(nil), promptIDs...)
	gen := make([]int32, 0, maxNew)
	for len(gen) < maxNew {
		embs, err := EmbedTokensBF16(g.Embed, ids, arch.Vocab, arch.Hidden, embedScale)
		if err != nil {
			return nil, err
		}
		hidden, err := backend.DecodeForward(embs)
		if err != nil {
			return nil, err
		}
		logits, err := LMHeadBF16(hidden[len(hidden)-1], g.FinalNorm, g.LMHead, arch.Hidden, arch.Vocab, arch.Eps, arch.SoftCap)
		if err != nil {
			return nil, err
		}
		next, err := model.Greedy(logits, arch.Vocab)
		if err != nil {
			return nil, err
		}
		ids = append(ids, next)
		gen = append(gen, next)
		if eosID >= 0 && int(next) == eosID {
			break
		}
	}
	return gen, nil
}
