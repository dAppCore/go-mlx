// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"time"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/pkg/metal/model/gemma4"
)

// multimodalDecodeResult carries the shared verb decode-loop outcome.
type multimodalDecodeResult struct {
	Generated  []int32
	PrefillDur time.Duration
	DecodeDur  time.Duration
}

// multimodalGreedyDecode runs the self-contained verb loop shared by the
// audio and vision commands: multimodal prefill over the placeholder-bearing
// token sequence, then greedy decode until a stop token or the bound.
func multimodalGreedyDecode(ctx context.Context, m *gemma4.Gemma4Model, ids []int32, images, audio, video []*metal.Array, maxTokens int) (multimodalDecodeResult, error) {
	var res multimodalDecodeResult

	capacity := len(ids) + maxTokens + 64
	caches := make([]metal.Cache, m.NumLayers())
	for i := range caches {
		caches[i] = metal.NewFixedKVCache(capacity)
	}
	defer metal.FreeCaches(caches)

	stopIDs := map[int32]struct{}{m.Tok.EOSToken(): {}}
	if eot := m.Tok.Encode("<turn|>"); len(eot) == 1 {
		stopIDs[eot[0]] = struct{}{}
	}

	start := time.Now()
	prefill := metal.FromValues(ids, 1, len(ids))
	logits := m.ForwardUnifiedVideoMultiModal(prefill, images, audio, video, caches)
	metal.Free(prefill)
	res.PrefillDur = time.Since(start)

	res.Generated = make([]int32, 0, maxTokens)
	decodeStart := time.Now()
	for len(res.Generated) < maxTokens {
		select {
		case <-ctx.Done():
			metal.Free(logits)
			return res, core.NewError("mlx: cancelled")
		default:
		}
		last := metal.SliceAxis(logits, 1, int32(logits.Dim(1)-1), int32(logits.Dim(1)))
		next := metal.Argmax(last, -1, false)
		if err := metal.Eval(next); err != nil {
			metal.Free(logits, last, next)
			return res, err
		}
		id := int32(next.Int())
		metal.Free(logits, last, next)
		metal.DetachCaches(caches)
		if _, stop := stopIDs[id]; stop {
			res.DecodeDur = time.Since(decodeStart)
			return res, nil
		}
		res.Generated = append(res.Generated, id)
		step := metal.FromValues([]int32{id}, 1, 1)
		logits = m.Forward(step, caches)
		metal.Free(step)
	}
	metal.Free(logits)
	res.DecodeDur = time.Since(decodeStart)
	return res, nil
}

// countTokenID reports how many times id occurs in ids.
func countTokenID(ids []int32, id int32) int {
	n := 0
	for _, v := range ids {
		if v == id {
			n++
		}
	}
	return n
}
