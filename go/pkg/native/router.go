// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// bf16ToF32 decodes one little-endian bf16 (2 bytes: lo, hi) to float32 — the
// inverse of f32ToBF16, for reading a device result back to the host.
func bf16ToF32(lo, hi byte) float32 {
	return math.Float32frombits(uint32(uint16(lo)|uint16(hi)<<8) << 16)
}

// topKByScore returns the indices of the topK highest scores, highest first,
// with ties broken by lower index. It deliberately selects only the requested
// experts instead of sorting the full expert list, matching the router hot path's
// top-k shape.
func topKByScore(scores []float32, topK int) []int32 {
	out := make([]int32, topK)
	for slot := 0; slot < topK; slot++ {
		best := -1
		for i, score := range scores {
			if selectedExpert(out[:slot], int32(i)) {
				continue
			}
			if best < 0 || score > scores[best] {
				best = i
			}
		}
		out[slot] = int32(best)
	}
	return out
}

func selectedExpert(selected []int32, expert int32) bool {
	for _, v := range selected {
		if v == expert {
			return true
		}
	}
	return false
}

// softmaxAt returns softmax over the scores at idx (max-subtracted for stability),
// in idx order, as float32.
func softmaxAt(scores []float32, idx []int32) []float32 {
	maxS := float32(math.Inf(-1))
	for _, e := range idx {
		if scores[e] > maxS {
			maxS = scores[e]
		}
	}
	w := make([]float32, len(idx))
	var sum float32
	for i, e := range idx {
		w[i] = float32(math.Exp(float64(scores[e] - maxS)))
		sum += w[i]
	}
	for i := range w {
		w[i] /= sum
	}
	return w
}

// MoERouter runs the gemma4 MoE router: it RMS-norms x with the pre-scaled router
// norm weight, projects to per-expert scores, selects the topK highest-scoring
// experts and softmaxes their scores — optionally multiplying each by its per-expert
// scale. Returns (idx, weights) ready to feed MoEExperts.
//
// normWScaled is the router norm weight ALREADY scaled by RootSize (= dModel^-0.5),
// folded once at load exactly like the metal model caches ScaleScaled = Scale·RootSize
// — so this sub-slice needs no on-device scalar-mul. perExpertScale (numExperts bf16)
// is optional; pass nil to skip it. routerW is [numExperts × dModel] row-major bf16
// (each expert is a row), x is dModel bf16; idx is topK int32, weights topK bf16.
//
// The hot path keeps RMSNorm, score projection, top-k, softmax, and optional
// per-expert scaling in one command buffer via the native router top-k kernel,
// mirroring pkg/metal's NativeMoERouterTopK feature. The host selector remains
// only for shapes the copied kernel does not support, such as topK > 32.
//
// The routing decision is order-INVARIANT: each selected expert's weight is
// independent of the order idx is returned in (softmax is over the selected scores;
// the downstream combine is a commutative weighted sum). The parity gate therefore
// compares expert→weight maps, not positional sequences.
func MoERouter(x, normWScaled, routerW, perExpertScale []byte, numExperts, topK, dModel int, eps float32) ([]int32, []byte, error) {
	if err := ensureInit(); err != nil {
		return nil, nil, err
	}
	if len(x) != dModel*bf16Size {
		return nil, nil, core.NewError("native.MoERouter: x must be dModel bf16 bytes")
	}
	if len(normWScaled) != dModel*bf16Size {
		return nil, nil, core.NewError("native.MoERouter: normWScaled must be dModel bf16 bytes")
	}
	if len(routerW) != numExperts*dModel*bf16Size {
		return nil, nil, core.NewError("native.MoERouter: routerW must be numExperts*dModel bf16 bytes")
	}
	if perExpertScale != nil && len(perExpertScale) != numExperts*bf16Size {
		return nil, nil, core.NewError("native.MoERouter: perExpertScale must be numExperts bf16 bytes (or nil)")
	}
	if topK <= 0 || topK > numExperts {
		return nil, nil, core.NewError("native.MoERouter: topK must be in 1..numExperts")
	}

	if idx, weights, ok, err := moeRouterBF16DeviceTopK(x, normWScaled, routerW, perExpertScale, numExperts, topK, dModel, eps); ok || err != nil {
		return idx, weights, err
	}

	// fallback: RMS-norm and score projection on device, host top-k/softmax.
	normed, err := RMSNormBF16(x, normWScaled, 1, dModel, eps)
	if err != nil {
		return nil, nil, err
	}
	scoresB, err := matVecBF16Resident(routerW, normed, numExperts, dModel)
	if err != nil {
		return nil, nil, err
	}
	idx, weights := routerSelect(scoresB, perExpertScale, numExperts, topK)
	return idx, weights, nil
}

func moeRouterBF16DeviceTopK(x, normWScaled, routerW, perExpertScale []byte, numExperts, topK, dModel int, eps float32) ([]int32, []byte, bool, error) {
	if !routerTopKUsable(numExperts, topK) {
		return nil, nil, false, nil
	}
	var idx []int32
	var weights []byte
	var encErr error
	withAutoreleasePool(func() {
		xBuf := sharedBytes(x)
		normBuf := residentBytes(normWScaled)
		routerBuf := residentBytes(routerW)
		var scaleBuf metal.MTLBuffer
		if perExpertScale != nil {
			scaleBuf = residentBytes(perExpertScale)
		}
		normedBuf := scratchBF16(dModel)
		scoresBuf := scratchBF16(numExperts)
		idxBuf := device.NewBufferWithLengthOptions(uint(topK*4), metal.MTLResourceStorageModeShared)
		weightBuf := scratchBF16(topK)

		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		if encErr = encRMSNormBF16(enc, xBuf, normBuf, normedBuf, 0, dModel, eps); encErr != nil {
			enc.EndEncoding()
			return
		}
		if encErr = encGemvBF16(enc, routerBuf, normedBuf, scoresBuf, numExperts, dModel); encErr != nil {
			enc.EndEncoding()
			return
		}
		if encErr = encRouterTopKBF16(enc, scoresBuf, scaleBuf, idxBuf, weightBuf, 0, numExperts, topK, perExpertScale != nil); encErr != nil {
			enc.EndEncoding()
			return
		}
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		idx, weights = copyRouterTopKOutput(idxBuf, weightBuf, topK)
	})
	if encErr != nil {
		return nil, nil, true, encErr
	}
	return idx, weights, true, nil
}

func matVecBF16Resident(mat, vec []byte, outDim, inDim int) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(mat) != outDim*inDim*bf16Size {
		return nil, core.NewError("native.matVecBF16Resident: mat must be outDim*inDim bf16 bytes")
	}
	if len(vec) != inDim*bf16Size {
		return nil, core.NewError("native.matVecBF16Resident: vec must be inDim bf16 bytes")
	}
	if outDim == 0 || inDim == 0 {
		return make([]byte, outDim*bf16Size), nil
	}
	return MatVecBF16Buf(bufView{buf: residentBytes(mat)}, vec, outDim, inDim)
}

// routerSelect performs the host top-k + softmax (+ optional per-expert scale) over the raw
// per-expert scores (numExperts bf16) — the routing decision shared by MoERouter and
// MoERouterQuant (they differ only in how the scores are projected: bf16 gemv vs 4-bit qmv).
func routerSelect(scoresB, perExpertScale []byte, numExperts, topK int) ([]int32, []byte) {
	scores := make([]float32, numExperts)
	for e := 0; e < numExperts; e++ {
		scores[e] = bf16ToF32(scoresB[e*bf16Size], scoresB[e*bf16Size+1])
	}
	idx := topKByScore(scores, topK)
	w := softmaxAt(scores, idx)
	if perExpertScale != nil {
		for i, e := range idx {
			w[i] *= bf16ToF32(perExpertScale[int(e)*bf16Size], perExpertScale[int(e)*bf16Size+1])
		}
	}
	weights := make([]byte, topK*bf16Size)
	for i, v := range w {
		h := f32ToBF16(v)
		weights[i*bf16Size] = byte(h)
		weights[i*bf16Size+1] = byte(h >> 8)
	}
	return idx, weights
}

// MoERouterQuant is MoERouter with a quantised expert-score projection (gemma4
// 26B-A4B's router.proj is affine-quantised). RMS-norm, resident QMV score
// projection, top-k, softmax, and optional scale use the same device router
// top-k path as MoERouter when the copied kernel supports the shape.
func MoERouterQuant(x, normWScaled []byte, routerProj QuantWeight, perExpertScale []byte, numExperts, topK, dModel, groupSize, bits int, eps float32) ([]int32, []byte, error) {
	return moeRouterQuantWithViews(x, normWScaled, bufView{}, routerProj, perExpertScale, bufView{}, numExperts, topK, dModel, groupSize, bits, eps)
}

func moeRouterQuantWithViews(x, normWScaled []byte, normView bufView, routerProj QuantWeight, perExpertScale []byte, perExpertScaleView bufView, numExperts, topK, dModel, groupSize, bits int, eps float32) ([]int32, []byte, error) {
	if err := ensureInit(); err != nil {
		return nil, nil, err
	}
	if len(x) != dModel*bf16Size {
		return nil, nil, core.NewError("native.MoERouterQuant: x must be dModel bf16 bytes")
	}
	if len(normWScaled) != dModel*bf16Size {
		return nil, nil, core.NewError("native.MoERouterQuant: normWScaled must be dModel bf16 bytes")
	}
	if topK <= 0 || topK > numExperts {
		return nil, nil, core.NewError("native.MoERouterQuant: topK must be in 1..numExperts")
	}
	if perExpertScale != nil && len(perExpertScale) != numExperts*bf16Size {
		return nil, nil, core.NewError("native.MoERouterQuant: perExpertScale must be numExperts bf16 bytes (or nil)")
	}
	groupSize, bits = quantWeightGeometryForShape(routerProj, numExperts, dModel, groupSize, bits)
	if groupSize <= 0 || dModel%groupSize != 0 {
		return nil, nil, core.NewError("native.MoERouterQuant: groupSize must divide dModel")
	}
	wantPacked, wantSB := numExperts*dModel*bits/8, numExperts*(dModel/groupSize)*bf16Size
	if len(routerProj.Packed) != wantPacked || len(routerProj.Scales) != wantSB || len(routerProj.Biases) != wantSB {
		return nil, nil, core.NewError("native.MoERouterQuant: routerProj size mismatch vs numExperts×dModel")
	}

	if idx, weights, ok, err := moeRouterQuantDeviceTopK(x, normWScaled, normView, routerProj, perExpertScale, perExpertScaleView, numExperts, topK, dModel, groupSize, bits, eps); ok || err != nil {
		return idx, weights, err
	}

	normed, err := rmsNormBF16View(x, normWScaled, normView, 1, dModel, eps)
	if err != nil {
		return nil, nil, err
	}
	scoresB, err := qmvBF16Resident(normed, routerProj, numExperts, dModel, groupSize, bits)
	if err != nil {
		return nil, nil, err
	}
	idx, weights := routerSelect(scoresB, perExpertScale, numExperts, topK)
	return idx, weights, nil
}

func moeRouterQuantDeviceTopK(x, normWScaled []byte, normView bufView, routerProj QuantWeight, perExpertScale []byte, perExpertScaleView bufView, numExperts, topK, dModel, groupSize, bits int, eps float32) ([]int32, []byte, bool, error) {
	if !routerTopKUsable(numExperts, topK) {
		return nil, nil, false, nil
	}
	var idx []int32
	var weights []byte
	var encErr error
	withAutoreleasePool(func() {
		xBuf := sharedBytes(x)
		normBuf := bf16WeightView(normWScaled, normView)
		wBuf, scalesBuf, biasesBuf := quantWeightViews(routerProj)
		var scaleBuf metal.MTLBuffer
		var scaleOff uint
		if perExpertScale != nil {
			scaleView := bf16WeightView(perExpertScale, perExpertScaleView)
			scaleBuf, scaleOff = scaleView.buf, scaleView.off
		}
		normedBuf := scratchBF16(dModel)
		scoresBuf := scratchBF16(numExperts)
		idxBuf := device.NewBufferWithLengthOptions(uint(topK*4), metal.MTLResourceStorageModeShared)
		weightBuf := scratchBF16(topK)

		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		if encErr = encRMSNormBF16(enc, xBuf, normBuf.buf, normedBuf, normBuf.off, dModel, eps); encErr != nil {
			enc.EndEncoding()
			return
		}
		if encErr = encQMVBF16(enc, wBuf.buf, scalesBuf.buf, biasesBuf.buf, normedBuf, scoresBuf, wBuf.off, scalesBuf.off, biasesBuf.off, 0, numExperts, dModel, groupSize, bits); encErr != nil {
			enc.EndEncoding()
			return
		}
		if encErr = encRouterTopKBF16(enc, scoresBuf, scaleBuf, idxBuf, weightBuf, scaleOff, numExperts, topK, perExpertScale != nil); encErr != nil {
			enc.EndEncoding()
			return
		}
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		idx, weights = copyRouterTopKOutput(idxBuf, weightBuf, topK)
	})
	if encErr != nil {
		return nil, nil, true, encErr
	}
	return idx, weights, true, nil
}

func routerTopKUsable(numExperts, topK int) bool {
	if topK <= 0 || topK > numExperts || topK > routerTopKMaxK {
		return false
	}
	_, err := routerTopKPipeline()
	return err == nil
}

func copyRouterTopKOutput(idxBuf, weightBuf metal.MTLBuffer, topK int) ([]int32, []byte) {
	idx := make([]int32, topK)
	weights := make([]byte, topK*bf16Size)
	copy(idx, unsafe.Slice((*int32)(idxBuf.Contents()), topK))
	copy(weights, unsafe.Slice((*byte)(weightBuf.Contents()), topK*bf16Size))
	return idx, weights
}
