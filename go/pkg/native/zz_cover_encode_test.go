// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"sort"
	"testing"

	"github.com/tmc/apple/metal"
)

// zz_cover_encode_test.go closes the per-op error legs INSIDE the single-command
// encoders (MLPBlockBF16, DecodeStepKV, AttentionStepKV, the MoE blocks,
// PerLayerInputs, the measure / chain / head_nocopy entries). Each builds its
// pipelines up front, opens a command encoder, and guards every encode step with
// `if encErr = encXxx(...); encErr != nil { enc.EndEncoding(); return }`. The
// per-op guard suite nulls the library, so the FIRST pipeline build fails before
// any encode runs — those legs are unreachable that way.
//
// The same single-key cache-eviction used for the ICB recorders works here on the
// NON-ICB caches: warm the whole op (a real successful call), then evict exactly
// one warmed pipeline key (across psoCache / ropePSOCache / ropePSOBF16Cache /
// ropeFreqsPSOBF16Cache / sdpaPSOCache) with the library nulled — so every
// earlier kernel is still cached and the lone evicted kernel's rebuild hits
// `library == nil`, surfacing through encXxx into the encode-step error leg at its
// call site. Evicting EVERY distinct warmed key in turn harvests exactly the
// independently-reachable legs (collision-siblings share a key and are skipped
// for free).

// the five non-ICB pipeline caches, snapshotted/cleared/restored as a set.
type psoCaches struct {
	plain    map[string]metal.MTLComputePipelineState
	rope     map[string]metal.MTLComputePipelineState
	ropeBF16 map[string]metal.MTLComputePipelineState
	freqs    map[string]metal.MTLComputePipelineState
	sdpa     map[string]metal.MTLComputePipelineState
}

func snapshotPSOCaches() psoCaches {
	cp := func(mu muLocker, m map[string]metal.MTLComputePipelineState) map[string]metal.MTLComputePipelineState {
		mu.Lock()
		defer mu.Unlock()
		out := make(map[string]metal.MTLComputePipelineState, len(m))
		for k, v := range m {
			out[k] = v
		}
		return out
	}
	return psoCaches{
		plain:    cp(&psoMu, psoCache),
		rope:     cp(&ropePSOMu, ropePSOCache),
		ropeBF16: cp(&ropePSOBF16Mu, ropePSOBF16Cache),
		freqs:    cp(&ropeFreqsPSOBF16Mu, ropeFreqsPSOBF16Cache),
		sdpa:     cp(&sdpaPSOMu, sdpaPSOCache),
	}
}

// muLocker abstracts *sync.Mutex so the snapshot/restore helpers can take a
// pointer to each cache's mutex uniformly.
type muLocker = interface {
	Lock()
	Unlock()
}

// installPSOCaches overwrites all five caches with the given snapshot (optionally
// dropping one key from whichever cache holds it).
func installPSOCaches(s psoCaches, dropKey string) {
	put := func(mu muLocker, dst *map[string]metal.MTLComputePipelineState, src map[string]metal.MTLComputePipelineState) {
		mu.Lock()
		defer mu.Unlock()
		m := make(map[string]metal.MTLComputePipelineState, len(src))
		for k, v := range src {
			if k == dropKey {
				continue
			}
			m[k] = v
		}
		*dst = m
	}
	put(&psoMu, &psoCache, s.plain)
	put(&ropePSOMu, &ropePSOCache, s.rope)
	put(&ropePSOBF16Mu, &ropePSOBF16Cache, s.ropeBF16)
	put(&ropeFreqsPSOBF16Mu, &ropeFreqsPSOBF16Cache, s.freqs)
	put(&sdpaPSOMu, &sdpaPSOCache, s.sdpa)
}

// allPSOKeys returns every key across the five caches in a snapshot, sorted.
func allPSOKeys(s psoCaches) []string {
	var ks []string
	for _, m := range []map[string]metal.MTLComputePipelineState{s.plain, s.rope, s.ropeBF16, s.freqs, s.sdpa} {
		for k := range m {
			ks = append(ks, k)
		}
	}
	sort.Strings(ks)
	return ks
}

// coverEncodeEvictAll warms invoke, then for each distinct warmed non-ICB pipeline
// key evicts it (library nulled) so the encode-step error leg at its call site
// fires. Library + caches are restored inline before each assert, so a t.Fatal
// never poisons a later test file.
func coverEncodeEvictAll(t *testing.T, invoke func() error) {
	t.Helper()
	// clear all five caches so the warmed snapshot is exactly this invoke's keys.
	clearPSOCaches()
	if err := invoke(); err != nil {
		t.Fatalf("warm: %v", err)
	}
	snap := snapshotPSOCaches()
	keys := allPSOKeys(snap)
	if len(keys) == 0 {
		t.Fatal("no non-ICB pipelines warmed")
	}
	oldLib := library
	errored := 0
	for _, key := range keys {
		installPSOCaches(snap, key)
		library = nil
		err := invoke()
		library = oldLib
		installPSOCaches(snap, "")
		// A warmed key that the invoke does NOT rebuild on its critical path (a
		// collision sibling, or a conditionally-taken branch like the composed-vs-
		// fused gelu) yields no error on eviction — that is expected, not a failure.
		// What the test asserts is that whenever a load-bearing pipeline fails, the
		// op surfaces the error instead of panicking or returning a nil-err buffer.
		if err != nil {
			errored++
		}
	}
	if errored == 0 {
		t.Fatal("no evicted pipeline produced an error — the eviction mechanism did not bite this op")
	}
}

func clearPSOCaches() {
	psoMu.Lock()
	psoCache = map[string]metal.MTLComputePipelineState{}
	psoMu.Unlock()
	ropePSOMu.Lock()
	ropePSOCache = map[string]metal.MTLComputePipelineState{}
	ropePSOMu.Unlock()
	ropePSOBF16Mu.Lock()
	ropePSOBF16Cache = map[string]metal.MTLComputePipelineState{}
	ropePSOBF16Mu.Unlock()
	ropeFreqsPSOBF16Mu.Lock()
	ropeFreqsPSOBF16Cache = map[string]metal.MTLComputePipelineState{}
	ropeFreqsPSOBF16Mu.Unlock()
	sdpaPSOMu.Lock()
	sdpaPSOCache = map[string]metal.MTLComputePipelineState{}
	sdpaPSOMu.Unlock()
}

// TestCoverMLPBlockBF16EncodeLegs covers the encode-step error legs in
// MLPBlockBF16 (the rms / gate-gemv / down-gemv / residual-add steps, plus the
// post-gelu error check) via single-key eviction.
func TestCoverMLPBlockBF16EncodeLegs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, dFF = 64, 256
	const eps = float32(1e-6)
	x := toBF16Bytes(syntheticFloat32(dModel, 1))
	normW := toBF16Bytes(syntheticFloat32(dModel, 3))
	wGate := toBF16Bytes(syntheticFloat32(dFF*dModel, 5))
	wUp := toBF16Bytes(syntheticFloat32(dFF*dModel, 7))
	wDown := toBF16Bytes(syntheticFloat32(dModel*dFF, 9))

	coverEncodeEvictAll(t, func() error {
		_, e := MLPBlockBF16(x, normW, wGate, wUp, wDown, dModel, dFF, eps)
		return e
	})
}

// TestCoverAttentionStepKVEncodeLegs covers the encAttnHalfKV error leg in
// AttentionStepKV via single-key eviction.
func TestCoverAttentionStepKVEncodeLegs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, maxLen, pos, dFF = 64, 4, 2, 64, 4, 0, 256
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-6)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 1))
	kCache := make([]byte, nKV*maxLen*headDim*bf16Size)
	vCache := make([]byte, nKV*maxLen*headDim*bf16Size)

	coverEncodeEvictAll(t, func() error {
		_, e := AttentionStepKV(x, layer.AttnNormW, layer.WQ, layer.WK, layer.WV, layer.WO, kCache, vCache,
			dModel, nHeads, nKV, headDim, maxLen, pos, base, scale, eps)
		return e
	})
}

// TestCoverDecodeStepKVEncodeLegs covers the attention-half + MLP-half encode
// error legs in DecodeStepKV via single-key eviction.
func TestCoverDecodeStepKVEncodeLegs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, maxLen, pos, dFF = 64, 4, 2, 64, 4, 0, 256
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-6)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 1))
	kCache := make([]byte, nKV*maxLen*headDim*bf16Size)
	vCache := make([]byte, nKV*maxLen*headDim*bf16Size)

	coverEncodeEvictAll(t, func() error {
		_, e := DecodeStepKV(x, layer.AttnNormW, layer.WQ, layer.WK, layer.WV, layer.WO, kCache, vCache,
			layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown,
			dModel, nHeads, nKV, headDim, maxLen, dFF, pos, base, scale, eps)
		return e
	})
}

// TestCoverMoEBlockBF16EncodeLegs covers the encode/op error legs in MoEBlockBF16
// and its mlpTransformBF16 helper via single-key eviction.
func TestCoverMoEBlockBF16EncodeLegs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, dFF, expertDFF = 64, 256, 256
	const eps = float32(1e-6)
	w := moeLayerWeightsFixture(2, 2, dModel, dFF, expertDFF, 3)
	h := toBF16Bytes(syntheticFloat32(dModel, 1))

	coverEncodeEvictAll(t, func() error {
		_, e := MoEBlockBF16(h, w, dModel, dFF, eps)
		return e
	})
}

// TestCoverMoEBlockQuantEncodeLegs covers the encode/op error legs in
// MoEBlockQuant and its mlpTransformQuant helper via single-key eviction.
func TestCoverMoEBlockQuantEncodeLegs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, dFF, expertDFF = 64, 256, 256
	const gs, bits = 64, 4
	const eps = float32(1e-6)
	w := quantMoELayerWeightsGuard(t, 2, 2, dModel, dFF, expertDFF, gs, bits)
	h := toBF16Bytes(syntheticFloat32(dModel, 1))

	coverEncodeEvictAll(t, func() error {
		_, e := MoEBlockQuant(h, w, dModel, dFF, eps)
		return e
	})
}

// TestCoverPerLayerInputsEncodeLegs covers the downstream-op error legs in
// PerLayerInputs (the bf16-projection path) via single-key eviction. Each step
// (embed gather, project matvec, scale-mul, rms, add, combine-mul) uses a distinct
// kernel sequence, so the legs flip independently.
func TestCoverPerLayerInputsEncodeLegs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, pliDim, numLayers, vocabPLI = 64, 32, 2, 8
	const eps = float32(1e-5)
	plDim := numLayers * pliDim
	embedPacked := toBF16Bytes(syntheticFloat32(vocabPLI*plDim, 3))
	projW := toBF16Bytes(syntheticFloat32(plDim*dModel, 5))
	projNormW := toBF16Bytes(syntheticFloat32(pliDim, 7))
	hidden := toBF16Bytes(syntheticFloat32(dModel, 9))

	coverEncodeEvictAll(t, func() error {
		_, e := PerLayerInputs(embedPacked, nil, nil, projW, nil, nil, projNormW, 0, hidden,
			vocabPLI, numLayers, pliDim, dModel, 0, 0, 0, 0, eps)
		return e
	})
}

// TestCoverLMHeadEncodeLegs covers the downstream-op error leg in LMHeadBF16
// (the final matvec after the norm) via single-key eviction.
func TestCoverLMHeadEncodeLegs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, vocab = 64, 32
	const eps = float32(1e-6)
	hidden := toBF16Bytes(syntheticFloat32(dModel, 1))
	finalNormW := toBF16Bytes(syntheticFloat32(dModel, 3))
	outWeight := toBF16Bytes(syntheticFloat32(vocab*dModel, 5))

	coverEncodeEvictAll(t, func() error {
		_, e := LMHeadBF16(hidden, finalNormW, outWeight, dModel, vocab, eps, 0)
		return e
	})
}

// TestCoverChainEncodeLegs covers the float32 chain ops MLPBlock + NormProject
// (the gemv/encode legs) via single-key eviction.
func TestCoverChainEncodeLegs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, dFF = 64, 256
	const eps = float32(1e-6)
	x := syntheticFloat32(dModel, 1)
	normW := syntheticFloat32(dModel, 3)
	wGate := syntheticFloat32(dFF*dModel, 5)
	wUp := syntheticFloat32(dFF*dModel, 7)
	wDown := syntheticFloat32(dModel*dFF, 9)
	projW := syntheticFloat32(dModel*dModel, 11)

	coverEncodeEvictAll(t, func() error {
		if _, e := MLPBlock(x, normW, wGate, wUp, wDown, dModel, dFF, eps); e != nil {
			return e
		}
		_, e := NormProject(x, normW, projW, dModel, dModel, eps)
		return e
	})
}
