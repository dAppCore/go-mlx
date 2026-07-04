// SPDX-Licence-Identifier: EUPL-1.2

package m2

import (
	core "dappco.re/go"
	"dappco.re/go/inference/quant/jang"
	"dappco.re/go/inference/probe"
	"dappco.re/go/inference/safetensors"
)

// LoadPackedExpertsForDecisions reads only the routed
// experts referenced by decisions from safetensors shards.
func LoadPackedExpertsForDecisions(plan TensorPlan, weightFiles []string, layer int, decisions []RouterDecision) (map[int]PackedExpertWeights, error) {
	return LoadPackedExperts(plan, weightFiles, layer, decisionExpertIDs(decisions))
}

// LoadLazyExpertsForHidden loads the router, computes
// top-k decisions for hidden states, and then reads only the selected routed
// expert payloads from safetensors.
func LoadLazyExpertsForHidden(plan TensorPlan, weightFiles []string, layer int, hidden [][]float32, tokenIDs []int32, sink probe.Sink) (LazyExpertLoad, error) {
	router, err := LoadRouter(plan, weightFiles, layer)
	if err != nil {
		return LazyExpertLoad{}, err
	}
	scores, err := ProjectRouterScores(hidden, router)
	if err != nil {
		return LazyExpertLoad{}, err
	}
	decisions, err := RouteTokens(plan.Config, scores, router.Bias)
	if err != nil {
		return LazyExpertLoad{}, err
	}
	experts, err := LoadPackedExpertsForDecisions(plan, weightFiles, layer, decisions)
	if err != nil {
		return LazyExpertLoad{}, err
	}
	events := RouterProbeEvents(layer, tokenIDs, decisions)
	for _, event := range events {
		if sink != nil {
			sink.EmitProbe(event)
		}
	}
	return LazyExpertLoad{
		Layer:             layer,
		Router:            router,
		Scores:            scores,
		Decisions:         decisions,
		SelectedExpertIDs: decisionExpertIDsSorted(decisions),
		Experts:           experts,
		LoadedPackedBytes: packedExpertLoadedBytes(experts),
		ProbeEvents:       events,
	}, nil
}

// LoadPackedExperts resolves selected MiniMax M2 routed
// expert projections from safetensors metadata and reads only their packed
// bytes plus quantisation sidecars.
func LoadPackedExperts(plan TensorPlan, weightFiles []string, layer int, expertIDs []int) (map[int]PackedExpertWeights, error) {
	if len(weightFiles) == 0 {
		return nil, core.NewError("mlx: MiniMax M2 packed expert loading requires safetensors weight files")
	}
	index, err := safetensors.IndexFiles(weightFiles)
	if err != nil {
		return nil, core.E("minimax_m2.packed_experts", "index safetensors", err)
	}
	// Every routed expert's three projections (and their packed weight +
	// scales/biases sidecars) live in the model's safetensors shards, and an
	// 8-expert load reads ~70 tensors. The leaf ReadRefRaw/ReadRefValues each
	// core.Open(ref.Path) per ref, so without a shared handle this reopened
	// the same shard once per tensor — os.newFile + the path→C-string
	// syscall.ByteSliceFromString alloc ~70 times. The ShardCache opens each
	// distinct shard once and serves every ref over the shared handle via
	// ReadAt; reads stay byte-identical to the leaf functions. Closed once
	// the whole expert set is loaded.
	cache := safetensors.NewShardCache()
	defer cache.Close()
	out := make(map[int]PackedExpertWeights, len(expertIDs))
	for _, expertID := range uniqueExpertIDs(expertIDs) {
		// Only the three routed-expert projections are consumed per expert;
		// expertProjectionSpecs builds exactly those, avoiding the four
		// attention specs + router gate/bias specs (and their thrown-away
		// packed descriptors) that LayerTensorSpecs would rebuild on every
		// expert at the MoE-load multiplier.
		gateSpec, upSpec, downSpec, err := plan.expertProjectionSpecs(layer, expertID)
		if err != nil {
			return nil, err
		}
		gate, err := loadPackedProjection(cache, index, &gateSpec)
		if err != nil {
			return nil, core.E("minimax_m2.packed_experts", core.Sprintf("expert %d gate_proj", expertID), err)
		}
		up, err := loadPackedProjection(cache, index, &upSpec)
		if err != nil {
			return nil, core.E("minimax_m2.packed_experts", core.Sprintf("expert %d up_proj", expertID), err)
		}
		down, err := loadPackedProjection(cache, index, &downSpec)
		if err != nil {
			return nil, core.E("minimax_m2.packed_experts", core.Sprintf("expert %d down_proj", expertID), err)
		}
		out[expertID] = PackedExpertWeights{GateProj: gate, UpProj: up, DownProj: down}
	}
	return out, nil
}

// DequantizedExperts expands all loaded packed expert projections with the
// reference JANG dequantizer. Native fused kernels can bypass this host path.
func (load LazyExpertLoad) DequantizedExperts() (map[int]DenseExpertWeights, error) {
	out := make(map[int]DenseExpertWeights, len(load.Experts))
	for expertID, expert := range load.Experts {
		gate, err := DequantizeJANGPackedProjection(expert.GateProj)
		if err != nil {
			return nil, core.E("minimax_m2.dequantized_experts", core.Sprintf("expert %d gate_proj", expertID), err)
		}
		up, err := DequantizeJANGPackedProjection(expert.UpProj)
		if err != nil {
			return nil, core.E("minimax_m2.dequantized_experts", core.Sprintf("expert %d up_proj", expertID), err)
		}
		down, err := DequantizeJANGPackedProjection(expert.DownProj)
		if err != nil {
			return nil, core.E("minimax_m2.dequantized_experts", core.Sprintf("expert %d down_proj", expertID), err)
		}
		out[expertID] = DenseExpertWeights{GateProj: gate, UpProj: up, DownProj: down}
	}
	return out, nil
}

// DequantizeJANGPackedProjection expands one packed projection payload using
// its descriptor and affine sidecars.
func DequantizeJANGPackedProjection(tensor JANGPackedProjectionTensor) (DenseProjectionTensor, error) {
	weight, err := jang.DequantizePackedTensor(tensor.Descriptor, tensor.Packed, tensor.Scales, tensor.Biases)
	if err != nil {
		return DenseProjectionTensor{}, err
	}
	return DenseProjectionTensor{
		Descriptor: tensor.Descriptor,
		Weight:     weight,
		Bias:       core.SliceClone(tensor.Bias),
	}, nil
}

// LoadRouter resolves and reads the dense MiniMax M2
// router gate for one layer from safetensors shards.
func LoadRouter(plan TensorPlan, weightFiles []string, layer int) (RouterWeights, error) {
	if len(weightFiles) == 0 {
		return RouterWeights{}, core.NewError("mlx: MiniMax M2 router loading requires safetensors weight files")
	}
	specs, err := plan.LayerTensorSpecs(layer, 0)
	if err != nil {
		return RouterWeights{}, err
	}
	routerSpec := findTensorSpec(specs, TensorRoleRouterGate)
	index, err := safetensors.IndexFiles(weightFiles)
	if err != nil {
		return RouterWeights{}, core.E("minimax_m2.router", "index safetensors", err)
	}
	ref, name, ok := findSafetensorRef(index, routerGateCandidates(&routerSpec))
	if !ok {
		return RouterWeights{}, core.NewError("mlx: MiniMax M2 router missing gate tensor: " + routerSpec.Name)
	}
	// Gate + correction bias live in the same shard; a ShardCache opens it
	// once and serves both reads over one handle (byte-identical to the leaf
	// ReadRefValues). Closed when the router load completes.
	cache := safetensors.NewShardCache()
	defer cache.Close()
	weight, err := cache.ReadRefValues(ref)
	if err != nil {
		return RouterWeights{}, core.E("minimax_m2.router", "read gate", err)
	}
	if len(ref.Shape) != 2 || int(ref.Shape[0]) != plan.Config.NumLocalExperts || int(ref.Shape[1]) != plan.Config.HiddenSize {
		return RouterWeights{}, core.NewError(core.Sprintf("mlx: MiniMax M2 router gate shape %+v, expected [%d %d]", ref.Shape, plan.Config.NumLocalExperts, plan.Config.HiddenSize))
	}
	router := RouterWeights{
		Name:       name,
		Weight:     weight,
		NumExperts: int(ref.Shape[0]),
		HiddenSize: int(ref.Shape[1]),
	}
	biasSpec := findTensorSpec(specs, TensorRoleRouterBias)
	if biasRef, _, ok := findSafetensorRef(index, routerBiasCandidates(&biasSpec, layer)); ok {
		router.Bias, err = cache.ReadRefValues(biasRef)
		if err != nil {
			return RouterWeights{}, core.E("minimax_m2.router", "read correction bias", err)
		}
		if len(router.Bias) != router.NumExperts {
			return RouterWeights{}, core.NewError(core.Sprintf("mlx: MiniMax M2 router bias length %d, expected %d", len(router.Bias), router.NumExperts))
		}
	} else if plan.Config.UseRoutingBias {
		return RouterWeights{}, core.NewError("mlx: MiniMax M2 router missing correction bias")
	}
	return router, nil
}

// BuildLayerForwardSkeleton resolves and validates the
// attention/router tensor contract for one MiniMax M2 layer using safetensors
// metadata only. It does not read payloads or run kernels.
func BuildLayerForwardSkeleton(plan TensorPlan, weightFiles []string, layer int) (LayerForwardSkeleton, error) {
	if len(weightFiles) == 0 {
		return LayerForwardSkeleton{}, core.NewError("mlx: MiniMax M2 layer skeleton requires safetensors weight files")
	}
	specs, err := plan.LayerTensorSpecs(layer, 0)
	if err != nil {
		return LayerForwardSkeleton{}, err
	}
	index, err := safetensors.IndexFiles(weightFiles)
	if err != nil {
		return LayerForwardSkeleton{}, core.E("minimax_m2.layer_skeleton", "index safetensors", err)
	}
	skeleton := LayerForwardSkeleton{Layer: layer, Attention: make([]ResolvedTensor, 0, 4)}
	for _, role := range attentionSkeletonRoles {
		resolved, err := resolveSkeletonTensor(index, findTensorSpec(specs, role), packedWeightCandidates)
		if err != nil {
			return LayerForwardSkeleton{}, err
		}
		skeleton.Attention = append(skeleton.Attention, resolved)
	}
	routerGate, err := resolveSkeletonTensor(index, findTensorSpec(specs, TensorRoleRouterGate), routerGateCandidates)
	if err != nil {
		return LayerForwardSkeleton{}, err
	}
	skeleton.RouterGate = routerGate
	if plan.Config.UseRoutingBias {
		biasSpec := findTensorSpec(specs, TensorRoleRouterBias)
		routerBias, err := resolveSkeletonTensor(index, biasSpec, func(spec *TensorSpec) []string {
			return routerBiasCandidates(spec, layer)
		})
		if err != nil {
			return LayerForwardSkeleton{}, err
		}
		skeleton.RouterBias = &routerBias
	}
	return skeleton, nil
}

// loadPackedProjection reads one packed projection's weight + scales/biases
// (and optional projection bias) from the safetensors shards. The four/five
// payload reads route through the caller's ShardCache so all refs in a shard
// share one open handle — the per-ref core.Open the leaf ReadRef* would pay
// is eliminated across the whole expert load. Read bytes are byte-identical
// to the leaf functions (same ReadAt offset + DecodeFloatData).
func loadPackedProjection(cache *safetensors.ShardCache, index safetensors.Index, spec *TensorSpec) (JANGPackedProjectionTensor, error) {
	if spec.Packed == nil {
		return JANGPackedProjectionTensor{}, core.NewError("mlx: MiniMax M2 packed projection missing descriptor: " + spec.Name)
	}
	weightRef, weightName, ok := findPackedWeightRef(index, spec)
	if !ok {
		return JANGPackedProjectionTensor{}, core.NewError("mlx: MiniMax M2 packed projection missing weight tensor: " + spec.Name)
	}
	if !packedDType(weightRef.DType) {
		return JANGPackedProjectionTensor{}, core.NewError(core.Sprintf("mlx: MiniMax M2 packed projection %s dtype %s is not U8", weightName, weightRef.DType))
	}
	packed, err := cache.ReadRefRaw(weightRef)
	if err != nil {
		return JANGPackedProjectionTensor{}, err
	}
	// The production loader uses only the resolved ref (error diagnostics key
	// off spec.Name), so it walks the no-name fan-out and skips the per-hit
	// sidecar-name materialisation findSidecarRef performs for its callers.
	scaleRef, _, _, ok := lookupSidecarRef(index, spec, weightName, "scales")
	if !ok {
		return JANGPackedProjectionTensor{}, core.NewError("mlx: MiniMax M2 packed projection missing scales for " + spec.Name)
	}
	scales, err := cache.ReadRefValues(scaleRef)
	if err != nil {
		return JANGPackedProjectionTensor{}, core.E("minimax_m2.packed_projection", "read scales", err)
	}
	biasRef, _, _, ok := lookupSidecarRef(index, spec, weightName, "biases")
	if !ok {
		return JANGPackedProjectionTensor{}, core.NewError("mlx: MiniMax M2 packed projection missing biases for " + spec.Name)
	}
	biases, err := cache.ReadRefValues(biasRef)
	if err != nil {
		return JANGPackedProjectionTensor{}, core.E("minimax_m2.packed_projection", "read biases", err)
	}
	tensor := JANGPackedProjectionTensor{
		Descriptor: *spec.Packed,
		Packed:     packed,
		Scales:     scales,
		Biases:     biases,
	}
	if projBiasRef, _, ok := findProjectionBiasRef(index, spec, weightName); ok {
		tensor.Bias, err = cache.ReadRefValues(projBiasRef)
		if err != nil {
			return JANGPackedProjectionTensor{}, core.E("minimax_m2.packed_projection", "read projection bias", err)
		}
	}
	if err := jang.ValidatePackedTensor(tensor.Descriptor, tensor.Packed, tensor.Scales, tensor.Biases); err != nil {
		return JANGPackedProjectionTensor{}, err
	}
	return tensor, nil
}

func resolveSkeletonTensor(index safetensors.Index, spec TensorSpec, candidates func(*TensorSpec) []string) (ResolvedTensor, error) {
	if spec.Name == "" {
		return ResolvedTensor{}, core.NewError("mlx: MiniMax M2 layer skeleton received empty tensor spec")
	}
	ref, name, ok := findSafetensorRef(index, candidates(&spec))
	if !ok {
		return ResolvedTensor{}, core.NewError("mlx: MiniMax M2 layer skeleton missing tensor: " + spec.Name)
	}
	resolved := ResolvedTensor{
		Name:         name,
		Role:         spec.Role,
		Layer:        spec.Layer,
		DType:        ref.DType,
		Shape:        core.SliceClone(ref.Shape),
		LogicalShape: core.SliceClone(spec.Shape),
	}
	if spec.Packed != nil {
		if !packedDType(ref.DType) {
			return ResolvedTensor{}, core.NewError(core.Sprintf("mlx: MiniMax M2 layer skeleton %s dtype %s is not packed U8", name, ref.DType))
		}
		resolved.PackedBytes = spec.Packed.PackedBytes
		if int(ref.ByteLen) != spec.Packed.PackedBytes || ref.Elements != spec.Packed.PackedBytes {
			return ResolvedTensor{}, core.NewError(core.Sprintf("mlx: MiniMax M2 layer skeleton %s packed bytes %d/%d, expected %d", name, ref.ByteLen, ref.Elements, spec.Packed.PackedBytes))
		}
		return resolved, nil
	}
	if !floatDType(ref.DType) {
		return ResolvedTensor{}, core.NewError(core.Sprintf("mlx: MiniMax M2 layer skeleton %s dtype %s is not floating point", name, ref.DType))
	}
	if !sameUint64Slice(ref.Shape, spec.Shape) {
		return ResolvedTensor{}, core.NewError(core.Sprintf("mlx: MiniMax M2 layer skeleton %s shape %+v, expected %+v", name, ref.Shape, spec.Shape))
	}
	return resolved, nil
}

func packedWeightCandidates(spec *TensorSpec) []string {
	bases := make([]string, 0, 1+len(spec.Aliases))
	bases = append(bases, spec.Name)
	bases = append(bases, spec.Aliases...)
	out := make([]string, 0, len(bases)*4)
	for _, base := range bases {
		out = append(out, base, base+".packed", base+".qweight", trimWeightSuffix(base)+".qweight")
	}
	return out
}

func routerGateCandidates(spec *TensorSpec) []string {
	hasName := spec.Name != ""
	extra := 0
	if hasName {
		extra = 1
	}
	out := make([]string, 0, 1+len(spec.Aliases)+extra)
	out = append(out, spec.Name)
	out = append(out, spec.Aliases...)
	if hasName {
		out = append(out, trimWeightSuffix(spec.Name)+".gate")
	}
	return out
}

func routerBiasCandidates(spec *TensorSpec, layer int) []string {
	layerPrefix := core.Concat("model.layers.", core.Itoa(layer), ".")
	// Build the non-empty candidate set directly into a single slice. The
	// previous form allocated a fixed names literal, then append-grew it with
	// spec.Aliases (a second backing alloc whenever an alias was present),
	// then filtered into a third slice — three allocs on the aliased path.
	// Pre-sizing for the 4 fixed shapes + aliases and guarding each push keeps
	// the same output order and non-empty filtering in one allocation.
	out := make([]string, 0, 4+len(spec.Aliases))
	appendIfNonEmpty(&out, spec.Name)
	appendIfNonEmpty(&out, core.Concat(layerPrefix, "block_sparse_moe.e_score_correction_bias"))
	appendIfNonEmpty(&out, core.Concat(layerPrefix, "mlp.e_score_correction_bias"))
	appendIfNonEmpty(&out, core.Concat(layerPrefix, "block_sparse_moe.gate.e_score_correction_bias"))
	for _, alias := range spec.Aliases {
		appendIfNonEmpty(&out, alias)
	}
	return out
}

// appendIfNonEmpty appends name to out only when it is non-empty, preserving
// the empty-name filtering that routerBiasCandidates applied.
func appendIfNonEmpty(out *[]string, name string) {
	if name != "" {
		*out = append(*out, name)
	}
}

// findProjectionBiasRef inlines the projectionBiasCandidates fan-out +
// findSafetensorRef loop. Projection bias is typically absent for
// MiniMax M2 packed experts, so the common case is a full miss — but
// the per-projection path still pays for the candidate slice every
// time. The inline path lets us skip the slice + per-string-concat
// allocs on every load whether the bias resolves or not (a miss only
// walks the existence-check probes; a hit returns immediately).
//
//	ref, name, ok := findProjectionBiasRef(index, spec, weightName)
func findProjectionBiasRef(index safetensors.Index, spec *TensorSpec, weightName string) (safetensors.TensorRef, string, bool) {
	var scratch [nameProbeScratch]byte
	buf := scratch[:0]
	if ref, name, ok := tryProjectionBiasName(index, buf, weightName); ok {
		return ref, name, true
	}
	if spec.Name != weightName {
		if ref, name, ok := tryProjectionBiasName(index, buf, spec.Name); ok {
			return ref, name, true
		}
	}
	for _, alias := range spec.Aliases {
		if ref, name, ok := tryProjectionBiasName(index, buf, alias); ok {
			return ref, name, true
		}
	}
	return safetensors.TensorRef{}, "", false
}

// tryProjectionBiasName probes the three projection-bias name shapes
// (trim(name)+".bias", name+".proj_bias", trim(name)+".proj_bias")
// against the safetensors index and returns on the first hit. Candidates
// are built into the caller's reusable scratch buffer and probed via the
// compiler's no-alloc map[string([]byte)] lookup special-case, so the
// common full-miss path builds zero throwaway candidate strings. The
// returned name on a hit is materialised once (output, retained by the
// caller's loader/error path).
func tryProjectionBiasName(index safetensors.Index, buf []byte, name string) (safetensors.TensorRef, string, bool) {
	trimmed := trimWeightSuffix(name)
	if ref, ok := index.Tensors[string(appendConcat(buf, trimmed, ".bias"))]; ok {
		return ref, trimmed + ".bias", true
	}
	if ref, ok := index.Tensors[string(appendConcat(buf, name, ".proj_bias"))]; ok {
		return ref, name + ".proj_bias", true
	}
	if trimmed != name {
		if ref, ok := index.Tensors[string(appendConcat(buf, trimmed, ".proj_bias"))]; ok {
			return ref, trimmed + ".proj_bias", true
		}
	}
	return safetensors.TensorRef{}, "", false
}

// findPackedWeightRef inlines the packedWeightCandidates fan-out +
// findSafetensorRef loop so common-case hits return before materialising
// the full candidate slice. Mirrors findSidecarRef for the canonical
// weight tensor — the first probe is spec.Name itself, the canonical
// production-checkpoint layout. resolveSkeletonTensor still routes
// through packedWeightCandidates because the function-as-arg shape
// there serves all skeleton roles uniformly; only loadPackedProjection
// (the per-expert hot path) routes through this inline variant.
//
//	ref, name, ok := findPackedWeightRef(index, spec)
func findPackedWeightRef(index safetensors.Index, spec *TensorSpec) (safetensors.TensorRef, string, bool) {
	var scratch [nameProbeScratch]byte
	buf := scratch[:0]
	if ref, name, ok := tryPackedWeightName(index, buf, spec.Name); ok {
		return ref, name, true
	}
	for _, alias := range spec.Aliases {
		if ref, name, ok := tryPackedWeightName(index, buf, alias); ok {
			return ref, name, true
		}
	}
	return safetensors.TensorRef{}, "", false
}

// tryPackedWeightName probes the four packed-weight name shapes
// (base, base+".packed", base+".qweight", trim(base)+".qweight")
// against the safetensors index and returns on the first hit. Suffixed
// candidates are built into the caller's reusable scratch buffer and
// probed via the no-alloc map[string([]byte)] lookup, so a full miss
// builds zero throwaway strings. The canonical first probe is base
// itself (the production-checkpoint layout): it indexes the original
// string directly and returns it as-is, so a first-probe hit stays
// allocation-free. Only a suffixed hit materialises a new name (output).
func tryPackedWeightName(index safetensors.Index, buf []byte, base string) (safetensors.TensorRef, string, bool) {
	if ref, ok := index.Tensors[base]; ok {
		return ref, base, true
	}
	if ref, ok := index.Tensors[string(appendConcat(buf, base, ".packed"))]; ok {
		return ref, base + ".packed", true
	}
	if ref, ok := index.Tensors[string(appendConcat(buf, base, ".qweight"))]; ok {
		return ref, base + ".qweight", true
	}
	if trimmed := trimWeightSuffix(base); trimmed != base {
		if ref, ok := index.Tensors[string(appendConcat(buf, trimmed, ".qweight"))]; ok {
			return ref, trimmed + ".qweight", true
		}
	}
	return safetensors.TensorRef{}, "", false
}

// sidecarForm identifies which of the three sidecar-name shapes a probe
// matched, so the matched name can be reconstructed once on a hit without
// the probe materialising it eagerly. The miss path never builds a name.
type sidecarForm uint8

const (
	sidecarFormNone       sidecarForm = iota
	sidecarFormDot                    // base + "." + sidecar
	sidecarFormTrimDot                // trim(base) + "." + sidecar
	sidecarFormUnderscore             // base + "_" + sidecar
)

// lookupSidecarRef walks the four candidate bases (weightName, its packed-trim,
// spec.Name, then aliases) and, on the first hit, reports the matched ref plus
// the base+form that resolved it — but builds no name string. Production
// sidecar resolution (loadPackedProjection) discards the name and uses only
// the ref, so this no-name core runs twice per packed projection × every
// layer×expert without the per-hit name allocation the name-returning
// findSidecarRef paid. The base/form return lets the name-returning wrapper
// reconstruct the exact matched name lazily, keeping the four-base fan-out and
// three-form probe each defined once.
//
//	ref, _, _, ok := lookupSidecarRef(index, spec, weightName, "scales")
func lookupSidecarRef(index safetensors.Index, spec *TensorSpec, weightName, sidecar string) (safetensors.TensorRef, string, sidecarForm, bool) {
	// The sidecar separator ("." / "_") is folded into the per-candidate
	// build below rather than pre-concatenated into dot/underscore strings,
	// removing the two per-call separator allocations the old form paid
	// before the first probe.
	var scratch [nameProbeScratch]byte
	buf := scratch[:0]
	if ref, form, ok := trySidecarRef(index, buf, weightName, sidecar); ok {
		return ref, weightName, form, true
	}
	if trimmed := trimPackedSuffix(weightName); trimmed != weightName {
		if ref, form, ok := trySidecarRef(index, buf, trimmed, sidecar); ok {
			return ref, trimmed, form, true
		}
	}
	if ref, form, ok := trySidecarRef(index, buf, spec.Name, sidecar); ok {
		return ref, spec.Name, form, true
	}
	for _, alias := range spec.Aliases {
		if ref, form, ok := trySidecarRef(index, buf, alias, sidecar); ok {
			return ref, alias, form, true
		}
	}
	return safetensors.TensorRef{}, "", sidecarFormNone, false
}

// findSidecarRef is the name-returning wrapper over lookupSidecarRef. It
// resolves the ref through the shared no-name fan-out and materialises the
// matched name once on a hit. The name is part of the resolver contract
// (asserted by tests) but is discarded by the production loader, which calls
// lookupSidecarRef directly to skip this allocation.
//
//	ref, name, ok := findSidecarRef(index, spec, weightName, "scales")
func findSidecarRef(index safetensors.Index, spec *TensorSpec, weightName, sidecar string) (safetensors.TensorRef, string, bool) {
	ref, base, form, ok := lookupSidecarRef(index, spec, weightName, sidecar)
	if !ok {
		return safetensors.TensorRef{}, "", false
	}
	return ref, sidecarName(base, sidecar, form), true
}

// sidecarName reconstructs the matched sidecar tensor name from the base and
// form a probe resolved, materialising the single output string a hit retains.
func sidecarName(base, sidecar string, form sidecarForm) string {
	switch form {
	case sidecarFormTrimDot:
		return trimWeightSuffix(base) + "." + sidecar
	case sidecarFormUnderscore:
		return base + "_" + sidecar
	default:
		return base + "." + sidecar
	}
}

// trySidecarRef probes the three sidecar-name shapes (name+"."+sidecar,
// trim(name)+"."+sidecar, name+"_"+sidecar) against the safetensors index
// and returns on the first hit, reporting which shape matched. Candidates
// are built into the caller's reusable scratch buffer and probed via the
// no-alloc map[string([]byte)] lookup, so neither a hit nor a miss builds a
// throwaway candidate or output string — the caller reconstructs the name
// from the form only when it needs it.
func trySidecarRef(index safetensors.Index, buf []byte, name, sidecar string) (safetensors.TensorRef, sidecarForm, bool) {
	if ref, ok := index.Tensors[string(appendConcat3(buf, name, ".", sidecar))]; ok {
		return ref, sidecarFormDot, true
	}
	if trimmed := trimWeightSuffix(name); trimmed != name {
		if ref, ok := index.Tensors[string(appendConcat3(buf, trimmed, ".", sidecar))]; ok {
			return ref, sidecarFormTrimDot, true
		}
	}
	if ref, ok := index.Tensors[string(appendConcat3(buf, name, "_", sidecar))]; ok {
		return ref, sidecarFormUnderscore, true
	}
	return safetensors.TensorRef{}, sidecarFormNone, false
}

func findSafetensorRef(index safetensors.Index, candidates []string) (safetensors.TensorRef, string, bool) {
	for _, name := range candidates {
		ref, ok := index.Tensors[name]
		if ok {
			return ref, name, true
		}
	}
	return safetensors.TensorRef{}, "", false
}

// nameProbeScratch sizes the stack-resident byte buffer the find*Ref
// helpers reuse when probing the safetensors index for candidate tensor
// names. The longest MiniMax M2 candidate (a fully-qualified per-expert
// projection name plus the longest sidecar/qweight suffix) is well under
// this bound, so candidate builds never grow the buffer and the array
// stays on the caller's stack — the probe lookups allocate nothing.
const nameProbeScratch = 256

// appendConcat appends a+b into buf[:0] and returns the filled slice for a
// no-alloc map[string([]byte)] index probe. The buffer keeps its backing
// array (caller stack), so reusing it across probes builds zero throwaway
// strings; materialise a returned name with a+b only on a hit.
//
//	if ref, ok := index.Tensors[string(appendConcat(buf, base, ".packed"))]; ok { ... }
func appendConcat(buf []byte, a, b string) []byte {
	buf = append(buf[:0], a...)
	return append(buf, b...)
}

// appendConcat3 is appendConcat for a three-part candidate (a+b+c), used
// for the sidecar shapes where the separator (b) is folded in rather than
// pre-concatenated onto the sidecar word.
//
//	if ref, ok := index.Tensors[string(appendConcat3(buf, name, ".", sidecar))]; ok { ... }
func appendConcat3(buf []byte, a, b, c string) []byte {
	buf = append(buf[:0], a...)
	buf = append(buf, b...)
	return append(buf, c...)
}

func trimWeightSuffix(name string) string {
	if core.HasSuffix(name, ".weight") {
		return name[:len(name)-len(".weight")]
	}
	return name
}

var packedSuffixes = [...]string{".packed", ".qweight"}

// metaMinimaxM2 is the architecture-tag map attached to every probe.Event
// emitted by this package. The probe contract treats Meta as read-only on
// the publish path (recorder/exporter call cloneMeta before storing), so a
// shared sentinel removes one map alloc per emitted event.
//
//	event.Meta = metaMinimaxM2
var metaMinimaxM2 = map[string]string{"architecture": "minimax_m2"}

// attentionSkeletonRoles is the fixed list of attention projection roles
// resolved by BuildLayerForwardSkeleton. Lifted to a package-level array
// so the role loop doesn't allocate a fresh 4-elem slice per call.
//
//	for _, role := range attentionSkeletonRoles { ... }
var attentionSkeletonRoles = [...]TensorRole{
	TensorRoleAttentionQ,
	TensorRoleAttentionK,
	TensorRoleAttentionV,
	TensorRoleAttentionO,
}

func trimPackedSuffix(name string) string {
	for _, suffix := range packedSuffixes {
		if core.HasSuffix(name, suffix) {
			return name[:len(name)-len(suffix)]
		}
	}
	return name
}
