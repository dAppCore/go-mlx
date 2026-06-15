// SPDX-Licence-Identifier: EUPL-1.2

package m2

import (
	core "dappco.re/go"
	"dappco.re/go/inference/quant/jang"
	"dappco.re/go/mlx/probe"
	"dappco.re/go/mlx/safetensors"
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
	out := make(map[int]PackedExpertWeights, len(expertIDs))
	for _, expertID := range uniqueExpertIDs(expertIDs) {
		specs, err := plan.LayerTensorSpecs(layer, expertID)
		if err != nil {
			return nil, err
		}
		gateSpec := findTensorSpec(specs, TensorRoleExpertGate)
		gate, err := loadPackedProjection(index, &gateSpec)
		if err != nil {
			return nil, core.E("minimax_m2.packed_experts", core.Sprintf("expert %d gate_proj", expertID), err)
		}
		upSpec := findTensorSpec(specs, TensorRoleExpertUp)
		up, err := loadPackedProjection(index, &upSpec)
		if err != nil {
			return nil, core.E("minimax_m2.packed_experts", core.Sprintf("expert %d up_proj", expertID), err)
		}
		downSpec := findTensorSpec(specs, TensorRoleExpertDown)
		down, err := loadPackedProjection(index, &downSpec)
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
	weight, err := safetensors.ReadRefValues(ref)
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
		router.Bias, err = safetensors.ReadRefValues(biasRef)
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

func loadPackedProjection(index safetensors.Index, spec *TensorSpec) (JANGPackedProjectionTensor, error) {
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
	packed, err := safetensors.ReadRefRaw(weightRef)
	if err != nil {
		return JANGPackedProjectionTensor{}, err
	}
	scaleRef, _, ok := findSidecarRef(index, spec, weightName, "scales")
	if !ok {
		return JANGPackedProjectionTensor{}, core.NewError("mlx: MiniMax M2 packed projection missing scales for " + spec.Name)
	}
	scales, err := safetensors.ReadRefValues(scaleRef)
	if err != nil {
		return JANGPackedProjectionTensor{}, core.E("minimax_m2.packed_projection", "read scales", err)
	}
	biasRef, _, ok := findSidecarRef(index, spec, weightName, "biases")
	if !ok {
		return JANGPackedProjectionTensor{}, core.NewError("mlx: MiniMax M2 packed projection missing biases for " + spec.Name)
	}
	biases, err := safetensors.ReadRefValues(biasRef)
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
		tensor.Bias, err = safetensors.ReadRefValues(projBiasRef)
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
	names := []string{
		spec.Name,
		core.Concat(layerPrefix, "block_sparse_moe.e_score_correction_bias"),
		core.Concat(layerPrefix, "mlp.e_score_correction_bias"),
		core.Concat(layerPrefix, "block_sparse_moe.gate.e_score_correction_bias"),
	}
	names = append(names, spec.Aliases...)
	out := make([]string, 0, len(names))
	for _, name := range names {
		if name != "" {
			out = append(out, name)
		}
	}
	return out
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
	if ref, name, ok := tryProjectionBiasName(index, weightName); ok {
		return ref, name, true
	}
	if spec.Name != weightName {
		if ref, name, ok := tryProjectionBiasName(index, spec.Name); ok {
			return ref, name, true
		}
	}
	for _, alias := range spec.Aliases {
		if ref, name, ok := tryProjectionBiasName(index, alias); ok {
			return ref, name, true
		}
	}
	return safetensors.TensorRef{}, "", false
}

// tryProjectionBiasName probes the three projection-bias name shapes
// (trim(name)+".bias", name+".proj_bias", trim(name)+".proj_bias")
// against the safetensors index and returns on the first hit. Hoisted
// out so the call stays a plain dispatch.
func tryProjectionBiasName(index safetensors.Index, name string) (safetensors.TensorRef, string, bool) {
	trimmed := trimWeightSuffix(name)
	candidate := trimmed + ".bias"
	if ref, ok := index.Tensors[candidate]; ok {
		return ref, candidate, true
	}
	candidate = name + ".proj_bias"
	if ref, ok := index.Tensors[candidate]; ok {
		return ref, candidate, true
	}
	if trimmed != name {
		candidate = trimmed + ".proj_bias"
		if ref, ok := index.Tensors[candidate]; ok {
			return ref, candidate, true
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
	if ref, name, ok := tryPackedWeightName(index, spec.Name); ok {
		return ref, name, true
	}
	for _, alias := range spec.Aliases {
		if ref, name, ok := tryPackedWeightName(index, alias); ok {
			return ref, name, true
		}
	}
	return safetensors.TensorRef{}, "", false
}

// tryPackedWeightName probes the four packed-weight name shapes
// (base, base+".packed", base+".qweight", trim(base)+".qweight")
// against the safetensors index and returns on the first hit. Hoisted
// out so the call stays a plain dispatch.
func tryPackedWeightName(index safetensors.Index, base string) (safetensors.TensorRef, string, bool) {
	if ref, ok := index.Tensors[base]; ok {
		return ref, base, true
	}
	candidate := base + ".packed"
	if ref, ok := index.Tensors[candidate]; ok {
		return ref, candidate, true
	}
	candidate = base + ".qweight"
	if ref, ok := index.Tensors[candidate]; ok {
		return ref, candidate, true
	}
	if trimmed := trimWeightSuffix(base); trimmed != base {
		candidate = trimmed + ".qweight"
		if ref, ok := index.Tensors[candidate]; ok {
			return ref, candidate, true
		}
	}
	return safetensors.TensorRef{}, "", false
}

// findSidecarRef inlines the sidecarCandidates fan-out + findSafetensorRef
// loop so common-case hits return before materialising the full candidate
// slice. Sidecar resolution happens twice per packed projection (scales,
// biases) and each layer×expert pass walks through many projections, so
// shaving the slice + per-string-concat allocs adds up at model load. The
// first-hit early return mirrors the production checkpoint shape where
// weightName+"."+sidecar is the canonical layout — the alternatives only
// fire for legacy or aliased checkpoints.
//
//	ref, name, ok := findSidecarRef(index, spec, weightName, "scales")
func findSidecarRef(index safetensors.Index, spec *TensorSpec, weightName, sidecar string) (safetensors.TensorRef, string, bool) {
	dot := "." + sidecar
	underscore := "_" + sidecar
	if ref, name, ok := trySidecarName(index, weightName, dot, underscore); ok {
		return ref, name, true
	}
	if trimmed := trimPackedSuffix(weightName); trimmed != weightName {
		if ref, name, ok := trySidecarName(index, trimmed, dot, underscore); ok {
			return ref, name, true
		}
	}
	if ref, name, ok := trySidecarName(index, spec.Name, dot, underscore); ok {
		return ref, name, true
	}
	for _, alias := range spec.Aliases {
		if ref, name, ok := trySidecarName(index, alias, dot, underscore); ok {
			return ref, name, true
		}
	}
	return safetensors.TensorRef{}, "", false
}

// trySidecarName probes the three sidecar-name shapes (name+dot,
// trim(name)+dot, name+underscore) against the safetensors index and
// returns on the first hit. Hoisted out of findSidecarRef so the call
// is a plain function dispatch rather than a closure (which would
// escape to the heap and undo the alloc win).
func trySidecarName(index safetensors.Index, name, dot, underscore string) (safetensors.TensorRef, string, bool) {
	candidate := name + dot
	if ref, ok := index.Tensors[candidate]; ok {
		return ref, candidate, true
	}
	if trimmed := trimWeightSuffix(name); trimmed != name {
		candidate = trimmed + dot
		if ref, ok := index.Tensors[candidate]; ok {
			return ref, candidate, true
		}
	}
	candidate = name + underscore
	if ref, ok := index.Tensors[candidate]; ok {
		return ref, candidate, true
	}
	return safetensors.TensorRef{}, "", false
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
