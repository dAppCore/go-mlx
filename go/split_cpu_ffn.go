// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"math"
	"strconv"
	"sync"

	core "dappco.re/go"
	infjang "dappco.re/go/inference/quant/jang"
	"dappco.re/go/mlx/model"
	mp "dappco.re/go/mlx/pack"
	"dappco.re/go/mlx/safetensors"
)

// CPUSplitFFNConfig configures the CPU-side FFN executor.
type CPUSplitFFNConfig struct {
	// MaxCachedLayers limits retained CPU FFN layers. 0 keeps all loaded layers;
	// a negative value disables caching and reloads layer tensors every call.
	MaxCachedLayers int
}

// CPUSplitFFNMemoryReport describes CPU FFN residency for live layers or a
// preflight cache estimate.
type CPUSplitFFNMemoryReport struct {
	Estimated             bool    `json:"estimated,omitempty"`
	TotalLayers           int     `json:"total_layers,omitempty"`
	LoadedLayers          int     `json:"loaded_layers"`
	LayerLoads            int     `json:"layer_loads"`
	EvictedLayers         int     `json:"evicted_layers"`
	CacheLimit            int     `json:"cache_limit"`
	CacheDisabled         bool    `json:"cache_disabled,omitempty"`
	DenseProjections      int     `json:"dense_projections"`
	PackedProjections     int     `json:"packed_projections"`
	LayerNormBytes        int64   `json:"layer_norm_bytes"`
	ProjectionBiasBytes   int64   `json:"projection_bias_bytes"`
	DenseProjectionBytes  int64   `json:"dense_projection_bytes"`
	PackedProjectionBytes int64   `json:"packed_projection_bytes"`
	PackedSidecarBytes    int64   `json:"packed_sidecar_bytes"`
	ResidentBytes         int64   `json:"resident_bytes"`
	PeakResidentBytes     int64   `json:"peak_resident_bytes"`
	DenseEquivalentBytes  int64   `json:"dense_equivalent_bytes"`
	SavedBytes            int64   `json:"saved_bytes"`
	ResidentRatio         float64 `json:"resident_ratio,omitempty"`
}

// CPUSplitFFNOption configures LoadCPUSplitFFNExecutor.
type CPUSplitFFNOption func(*CPUSplitFFNConfig)

// WithCPUSplitFFNMaxCachedLayers limits how many FFN layers stay in RAM.
func WithCPUSplitFFNMaxCachedLayers(max int) CPUSplitFFNOption {
	return func(cfg *CPUSplitFFNConfig) {
		cfg.MaxCachedLayers = max
	}
}

// CPUSplitFFNExecutor runs omitted Qwen-style SwiGLU FFN layers on CPU.
type CPUSplitFFNExecutor struct {
	sourcePath string
	index      safetensors.Index
	cfg        cpuSplitQwenConfig
	cacheCfg   CPUSplitFFNConfig

	mu         sync.Mutex
	layerCache map[int]cpuSplitFFNLayer
	cacheOrder []int
	stats      cpuSplitFFNMemoryStats
}

type cpuSplitFFNMemoryStats struct {
	layerLoads        int
	evictedLayers     int
	peakResidentBytes int64
}

type cpuSplitQwenConfig struct {
	ModelType          string                      `json:"model_type"`
	HiddenSize         int                         `json:"hidden_size"`
	IntermediateSize   int                         `json:"intermediate_size"`
	NumHiddenLayers    int                         `json:"num_hidden_layers"`
	RMSNormEps         float32                     `json:"rms_norm_eps"`
	Quantization       *cpuSplitQuantizationConfig `json:"quantization,omitempty"`
	QuantizationConfig *cpuSplitQuantizationConfig `json:"quantization_config,omitempty"`
	PackedGroupSize    int                         `json:"-"`
	PackedBits         int                         `json:"-"`
	JANG               *infjang.Info               `json:"-"`
}

type cpuSplitQuantizationConfig struct {
	Method      string `json:"method,omitempty"`
	Mode        string `json:"mode,omitempty"`
	GroupSize   int    `json:"group_size,omitempty"`
	Bits        int    `json:"bits,omitempty"`
	BitsDefault int    `json:"bits_default,omitempty"`
}

type cpuSplitFFNLayer struct {
	norm         []float32
	gate         []float32
	gatePacked   *cpuSplitPackedMatrix
	gateBias     []float32
	up           []float32
	upPacked     *cpuSplitPackedMatrix
	upBias       []float32
	down         []float32
	downPacked   *cpuSplitPackedMatrix
	downBias     []float32
	hidden       int
	intermediate int
}

type cpuSplitPackedMatrix struct {
	desc   infjang.PackedTensorDescriptor
	packed []byte
	scales []float32
	biases []float32
	rows   int
	cols   int
	// Hot-path mirrors of desc fields. The per-element value() lookup ran
	// hundreds of millions of times per layer; reading them off the struct
	// directly avoids the chase through desc.GroupSize / desc.Bits each call.
	groupSize int
	bits      int
	elements  uint64
}

const cpuSplitFloat32Bytes = int64(4)

func (report *CPUSplitFFNMemoryReport) addLayer(layer cpuSplitFFNLayer) {
	report.addDenseVectorBytes(int64(len(layer.norm)) * cpuSplitFloat32Bytes)
	biasBytes := int64(len(layer.gateBias)+len(layer.upBias)+len(layer.downBias)) * cpuSplitFloat32Bytes
	report.ProjectionBiasBytes += biasBytes
	report.ResidentBytes += biasBytes
	report.DenseEquivalentBytes += biasBytes
	report.addProjection(layer.gate, layer.gatePacked)
	report.addProjection(layer.up, layer.upPacked)
	report.addProjection(layer.down, layer.downPacked)
}

func (report *CPUSplitFFNMemoryReport) addDenseVectorBytes(bytes int64) {
	report.LayerNormBytes += bytes
	report.ResidentBytes += bytes
	report.DenseEquivalentBytes += bytes
}

func (report *CPUSplitFFNMemoryReport) addProjection(dense []float32, packed *cpuSplitPackedMatrix) {
	if packed != nil {
		report.PackedProjections++
		packedBytes := int64(len(packed.packed))
		sidecarBytes := int64(len(packed.scales)+len(packed.biases)) * cpuSplitFloat32Bytes
		equivalentBytes := int64(packed.rows*packed.cols) * cpuSplitFloat32Bytes
		report.PackedProjectionBytes += packedBytes
		report.PackedSidecarBytes += sidecarBytes
		report.ResidentBytes += packedBytes + sidecarBytes
		report.DenseEquivalentBytes += equivalentBytes
		return
	}
	if len(dense) == 0 {
		return
	}
	report.DenseProjections++
	bytes := int64(len(dense)) * cpuSplitFloat32Bytes
	report.DenseProjectionBytes += bytes
	report.ResidentBytes += bytes
	report.DenseEquivalentBytes += bytes
}

func (report *CPUSplitFFNMemoryReport) addReport(other CPUSplitFFNMemoryReport) {
	report.DenseProjections += other.DenseProjections
	report.PackedProjections += other.PackedProjections
	report.LayerNormBytes += other.LayerNormBytes
	report.ProjectionBiasBytes += other.ProjectionBiasBytes
	report.DenseProjectionBytes += other.DenseProjectionBytes
	report.PackedProjectionBytes += other.PackedProjectionBytes
	report.PackedSidecarBytes += other.PackedSidecarBytes
	report.ResidentBytes += other.ResidentBytes
	report.DenseEquivalentBytes += other.DenseEquivalentBytes
}

func (report *CPUSplitFFNMemoryReport) finalise() {
	if report.PeakResidentBytes < report.ResidentBytes {
		report.PeakResidentBytes = report.ResidentBytes
	}
	if report.DenseEquivalentBytes <= 0 {
		return
	}
	report.SavedBytes = report.DenseEquivalentBytes - report.ResidentBytes
	if report.SavedBytes < 0 {
		report.SavedBytes = 0
	}
	report.ResidentRatio = float64(report.ResidentBytes) / float64(report.DenseEquivalentBytes)
}

func applyCPUSplitFFNOptions(opts []CPUSplitFFNOption) CPUSplitFFNConfig {
	var cfg CPUSplitFFNConfig
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

// LoadCPUSplitFFNExecutor loads source-pack metadata for CPU FFN execution.
func LoadCPUSplitFFNExecutor(ctx context.Context, sourcePath string, opts ...CPUSplitFFNOption) (*CPUSplitFFNExecutor, error) {
	return loadCPUSplitFFNExecutor(ctx, sourcePath, applyCPUSplitFFNOptions(opts))
}

// EstimateCPUSplitFFNMemory estimates CPU FFN residency from source-pack
// metadata without loading layer tensors into the cache.
func EstimateCPUSplitFFNMemory(ctx context.Context, sourcePath string, opts ...CPUSplitFFNOption) (CPUSplitFFNMemoryReport, error) {
	executor, err := LoadCPUSplitFFNExecutor(ctx, sourcePath, opts...)
	if err != nil {
		return CPUSplitFFNMemoryReport{}, err
	}
	return executor.EstimateMemoryReport(ctx)
}

func loadCPUSplitFFNExecutor(ctx context.Context, sourcePath string, cfg CPUSplitFFNConfig) (*CPUSplitFFNExecutor, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if core.Trim(sourcePath) == "" {
		return nil, core.NewError("mlx: CPU split FFN executor requires a source model path")
	}
	source, err := model.Inspect(sourcePath)
	if err != nil {
		return nil, err
	}
	if source.Format != mp.ModelPackFormatSafetensors || len(source.WeightFiles) == 0 {
		return nil, core.NewError("mlx: CPU split FFN executor requires a safetensors source pack")
	}
	qwenCfg, err := readCPUSplitQwenConfig(source.Root)
	if err != nil {
		return nil, err
	}
	jangInfo, err := infjang.ReadConfig(source.Root)
	if err != nil {
		return nil, err
	}
	qwenCfg.applyJANGInfo(jangInfo)
	if qwenCfg.HiddenSize <= 0 || qwenCfg.IntermediateSize <= 0 || qwenCfg.NumHiddenLayers <= 0 {
		return nil, core.NewError("mlx: CPU split FFN executor requires hidden, intermediate, and layer counts")
	}
	index, err := safetensors.IndexFiles(source.WeightFiles)
	if err != nil {
		return nil, err
	}
	cacheHint := cfg.MaxCachedLayers
	if cacheHint <= 0 {
		// Unbounded cache: hint against layer count to avoid map regrows.
		cacheHint = qwenCfg.NumHiddenLayers
	}
	return &CPUSplitFFNExecutor{
		sourcePath: sourcePath,
		index:      index,
		cfg:        qwenCfg,
		cacheCfg:   cfg,
		layerCache: make(map[int]cpuSplitFFNLayer, cacheHint),
		cacheOrder: make([]int, 0, cacheHint),
	}, nil
}

func readCPUSplitQwenConfig(root string) (cpuSplitQwenConfig, error) {
	read := core.ReadFile(core.PathJoin(root, "config.json"))
	if !read.OK {
		return cpuSplitQwenConfig{}, modelSliceResultError(read)
	}
	var raw struct {
		ModelType          string                      `json:"model_type"`
		HiddenSize         int                         `json:"hidden_size"`
		IntermediateSize   int                         `json:"intermediate_size"`
		NumHiddenLayers    int                         `json:"num_hidden_layers"`
		RMSNormEps         float32                     `json:"rms_norm_eps"`
		Quantization       *cpuSplitQuantizationConfig `json:"quantization"`
		QuantizationConfig *cpuSplitQuantizationConfig `json:"quantization_config"`
		TextConfig         *cpuSplitQwenConfig         `json:"text_config"`
	}
	if result := core.JSONUnmarshal(read.Value.([]byte), &raw); !result.OK {
		return cpuSplitQwenConfig{}, modelSliceResultError(result)
	}
	cfg := cpuSplitQwenConfig{
		ModelType:          raw.ModelType,
		HiddenSize:         raw.HiddenSize,
		IntermediateSize:   raw.IntermediateSize,
		NumHiddenLayers:    raw.NumHiddenLayers,
		RMSNormEps:         raw.RMSNormEps,
		Quantization:       raw.Quantization,
		QuantizationConfig: raw.QuantizationConfig,
	}
	if raw.TextConfig != nil {
		cfg = mergeCPUSplitQwenConfig(cfg, *raw.TextConfig)
	}
	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = 1e-6
	}
	cfg.applyQuantizationHints()
	return cfg, nil
}

func mergeCPUSplitQwenConfig(top, text cpuSplitQwenConfig) cpuSplitQwenConfig {
	if text.ModelType == "" {
		text.ModelType = top.ModelType
	}
	if text.HiddenSize == 0 {
		text.HiddenSize = top.HiddenSize
	}
	if text.IntermediateSize == 0 {
		text.IntermediateSize = top.IntermediateSize
	}
	if text.NumHiddenLayers == 0 {
		text.NumHiddenLayers = top.NumHiddenLayers
	}
	if text.RMSNormEps == 0 {
		text.RMSNormEps = top.RMSNormEps
	}
	if text.Quantization == nil {
		text.Quantization = top.Quantization
	}
	if text.QuantizationConfig == nil {
		text.QuantizationConfig = top.QuantizationConfig
	}
	return text
}

func (cfg *cpuSplitQwenConfig) applyQuantizationHints() {
	cfg.applyQuantizationHint(cfg.Quantization)
	cfg.applyQuantizationHint(cfg.QuantizationConfig)
}

func (cfg *cpuSplitQwenConfig) applyQuantizationHint(quant *cpuSplitQuantizationConfig) {
	if quant == nil {
		return
	}
	if cfg.PackedGroupSize <= 0 && quant.GroupSize > 0 {
		cfg.PackedGroupSize = quant.GroupSize
	}
	if cfg.PackedBits <= 0 {
		cfg.PackedBits = cpuSplitFirstPositive(quant.BitsDefault, quant.Bits)
	}
}

func (cfg *cpuSplitQwenConfig) applyJANGInfo(info *infjang.Info) {
	if info == nil {
		return
	}
	cfg.JANG = info
	if info.GroupSize > 0 {
		cfg.PackedGroupSize = info.GroupSize
	}
	if bits := cpuSplitFirstPositive(info.BitsDefault, infjang.ProfileBits(info.Profile)); bits > 0 {
		cfg.PackedBits = bits
	}
}

// ForwardFFN runs one FFN layer on CPU.
func (executor *CPUSplitFFNExecutor) ForwardFFN(ctx context.Context, req SplitFFNRequest) (SplitFFNResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return SplitFFNResult{}, err
	}
	if executor == nil {
		return SplitFFNResult{}, core.NewError("mlx: CPU split FFN executor is nil")
	}
	if req.Layer < 0 || req.Layer >= executor.cfg.NumHiddenLayers {
		return SplitFFNResult{}, core.Errorf("mlx: CPU split FFN layer %d out of range", req.Layer)
	}
	if len(req.Hidden) == 0 || len(req.Hidden)%executor.cfg.HiddenSize != 0 {
		return SplitFFNResult{}, core.NewError("mlx: CPU split FFN hidden state does not match model hidden size")
	}
	layer, err := executor.layer(ctx, req.Layer)
	if err != nil {
		return SplitFFNResult{}, err
	}
	// Hoist hidden size + eps out of the row loop — the original code reread
	// executor.cfg.HiddenSize three times per row and executor.cfg.RMSNormEps
	// once per row by chasing the struct fields through the call site.
	hiddenSize := executor.cfg.HiddenSize
	eps := executor.cfg.RMSNormEps
	hidden := req.Hidden
	out := make([]float32, len(hidden))
	rows := len(hidden) / hiddenSize
	normed := make([]float32, layer.hidden)
	activated := make([]float32, layer.intermediate)
	for row := 0; row < rows; row++ {
		if err := ctx.Err(); err != nil {
			return SplitFFNResult{}, err
		}
		start := row * hiddenSize
		end := start + hiddenSize
		cpuSplitForwardDenseRow(hidden[start:end], out[start:end], layer, eps, normed, activated)
	}
	return SplitFFNResult{Hidden: out}, nil
}

// MemoryReport returns the currently resident CPU FFN layer memory. With cache
// disabled, this intentionally reports no resident layers after a call returns.
func (executor *CPUSplitFFNExecutor) MemoryReport() CPUSplitFFNMemoryReport {
	if executor == nil {
		return CPUSplitFFNMemoryReport{}
	}
	executor.mu.Lock()
	defer executor.mu.Unlock()

	report := CPUSplitFFNMemoryReport{
		TotalLayers:       executor.cfg.NumHiddenLayers,
		LoadedLayers:      len(executor.layerCache),
		LayerLoads:        executor.stats.layerLoads,
		EvictedLayers:     executor.stats.evictedLayers,
		CacheLimit:        executor.cacheCfg.MaxCachedLayers,
		CacheDisabled:     executor.cacheCfg.MaxCachedLayers < 0,
		PeakResidentBytes: executor.stats.peakResidentBytes,
	}
	for _, layer := range executor.layerCache {
		report.addLayer(layer)
	}
	report.finalise()
	return report
}

// EstimateMemoryReport predicts CPU FFN residency for one full pass through all
// layers using only safetensor metadata. It does not populate the layer cache.
func (executor *CPUSplitFFNExecutor) EstimateMemoryReport(ctx context.Context) (CPUSplitFFNMemoryReport, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return CPUSplitFFNMemoryReport{}, err
	}
	if executor == nil {
		return CPUSplitFFNMemoryReport{}, core.NewError("mlx: CPU split FFN executor is nil")
	}
	report := CPUSplitFFNMemoryReport{
		Estimated:     true,
		TotalLayers:   executor.cfg.NumHiddenLayers,
		CacheLimit:    executor.cacheCfg.MaxCachedLayers,
		CacheDisabled: executor.cacheCfg.MaxCachedLayers < 0,
	}
	layerReports := make([]CPUSplitFFNMemoryReport, 0, executor.cfg.NumHiddenLayers)
	for layer := 0; layer < executor.cfg.NumHiddenLayers; layer++ {
		if err := ctx.Err(); err != nil {
			return CPUSplitFFNMemoryReport{}, err
		}
		layerReport, err := executor.estimateLayerMemory(layer)
		if err != nil {
			return CPUSplitFFNMemoryReport{}, err
		}
		layerReports = append(layerReports, layerReport)
	}

	max := executor.cacheCfg.MaxCachedLayers
	report.LayerLoads = len(layerReports)
	if max < 0 {
		for _, layerReport := range layerReports {
			if layerReport.ResidentBytes > report.PeakResidentBytes {
				report.PeakResidentBytes = layerReport.ResidentBytes
			}
		}
		report.finalise()
		return report, nil
	}

	residentCap := len(layerReports)
	if max > 0 && max < residentCap {
		residentCap = max
	}
	resident := make([]CPUSplitFFNMemoryReport, 0, residentCap)
	var currentBytes int64
	for _, layerReport := range layerReports {
		resident = append(resident, layerReport)
		currentBytes += layerReport.ResidentBytes
		if max > 0 && len(resident) > max {
			currentBytes -= resident[0].ResidentBytes
			resident = resident[1:]
			report.EvictedLayers++
		}
		if currentBytes > report.PeakResidentBytes {
			report.PeakResidentBytes = currentBytes
		}
	}
	report.LoadedLayers = len(resident)
	for _, layerReport := range resident {
		report.addReport(layerReport)
	}
	report.finalise()
	return report, nil
}

func (executor *CPUSplitFFNExecutor) layer(ctx context.Context, layer int) (cpuSplitFFNLayer, error) {
	executor.mu.Lock()
	if cached, ok := executor.layerCache[layer]; ok && executor.cacheCfg.MaxCachedLayers >= 0 {
		executor.mu.Unlock()
		return cached, nil
	}
	executor.mu.Unlock()

	loaded, err := executor.loadLayer(ctx, layer)
	if err != nil {
		return cpuSplitFFNLayer{}, err
	}
	if executor.cacheCfg.MaxCachedLayers < 0 {
		transient := cpuSplitFFNLayerResidentBytes(loaded)
		executor.mu.Lock()
		executor.stats.layerLoads++
		executor.updatePeakResidentBytesLocked(transient)
		executor.mu.Unlock()
		return loaded, nil
	}
	executor.mu.Lock()
	defer executor.mu.Unlock()
	if cached, ok := executor.layerCache[layer]; ok {
		return cached, nil
	}
	executor.stats.layerLoads++
	executor.layerCache[layer] = loaded
	executor.cacheOrder = append(executor.cacheOrder, layer)
	executor.stats.evictedLayers += executor.evictLocked()
	executor.updatePeakResidentBytesLocked(executor.residentBytesLocked())
	return loaded, nil
}

func (executor *CPUSplitFFNExecutor) evictLocked() int {
	max := executor.cacheCfg.MaxCachedLayers
	if max <= 0 {
		return 0
	}
	evicted := 0
	for len(executor.cacheOrder) > max {
		layer := executor.cacheOrder[0]
		executor.cacheOrder = executor.cacheOrder[1:]
		delete(executor.layerCache, layer)
		evicted++
	}
	return evicted
}

func (executor *CPUSplitFFNExecutor) residentBytesLocked() int64 {
	var bytes int64
	for _, layer := range executor.layerCache {
		bytes += cpuSplitFFNLayerResidentBytes(layer)
	}
	return bytes
}

func (executor *CPUSplitFFNExecutor) updatePeakResidentBytesLocked(bytes int64) {
	if bytes > executor.stats.peakResidentBytes {
		executor.stats.peakResidentBytes = bytes
	}
}

func cpuSplitFFNLayerResidentBytes(layer cpuSplitFFNLayer) int64 {
	bytes := int64(len(layer.norm)+len(layer.gateBias)+len(layer.upBias)+len(layer.downBias)) * cpuSplitFloat32Bytes
	bytes += cpuSplitProjectionResidentBytes(layer.gate, layer.gatePacked)
	bytes += cpuSplitProjectionResidentBytes(layer.up, layer.upPacked)
	bytes += cpuSplitProjectionResidentBytes(layer.down, layer.downPacked)
	return bytes
}

func cpuSplitProjectionResidentBytes(dense []float32, packed *cpuSplitPackedMatrix) int64 {
	if packed != nil {
		return int64(len(packed.packed)) + int64(len(packed.scales)+len(packed.biases))*cpuSplitFloat32Bytes
	}
	return int64(len(dense)) * cpuSplitFloat32Bytes
}

func (executor *CPUSplitFFNExecutor) estimateLayerMemory(layer int) (CPUSplitFFNMemoryReport, error) {
	if layer < 0 || layer >= executor.cfg.NumHiddenLayers {
		return CPUSplitFFNMemoryReport{}, core.Errorf("mlx: CPU split FFN layer %d out of range", layer)
	}
	prefix := "model.layers." + strconv.Itoa(layer)
	var report CPUSplitFFNMemoryReport
	if err := executor.estimateVectorMemory(&report, cpuSplitWeightCandidates(prefix+".post_attention_layernorm.weight"), prefix+".post_attention_layernorm.weight", executor.cfg.HiddenSize, true); err != nil {
		return CPUSplitFFNMemoryReport{}, err
	}
	gateName := prefix + ".mlp.gate_proj.weight"
	if err := executor.estimateMatrixMemory(&report, gateName, executor.cfg.IntermediateSize, executor.cfg.HiddenSize); err != nil {
		return CPUSplitFFNMemoryReport{}, err
	}
	if err := executor.estimateVectorMemory(&report, cpuSplitProjectionBiasCandidates(gateName), gateName+".bias", executor.cfg.IntermediateSize, false); err != nil {
		return CPUSplitFFNMemoryReport{}, err
	}
	upName := prefix + ".mlp.up_proj.weight"
	if err := executor.estimateMatrixMemory(&report, upName, executor.cfg.IntermediateSize, executor.cfg.HiddenSize); err != nil {
		return CPUSplitFFNMemoryReport{}, err
	}
	if err := executor.estimateVectorMemory(&report, cpuSplitProjectionBiasCandidates(upName), upName+".bias", executor.cfg.IntermediateSize, false); err != nil {
		return CPUSplitFFNMemoryReport{}, err
	}
	downName := prefix + ".mlp.down_proj.weight"
	if err := executor.estimateMatrixMemory(&report, downName, executor.cfg.HiddenSize, executor.cfg.IntermediateSize); err != nil {
		return CPUSplitFFNMemoryReport{}, err
	}
	if err := executor.estimateVectorMemory(&report, cpuSplitProjectionBiasCandidates(downName), downName+".bias", executor.cfg.HiddenSize, false); err != nil {
		return CPUSplitFFNMemoryReport{}, err
	}
	report.finalise()
	return report, nil
}

func (executor *CPUSplitFFNExecutor) estimateVectorMemory(report *CPUSplitFFNMemoryReport, candidates []string, primary string, size int, required bool) error {
	ref, name, ok := executor.tensorRef(candidates)
	if !ok {
		if required {
			return core.NewError("mlx: CPU split FFN missing tensor " + primary)
		}
		return nil
	}
	if ref.Elements != size {
		return core.Errorf("mlx: CPU split FFN tensor %s has %d elements, want %d", name, ref.Elements, size)
	}
	bytes := int64(size) * cpuSplitFloat32Bytes
	if required {
		report.LayerNormBytes += bytes
	} else {
		report.ProjectionBiasBytes += bytes
	}
	report.ResidentBytes += bytes
	report.DenseEquivalentBytes += bytes
	return nil
}

func (executor *CPUSplitFFNExecutor) estimateMatrixMemory(report *CPUSplitFFNMemoryReport, name string, rows, cols int) error {
	ref, foundName, ok := executor.tensorRef(cpuSplitMatrixCandidates(name))
	if !ok {
		return core.NewError("mlx: CPU split FFN missing tensor " + name)
	}
	if cpuSplitPackedDType(ref.DType) {
		return executor.estimatePackedMatrixMemory(report, name, foundName, ref, rows, cols)
	}
	if ref.Elements != rows*cols {
		return core.Errorf("mlx: CPU split FFN tensor %s has %d elements, want %d", foundName, ref.Elements, rows*cols)
	}
	bytes := int64(rows*cols) * cpuSplitFloat32Bytes
	report.DenseProjections++
	report.DenseProjectionBytes += bytes
	report.ResidentBytes += bytes
	report.DenseEquivalentBytes += bytes
	return nil
}

func (executor *CPUSplitFFNExecutor) estimatePackedMatrixMemory(report *CPUSplitFFNMemoryReport, primaryName, foundName string, ref safetensors.TensorRef, rows, cols int) error {
	info := executor.packedInfo()
	if info == nil {
		return core.NewError("mlx: CPU split FFN packed tensor " + foundName + " requires JANG quantization metadata")
	}
	desc, err := infjang.NewPackedTensorDescriptor(primaryName, []uint64{uint64(rows), uint64(cols)}, info)
	if err != nil {
		return err
	}
	if ref.ByteLen != int64(desc.PackedBytes) {
		return core.Errorf("mlx: CPU split FFN packed tensor %s has %d bytes, want %d", foundName, ref.ByteLen, desc.PackedBytes)
	}
	scaleRef, _, ok := executor.tensorRef(cpuSplitSidecarCandidates(primaryName, foundName, "scales"))
	if !ok {
		return core.NewError("mlx: CPU split FFN packed tensor missing scales for " + primaryName)
	}
	if scaleRef.Elements != desc.ScaleCount {
		return core.Errorf("mlx: CPU split FFN packed tensor %s has %d scales, want %d", primaryName, scaleRef.Elements, desc.ScaleCount)
	}
	biasRef, _, ok := executor.tensorRef(cpuSplitSidecarCandidates(primaryName, foundName, "biases"))
	if !ok {
		return core.NewError("mlx: CPU split FFN packed tensor missing biases for " + primaryName)
	}
	if biasRef.Elements != desc.BiasCount {
		return core.Errorf("mlx: CPU split FFN packed tensor %s has %d biases, want %d", primaryName, biasRef.Elements, desc.BiasCount)
	}
	sidecarBytes := int64(scaleRef.Elements+biasRef.Elements) * cpuSplitFloat32Bytes
	equivalentBytes := int64(rows*cols) * cpuSplitFloat32Bytes
	report.PackedProjections++
	report.PackedProjectionBytes += ref.ByteLen
	report.PackedSidecarBytes += sidecarBytes
	report.ResidentBytes += ref.ByteLen + sidecarBytes
	report.DenseEquivalentBytes += equivalentBytes
	return nil
}

func (executor *CPUSplitFFNExecutor) loadLayer(ctx context.Context, layer int) (cpuSplitFFNLayer, error) {
	if err := ctx.Err(); err != nil {
		return cpuSplitFFNLayer{}, err
	}
	prefix := "model.layers." + strconv.Itoa(layer)
	norm, err := executor.loadVector(prefix+".post_attention_layernorm.weight", executor.cfg.HiddenSize)
	if err != nil {
		return cpuSplitFFNLayer{}, err
	}
	gateName := prefix + ".mlp.gate_proj.weight"
	gate, gatePacked, err := executor.loadMatrix(gateName, executor.cfg.IntermediateSize, executor.cfg.HiddenSize)
	if err != nil {
		return cpuSplitFFNLayer{}, err
	}
	gateBias, err := executor.loadOptionalVector(cpuSplitProjectionBiasCandidates(gateName), executor.cfg.IntermediateSize)
	if err != nil {
		return cpuSplitFFNLayer{}, err
	}
	upName := prefix + ".mlp.up_proj.weight"
	up, upPacked, err := executor.loadMatrix(upName, executor.cfg.IntermediateSize, executor.cfg.HiddenSize)
	if err != nil {
		return cpuSplitFFNLayer{}, err
	}
	upBias, err := executor.loadOptionalVector(cpuSplitProjectionBiasCandidates(upName), executor.cfg.IntermediateSize)
	if err != nil {
		return cpuSplitFFNLayer{}, err
	}
	downName := prefix + ".mlp.down_proj.weight"
	down, downPacked, err := executor.loadMatrix(downName, executor.cfg.HiddenSize, executor.cfg.IntermediateSize)
	if err != nil {
		return cpuSplitFFNLayer{}, err
	}
	downBias, err := executor.loadOptionalVector(cpuSplitProjectionBiasCandidates(downName), executor.cfg.HiddenSize)
	if err != nil {
		return cpuSplitFFNLayer{}, err
	}
	return cpuSplitFFNLayer{
		norm:         norm,
		gate:         gate,
		gatePacked:   gatePacked,
		gateBias:     gateBias,
		up:           up,
		upPacked:     upPacked,
		upBias:       upBias,
		down:         down,
		downPacked:   downPacked,
		downBias:     downBias,
		hidden:       executor.cfg.HiddenSize,
		intermediate: executor.cfg.IntermediateSize,
	}, nil
}

func (executor *CPUSplitFFNExecutor) loadVector(name string, size int) ([]float32, error) {
	return executor.loadVectorAny(cpuSplitWeightCandidates(name), name, size)
}

func (executor *CPUSplitFFNExecutor) loadOptionalVector(candidates []string, size int) ([]float32, error) {
	for _, name := range candidates {
		ref, ok := executor.index.Tensors[name]
		if !ok {
			continue
		}
		if ref.Elements != size {
			return nil, core.Errorf("mlx: CPU split FFN tensor %s has %d elements, want %d", name, ref.Elements, size)
		}
		return safetensors.ReadRefValues(ref)
	}
	return nil, nil
}

func (executor *CPUSplitFFNExecutor) loadVectorAny(candidates []string, primary string, size int) ([]float32, error) {
	ref, name, ok := executor.tensorRef(candidates)
	if !ok {
		return nil, core.NewError("mlx: CPU split FFN missing tensor " + primary)
	}
	if ref.Elements != size {
		return nil, core.Errorf("mlx: CPU split FFN tensor %s has %d elements, want %d", name, ref.Elements, size)
	}
	return safetensors.ReadRefValues(ref)
}

func (executor *CPUSplitFFNExecutor) loadMatrix(name string, rows, cols int) ([]float32, *cpuSplitPackedMatrix, error) {
	ref, foundName, ok := executor.tensorRef(cpuSplitMatrixCandidates(name))
	if !ok {
		return nil, nil, core.NewError("mlx: CPU split FFN missing tensor " + name)
	}
	if cpuSplitPackedDType(ref.DType) {
		return executor.loadPackedMatrix(name, foundName, ref, rows, cols)
	}
	if ref.Elements != rows*cols {
		return nil, nil, core.Errorf("mlx: CPU split FFN tensor %s has %d elements, want %d", foundName, ref.Elements, rows*cols)
	}
	values, err := safetensors.ReadRefValues(ref)
	return values, nil, err
}

func (executor *CPUSplitFFNExecutor) loadPackedMatrix(primaryName, foundName string, ref safetensors.TensorRef, rows, cols int) ([]float32, *cpuSplitPackedMatrix, error) {
	info := executor.packedInfo()
	if info == nil {
		return nil, nil, core.NewError("mlx: CPU split FFN packed tensor " + foundName + " requires JANG quantization metadata")
	}
	desc, err := infjang.NewPackedTensorDescriptor(primaryName, []uint64{uint64(rows), uint64(cols)}, info)
	if err != nil {
		return nil, nil, err
	}
	packed, err := safetensors.ReadRefRaw(ref)
	if err != nil {
		return nil, nil, err
	}
	scaleRef, _, ok := executor.tensorRef(cpuSplitSidecarCandidates(primaryName, foundName, "scales"))
	if !ok {
		return nil, nil, core.NewError("mlx: CPU split FFN packed tensor missing scales for " + primaryName)
	}
	scales, err := safetensors.ReadRefValues(scaleRef)
	if err != nil {
		return nil, nil, core.E("cpu_split_ffn.packed", "read scales", err)
	}
	biasRef, _, ok := executor.tensorRef(cpuSplitSidecarCandidates(primaryName, foundName, "biases"))
	if !ok {
		return nil, nil, core.NewError("mlx: CPU split FFN packed tensor missing biases for " + primaryName)
	}
	biases, err := safetensors.ReadRefValues(biasRef)
	if err != nil {
		return nil, nil, core.E("cpu_split_ffn.packed", "read biases", err)
	}
	if err := infjang.ValidatePackedTensor(desc, packed, scales, biases); err != nil {
		return nil, nil, err
	}
	return nil, &cpuSplitPackedMatrix{
		desc:      desc,
		packed:    packed,
		scales:    scales,
		biases:    biases,
		rows:      rows,
		cols:      cols,
		groupSize: desc.GroupSize,
		bits:      desc.Bits,
		elements:  desc.Elements,
	}, nil
}

func (executor *CPUSplitFFNExecutor) packedInfo() *infjang.Info {
	if executor.cfg.JANG != nil {
		return executor.cfg.JANG
	}
	if executor.cfg.PackedGroupSize <= 0 || executor.cfg.PackedBits <= 0 {
		return nil
	}
	return &infjang.Info{
		WeightFormat: "mxtq",
		Method:       "affine+mxtq",
		GroupSize:    executor.cfg.PackedGroupSize,
		BitsDefault:  executor.cfg.PackedBits,
	}
}

func (executor *CPUSplitFFNExecutor) tensorRef(candidates []string) (safetensors.TensorRef, string, bool) {
	for _, name := range candidates {
		if ref, ok := executor.index.Tensors[name]; ok {
			return ref, name, true
		}
	}
	return safetensors.TensorRef{}, "", false
}

func cpuSplitForwardDenseRow(hidden, out []float32, layer cpuSplitFFNLayer, eps float32, normed, activated []float32) {
	// Cache loop bounds + bias-presence checks before the inner loops. The
	// intermediate loop typically runs ~14336 iterations per token; re-doing
	// the len(layer.*Bias) > 0 check each pass shows up under perf.
	hiddenLen := layer.hidden
	intermediateLen := layer.intermediate
	hasGateBias := len(layer.gateBias) > 0
	hasUpBias := len(layer.upBias) > 0
	hasDownBias := len(layer.downBias) > 0

	var squares float64
	for _, value := range hidden {
		squares += float64(value * value)
	}
	scale := float32(1 / math.Sqrt(squares/float64(hiddenLen)+float64(eps)))
	// Re-slice all three views to hiddenLen up-front so the per-element
	// indexing has its bounds proved at the slice header — the compiler
	// can then drop the bounds checks on normed/hidden/layer.norm reads
	// in the inner loop.
	normedView := normed[:hiddenLen]
	hiddenView := hidden[:hiddenLen]
	normView := layer.norm[:hiddenLen]
	for i := 0; i < hiddenLen; i++ {
		normedView[i] = hiddenView[i] * scale * normView[i]
	}

	// Re-slice bias arrays + activated buffer to the loop bounds so the
	// per-row indexing in the projection-and-bias-fold loops compiles
	// without per-iter bounds checks. Loader keeps these matched to
	// intermediate/hidden sizes already, so the slice is exactly correct.
	activatedView := activated[:intermediateLen]
	var gateBiasView, upBiasView []float32
	if hasGateBias {
		gateBiasView = layer.gateBias[:intermediateLen]
	}
	if hasUpBias {
		upBiasView = layer.upBias[:intermediateLen]
	}
	for row := 0; row < intermediateLen; row++ {
		gate := cpuSplitProjectRow(normed, layer.gate, layer.gatePacked, row, hiddenLen)
		up := cpuSplitProjectRow(normed, layer.up, layer.upPacked, row, hiddenLen)
		if hasGateBias {
			gate += gateBiasView[row]
		}
		if hasUpBias {
			up += upBiasView[row]
		}
		activatedView[row] = cpuSplitSiLU(gate) * up
	}

	outView := out[:hiddenLen]
	hiddenViewRes := hidden[:hiddenLen]
	var downBiasView []float32
	if hasDownBias {
		downBiasView = layer.downBias[:hiddenLen]
	}
	for row := 0; row < hiddenLen; row++ {
		mlp := cpuSplitProjectRow(activated, layer.down, layer.downPacked, row, intermediateLen)
		if hasDownBias {
			mlp += downBiasView[row]
		}
		outView[row] = hiddenViewRes[row] + mlp
	}
}

func cpuSplitDot(a, b []float32) float32 {
	// Re-slice b to len(a) so the compiler can prove every b[i] is in
	// bounds when walking the indexed loop. Without the hint, each b[i]
	// triggers a per-iteration bounds check that dominates the inner dot
	// when len(a) is in the thousands (the projection row size).
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	a = a[:n]
	b = b[:n]
	var sum float32
	for i := 0; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

func cpuSplitProjectRow(input, dense []float32, packed *cpuSplitPackedMatrix, row, cols int) float32 {
	if packed != nil {
		return cpuSplitPackedDot(input, packed, row)
	}
	offset := row * cols
	return cpuSplitDot(input, dense[offset:offset+cols])
}

func cpuSplitPackedDot(input []float32, matrix *cpuSplitPackedMatrix, row int) float32 {
	if matrix == nil || row < 0 || row >= matrix.rows {
		return 0
	}
	// Hoist the loop bound: the original double-condition (col < matrix.cols
	// && col < len(input)) re-read both sources every iteration. min() once,
	// then a single-bound loop lets the compiler elide bounds checks on the
	// input slice when col stays under len(input).
	cols := matrix.cols
	if n := len(input); n < cols {
		cols = n
	}
	offset := row * matrix.cols
	in := input[:cols]
	// Hoist hot fields from matrix once — the per-element value() call
	// would chase each of these through the struct (and through the desc
	// for groupSize/bits/elements) on every element of every projection
	// row. With ~hidden_size elements per row and ~intermediate rows per
	// token, that ran into the billions per layer.
	//
	// matrix.elements equals matrix.rows * matrix.cols by construction
	// (PackedTensorDescriptor.Elements is the product of shape dims set in
	// NewPackedTensorDescriptor from []uint64{rows, cols}). With the row
	// bound check at the top of the function and col < cols <= matrix.cols
	// inside the loop, every idx is provably under elements, so the per-
	// element guard from the original (*cpuSplitPackedMatrix).value path
	// drops out entirely.
	packed := matrix.packed
	scales := matrix.scales
	biases := matrix.biases
	groupSize := matrix.groupSize
	bits := matrix.bits
	var sum float32
	for col := 0; col < cols; col++ {
		idx := offset + col
		group := idx / groupSize
		q := cpuSplitUnpackPackedValue(packed, idx, bits)
		sum += in[col] * (float32(q)*scales[group] + biases[group])
	}
	return sum
}

func (matrix *cpuSplitPackedMatrix) value(index int) float32 {
	if matrix == nil || index < 0 || uint64(index) >= matrix.elements {
		return 0
	}
	group := index / matrix.groupSize
	q := cpuSplitUnpackPackedValue(matrix.packed, index, matrix.bits)
	return float32(q)*matrix.scales[group] + matrix.biases[group]
}

func cpuSplitUnpackPackedValue(packed []byte, index, bits int) uint8 {
	// Fast paths for the byte-aligned bit widths actually emitted by the
	// JANG packers (8-bit dense, 4-bit nibble-packed). These cover the
	// overwhelmingly common cases and skip the per-bit walk loop, which is
	// hit hundreds of millions of times per layer otherwise.
	switch bits {
	case 8:
		return packed[index]
	case 4:
		b := packed[index>>1]
		if index&1 == 0 {
			return b & 0x0F
		}
		return b >> 4
	}
	bitOffset := index * bits
	remaining := bits
	shiftOut := 0
	value := uint16(0)
	for remaining > 0 {
		byteIndex := bitOffset / 8
		shiftIn := bitOffset % 8
		take := cpuSplitMinInt(remaining, 8-shiftIn)
		mask := uint16((1 << take) - 1)
		chunk := (uint16(packed[byteIndex]) >> shiftIn) & mask
		value |= chunk << shiftOut
		remaining -= take
		bitOffset += take
		shiftOut += take
	}
	return uint8(value)
}

func cpuSplitMinInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func cpuSplitSiLU(value float32) float32 {
	return value / (1 + float32(math.Exp(float64(-value))))
}

func cpuSplitFirstPositive(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func cpuSplitPackedDType(dtype string) bool {
	switch core.Upper(dtype) {
	case "U8", "UINT8":
		return true
	default:
		return false
	}
}

func cpuSplitWeightCandidates(name string) []string {
	if core.HasPrefix(name, "model.") {
		suffix := core.TrimPrefix(name, "model.")
		candidates := make([]string, 0, 5)
		return append(candidates,
			name,
			"language_model."+name,
			"language_model.model."+suffix,
			"model.language_model."+suffix,
			"model.language_model.model."+suffix,
		)
	}
	candidates := make([]string, 0, 6)
	return append(candidates,
		name,
		"model."+name,
		"language_model."+name,
		"language_model.model."+name,
		"model.language_model."+name,
		"model.language_model.model."+name,
	)
}

func cpuSplitMatrixCandidates(name string) []string {
	bases := cpuSplitWeightCandidates(name)
	candidates := make([]string, 0, len(bases)*4)
	for _, base := range bases {
		trimmed := cpuSplitTrimWeightSuffix(base)
		candidates = append(candidates, base, base+".packed", base+".qweight", trimmed+".qweight")
	}
	return cpuSplitUniqueStrings(candidates)
}

func cpuSplitProjectionBiasCandidates(weightName string) []string {
	weightCandidates := cpuSplitWeightCandidates(weightName)
	candidates := make([]string, 0, len(weightCandidates)*3)
	for _, name := range weightCandidates {
		trimmed := cpuSplitTrimWeightSuffix(name)
		candidates = append(candidates, trimmed+".bias", name+".proj_bias", trimmed+".proj_bias")
	}
	return candidates
}

func cpuSplitSidecarCandidates(primaryName, foundName, sidecar string) []string {
	// Pre-size names — foundName + optional trimmed-packed-suffix + primaryName
	// + the weight-candidate fan-out (up to 6 entries). Saves a couple of
	// underlying-array reallocs per packed-tensor load.
	base := cpuSplitWeightCandidates(primaryName)
	names := make([]string, 0, 2+1+len(base))
	names = append(names, foundName)
	if trimmed := cpuSplitTrimPackedSuffix(foundName); trimmed != foundName {
		names = append(names, trimmed)
	}
	names = append(names, primaryName)
	names = append(names, base...)
	candidates := make([]string, 0, len(names)*3)
	for _, name := range names {
		trimmed := cpuSplitTrimWeightSuffix(name)
		candidates = append(candidates, name+"."+sidecar, trimmed+"."+sidecar, name+"_"+sidecar)
	}
	return cpuSplitUniqueStrings(candidates)
}

func cpuSplitTrimWeightSuffix(name string) string {
	if core.HasSuffix(name, ".weight") {
		return core.TrimSuffix(name, ".weight")
	}
	return name
}

func cpuSplitTrimPackedSuffix(name string) string {
	for _, suffix := range []string{".packed", ".qweight"} {
		if core.HasSuffix(name, suffix) {
			return core.TrimSuffix(name, suffix)
		}
	}
	return name
}

func cpuSplitUniqueStrings(values []string) []string {
	seen := make(map[string]struct{}, len(values))
	out := make([]string, 0, len(values))
	for _, value := range values {
		if value == "" {
			continue
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		out = append(out, value)
	}
	return out
}
