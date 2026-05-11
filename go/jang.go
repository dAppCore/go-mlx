// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import core "dappco.re/go"

// JANGQuantizationInfo captures JANG/JANGTQ sidecar metadata for MLX safetensor packs.
type JANGQuantizationInfo struct {
	Version            int                            `json:"version,omitempty"`
	WeightFormat       string                         `json:"weight_format,omitempty"`
	Profile            string                         `json:"profile,omitempty"`
	Method             string                         `json:"method,omitempty"`
	GroupSize          int                            `json:"group_size,omitempty"`
	BitsDefault        int                            `json:"bits_default,omitempty"`
	AttentionBits      int                            `json:"attention_bits,omitempty"`
	SharedExpertBits   int                            `json:"shared_expert_bits,omitempty"`
	RoutedExpertBits   int                            `json:"routed_expert_bits,omitempty"`
	EmbedTokensBits    int                            `json:"embed_tokens_bits,omitempty"`
	LMHeadBits         int                            `json:"lm_head_bits,omitempty"`
	SourceName         string                         `json:"source_name,omitempty"`
	SourceOrg          string                         `json:"source_org,omitempty"`
	SourceArchitecture string                         `json:"source_architecture,omitempty"`
	Capabilities       JANGCapabilities               `json:"capabilities,omitempty"`
	Packed             *JANGPackedQuantizationProfile `json:"packed,omitempty"`
}

// JANGCapabilities records runtime-facing affordances declared by jang_config.json.
type JANGCapabilities struct {
	ReasoningParser  string `json:"reasoning_parser,omitempty"`
	ToolParser       string `json:"tool_parser,omitempty"`
	ThinkInTemplate  bool   `json:"think_in_template,omitempty"`
	SupportsTools    bool   `json:"supports_tools,omitempty"`
	SupportsThinking bool   `json:"supports_thinking,omitempty"`
	Family           string `json:"family,omitempty"`
	Modality         string `json:"modality,omitempty"`
	CacheType        string `json:"cache_type,omitempty"`
}

// JANGTensorRole classifies a packed tensor so mixed-precision JANGTQ profiles
// can choose the right bit width without hard-coding one global quant size.
type JANGTensorRole string

const (
	JANGTensorRoleDefault      JANGTensorRole = "default"
	JANGTensorRoleAttention    JANGTensorRole = "attention"
	JANGTensorRoleSharedExpert JANGTensorRole = "shared_expert"
	JANGTensorRoleRoutedExpert JANGTensorRole = "routed_expert"
	JANGTensorRoleEmbedTokens  JANGTensorRole = "embed_tokens"
	JANGTensorRoleLMHead       JANGTensorRole = "lm_head"
)

const (
	JANGBitOrderLSB0   = "lsb0"
	JANGEncodingAffine = "affine"
)

// JANGPackedQuantizationProfile describes the mixed-precision packed layout
// declared by jang_config.json. It is intentionally backend-neutral so future
// ROCm/CUDA/TPU implementations can reuse the same model-pack contract.
type JANGPackedQuantizationProfile struct {
	Type          string         `json:"type,omitempty"`
	Format        string         `json:"format,omitempty"`
	Profile       string         `json:"profile,omitempty"`
	Method        string         `json:"method,omitempty"`
	GroupSize     int            `json:"group_size,omitempty"`
	BitsDefault   int            `json:"bits_default,omitempty"`
	RoleBits      map[string]int `json:"role_bits,omitempty"`
	MinBits       int            `json:"min_bits,omitempty"`
	MaxBits       int            `json:"max_bits,omitempty"`
	Mixed         bool           `json:"mixed,omitempty"`
	BitOrder      string         `json:"bit_order,omitempty"`
	Encoding      string         `json:"encoding,omitempty"`
	ValuesPerByte int            `json:"values_per_byte,omitempty"`
}

// JANGPackedTensorDescriptor describes one packed tensor's logical and physical
// layout before backend-specific dequant kernels are selected.
type JANGPackedTensorDescriptor struct {
	Name          string         `json:"name,omitempty"`
	Type          string         `json:"type,omitempty"`
	Format        string         `json:"format,omitempty"`
	Profile       string         `json:"profile,omitempty"`
	Role          JANGTensorRole `json:"role,omitempty"`
	Shape         []uint64       `json:"shape,omitempty"`
	Elements      uint64         `json:"elements,omitempty"`
	Bits          int            `json:"bits,omitempty"`
	GroupSize     int            `json:"group_size,omitempty"`
	Groups        int            `json:"groups,omitempty"`
	PackedBytes   int            `json:"packed_bytes,omitempty"`
	ValuesPerByte int            `json:"values_per_byte,omitempty"`
	ScaleCount    int            `json:"scale_count,omitempty"`
	BiasCount     int            `json:"bias_count,omitempty"`
	BitOrder      string         `json:"bit_order,omitempty"`
	Encoding      string         `json:"encoding,omitempty"`
}

type jangConfigProbe struct {
	Version      int    `json:"version"`
	WeightFormat string `json:"weight_format"`
	Profile      string `json:"profile"`
	SourceModel  struct {
		Name         string `json:"name"`
		Org          string `json:"org"`
		Architecture string `json:"architecture"`
	} `json:"source_model"`
	MXTQBits struct {
		Attention    int `json:"attention"`
		SharedExpert int `json:"shared_expert"`
		RoutedExpert int `json:"routed_expert"`
		EmbedTokens  int `json:"embed_tokens"`
		LMHead       int `json:"lm_head"`
	} `json:"mxtq_bits"`
	Quantization struct {
		Method      string `json:"method"`
		GroupSize   int    `json:"group_size"`
		BitsDefault int    `json:"bits_default"`
	} `json:"quantization"`
	Capabilities JANGCapabilities `json:"capabilities"`
}

func readJANGQuantizationInfo(root string) (*JANGQuantizationInfo, error) {
	read := core.ReadFile(core.PathJoin(root, "jang_config.json"))
	if !read.OK {
		if core.IsNotExist(read.Value.(error)) {
			return nil, nil
		}
		return nil, read.Value.(error)
	}
	return parseJANGQuantizationInfo(read.Value.([]byte))
}

func parseJANGQuantizationInfo(data []byte) (*JANGQuantizationInfo, error) {
	var probe jangConfigProbe
	if result := core.JSONUnmarshal(data, &probe); !result.OK {
		return nil, result.Value.(error)
	}
	return finalizeJANGQuantizationInfo(&JANGQuantizationInfo{
		Version:            probe.Version,
		WeightFormat:       probe.WeightFormat,
		Profile:            probe.Profile,
		Method:             probe.Quantization.Method,
		GroupSize:          probe.Quantization.GroupSize,
		BitsDefault:        firstPositive(probe.Quantization.BitsDefault, probe.MXTQBits.RoutedExpert, jangProfileBits(probe.Profile)),
		AttentionBits:      probe.MXTQBits.Attention,
		SharedExpertBits:   probe.MXTQBits.SharedExpert,
		RoutedExpertBits:   probe.MXTQBits.RoutedExpert,
		EmbedTokensBits:    probe.MXTQBits.EmbedTokens,
		LMHeadBits:         probe.MXTQBits.LMHead,
		SourceName:         probe.SourceModel.Name,
		SourceOrg:          probe.SourceModel.Org,
		SourceArchitecture: normalizeKnownArchitecture(probe.SourceModel.Architecture),
		Capabilities:       probe.Capabilities,
	}), nil
}

func inferJANGQuantizationFromHF(meta HFModelMetadata) *JANGQuantizationInfo {
	needle := core.Lower(firstNonEmpty(meta.ID, meta.ModelID))
	for _, tag := range meta.Tags {
		needle = core.Concat(needle, " ", core.Lower(tag))
	}
	for _, file := range meta.Files {
		needle = core.Concat(needle, " ", core.Lower(file.filename()))
	}

	switch {
	case core.Contains(needle, "jangtq"):
		return finalizeJANGQuantizationInfo(&JANGQuantizationInfo{
			Profile:          "JANGTQ",
			WeightFormat:     "mxtq",
			Method:           "affine+mxtq",
			GroupSize:        hfJANGGroupSize(meta),
			BitsDefault:      2,
			RoutedExpertBits: 2,
		})
	case core.Contains(needle, "jang"):
		profile := inferJANGProfileName(needle)
		return finalizeJANGQuantizationInfo(&JANGQuantizationInfo{
			Profile:     profile,
			GroupSize:   hfJANGGroupSize(meta),
			BitsDefault: firstPositive(jangProfileBits(profile), 0),
		})
	default:
		return nil
	}
}

func hfJANGGroupSize(meta HFModelMetadata) int {
	if quant := meta.Config.QuantizationConfig; quant != nil && quant.GroupSize > 0 {
		return quant.GroupSize
	}
	if quant := meta.Config.Quantization; quant != nil && quant.GroupSize > 0 {
		return quant.GroupSize
	}
	return 64
}

func inferJANGProfileName(value string) string {
	for _, profile := range []string{"jang_1l", "jang_2s", "jang_2l", "jang_3l", "jang_4k", "jang_4m"} {
		if core.Contains(value, profile) {
			return core.Upper(profile)
		}
	}
	return "JANG"
}

func jangProfileBits(profile string) int {
	profile = core.Lower(profile)
	switch {
	case core.Contains(profile, "jangtq"):
		return 2
	case core.Contains(profile, "jang_1"):
		return 1
	case core.Contains(profile, "jang_2"):
		return 2
	case core.Contains(profile, "jang_3"):
		return 3
	case core.Contains(profile, "jang_4"):
		return 4
	default:
		return 0
	}
}

func jangQuantizationType(info *JANGQuantizationInfo) string {
	if info == nil {
		return ""
	}
	lower := core.Lower(core.Concat(info.Profile, " ", info.WeightFormat, " ", info.Method))
	if core.Contains(lower, "jangtq") || core.Contains(lower, "mxtq") {
		return "jangtq"
	}
	return "jang"
}

func finalizeJANGQuantizationInfo(info *JANGQuantizationInfo) *JANGQuantizationInfo {
	if info == nil {
		return nil
	}
	info.Packed = BuildJANGPackedQuantizationProfile(info)
	return info
}

// BuildJANGPackedQuantizationProfile returns the backend-neutral packed layout
// profile for JANG/JANGTQ metadata.
func BuildJANGPackedQuantizationProfile(info *JANGQuantizationInfo) *JANGPackedQuantizationProfile {
	if info == nil {
		return nil
	}
	roleBits := jangRoleBits(info)
	minBits, maxBits := jangMinMaxBits(roleBits)
	profile := &JANGPackedQuantizationProfile{
		Type:          jangQuantizationType(info),
		Format:        jangPackedFormat(info),
		Profile:       info.Profile,
		Method:        info.Method,
		GroupSize:     info.GroupSize,
		BitsDefault:   info.BitsDefault,
		RoleBits:      roleBits,
		MinBits:       minBits,
		MaxBits:       maxBits,
		Mixed:         minBits > 0 && maxBits > minBits,
		BitOrder:      JANGBitOrderLSB0,
		Encoding:      JANGEncodingAffine,
		ValuesPerByte: jangValuesPerByte(info.BitsDefault),
	}
	if profile.Format == "" {
		profile.Format = profile.Type
	}
	return profile
}

// CloneJANGPackedQuantizationProfile returns an independent copy of profile.
func CloneJANGPackedQuantizationProfile(profile *JANGPackedQuantizationProfile) *JANGPackedQuantizationProfile {
	if profile == nil {
		return nil
	}
	cloned := *profile
	cloned.RoleBits = cloneJANGRoleBits(profile.RoleBits)
	return &cloned
}

// NewJANGPackedTensorDescriptor builds and validates a packed tensor layout for
// the supplied logical tensor shape.
func NewJANGPackedTensorDescriptor(name string, shape []uint64, info *JANGQuantizationInfo) (JANGPackedTensorDescriptor, error) {
	if info == nil {
		return JANGPackedTensorDescriptor{}, core.NewError("mlx: JANG packed tensor descriptor requires quantization info")
	}
	role := inferJANGTensorRole(name)
	bits := jangBitsForRole(info, role)
	elements, err := jangShapeElements(shape)
	if err != nil {
		return JANGPackedTensorDescriptor{}, err
	}
	if err := validateJANGBits(bits, name); err != nil {
		return JANGPackedTensorDescriptor{}, err
	}
	if info.GroupSize <= 0 {
		return JANGPackedTensorDescriptor{}, core.NewError(core.Sprintf("mlx: JANG packed tensor %q has invalid group size %d", name, info.GroupSize))
	}
	if elements > ^uint64(0)/uint64(bits) {
		return JANGPackedTensorDescriptor{}, core.NewError(core.Sprintf("mlx: JANG packed tensor %q packed bit count overflows", name))
	}
	packedBits := elements * uint64(bits)
	packedBytes := ceilDivUint64(packedBits, 8)
	if packedBytes > uint64(maxIntValue()) {
		return JANGPackedTensorDescriptor{}, core.NewError(core.Sprintf("mlx: JANG packed tensor %q is too large", name))
	}
	groups := ceilDivUint64(elements, uint64(info.GroupSize))
	if groups > uint64(maxIntValue()) {
		return JANGPackedTensorDescriptor{}, core.NewError(core.Sprintf("mlx: JANG packed tensor %q has too many groups", name))
	}
	return JANGPackedTensorDescriptor{
		Name:          name,
		Type:          jangQuantizationType(info),
		Format:        jangPackedFormat(info),
		Profile:       info.Profile,
		Role:          role,
		Shape:         append([]uint64(nil), shape...),
		Elements:      elements,
		Bits:          bits,
		GroupSize:     info.GroupSize,
		Groups:        int(groups),
		PackedBytes:   int(packedBytes),
		ValuesPerByte: jangValuesPerByte(bits),
		ScaleCount:    int(groups),
		BiasCount:     int(groups),
		BitOrder:      JANGBitOrderLSB0,
		Encoding:      JANGEncodingAffine,
	}, nil
}

// ValidateJANGPackedTensor checks physical storage lengths against the descriptor.
func ValidateJANGPackedTensor(desc JANGPackedTensorDescriptor, packed []byte, scales, biases []float32) error {
	if err := validateJANGDescriptor(desc); err != nil {
		return err
	}
	if len(packed) != desc.PackedBytes {
		return core.NewError(core.Sprintf("mlx: JANG packed tensor %q packed length %d, expected %d", desc.Name, len(packed), desc.PackedBytes))
	}
	if len(scales) != desc.ScaleCount {
		return core.NewError(core.Sprintf("mlx: JANG packed tensor %q scale count %d, expected %d", desc.Name, len(scales), desc.ScaleCount))
	}
	if len(biases) != desc.BiasCount {
		return core.NewError(core.Sprintf("mlx: JANG packed tensor %q bias count %d, expected %d", desc.Name, len(biases), desc.BiasCount))
	}
	return nil
}

// DequantizeJANGPackedTensor is a small reference implementation used by tests
// and future backend parity checks. Native kernels should match this layout.
func DequantizeJANGPackedTensor(desc JANGPackedTensorDescriptor, packed []byte, scales, biases []float32) ([]float32, error) {
	if err := ValidateJANGPackedTensor(desc, packed, scales, biases); err != nil {
		return nil, err
	}
	if desc.Elements > uint64(maxIntValue()) {
		return nil, core.NewError(core.Sprintf("mlx: JANG packed tensor %q is too large to dequantize on CPU", desc.Name))
	}
	out := make([]float32, int(desc.Elements))
	for i := range out {
		group := i / desc.GroupSize
		q := unpackJANGQuantizedValue(packed, i, desc.Bits)
		out[i] = float32(q)*scales[group] + biases[group]
	}
	return out, nil
}

// PackJANGQuantizedValues packs logical quantized values using the descriptor's
// LSB-first bit layout. It is intended for fixtures and round-trip tests.
func PackJANGQuantizedValues(desc JANGPackedTensorDescriptor, values []uint8) ([]byte, error) {
	if err := validateJANGDescriptor(desc); err != nil {
		return nil, err
	}
	if uint64(len(values)) != desc.Elements {
		return nil, core.NewError(core.Sprintf("mlx: JANG packed tensor %q value count %d, expected %d", desc.Name, len(values), desc.Elements))
	}
	out := make([]byte, desc.PackedBytes)
	maxValue := uint8((1 << desc.Bits) - 1)
	for i, value := range values {
		if value > maxValue {
			return nil, core.NewError(core.Sprintf("mlx: JANG packed tensor %q value %d exceeds %d-bit max %d", desc.Name, value, desc.Bits, maxValue))
		}
		writeJANGQuantizedValue(out, i, desc.Bits, value)
	}
	return out, nil
}

func inferJANGTensorRole(name string) JANGTensorRole {
	lower := core.Lower(name)
	switch {
	case core.Contains(lower, "embed_tokens"):
		return JANGTensorRoleEmbedTokens
	case core.Contains(lower, "lm_head"):
		return JANGTensorRoleLMHead
	case core.Contains(lower, "shared_expert"):
		return JANGTensorRoleSharedExpert
	case core.Contains(lower, "experts.") || core.Contains(lower, "block_sparse_moe"):
		return JANGTensorRoleRoutedExpert
	case core.Contains(lower, "self_attn") || core.Contains(lower, ".attention.") || core.Contains(lower, ".q_proj") || core.Contains(lower, ".k_proj") || core.Contains(lower, ".v_proj") || core.Contains(lower, ".o_proj"):
		return JANGTensorRoleAttention
	default:
		return JANGTensorRoleDefault
	}
}

func jangBitsForRole(info *JANGQuantizationInfo, role JANGTensorRole) int {
	switch role {
	case JANGTensorRoleAttention:
		return firstPositive(info.AttentionBits, info.BitsDefault, jangProfileBits(info.Profile))
	case JANGTensorRoleSharedExpert:
		return firstPositive(info.SharedExpertBits, info.BitsDefault, jangProfileBits(info.Profile))
	case JANGTensorRoleRoutedExpert:
		return firstPositive(info.RoutedExpertBits, info.BitsDefault, jangProfileBits(info.Profile))
	case JANGTensorRoleEmbedTokens:
		return firstPositive(info.EmbedTokensBits, info.BitsDefault, jangProfileBits(info.Profile))
	case JANGTensorRoleLMHead:
		return firstPositive(info.LMHeadBits, info.BitsDefault, jangProfileBits(info.Profile))
	default:
		return firstPositive(info.BitsDefault, jangProfileBits(info.Profile))
	}
}

func jangRoleBits(info *JANGQuantizationInfo) map[string]int {
	if info == nil {
		return nil
	}
	roles := []JANGTensorRole{
		JANGTensorRoleDefault,
		JANGTensorRoleAttention,
		JANGTensorRoleSharedExpert,
		JANGTensorRoleRoutedExpert,
		JANGTensorRoleEmbedTokens,
		JANGTensorRoleLMHead,
	}
	out := map[string]int{}
	for _, role := range roles {
		if bits := jangBitsForRole(info, role); bits > 0 {
			out[string(role)] = bits
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func jangMinMaxBits(roleBits map[string]int) (int, int) {
	minBits, maxBits := 0, 0
	for _, bits := range roleBits {
		if bits <= 0 {
			continue
		}
		if minBits == 0 || bits < minBits {
			minBits = bits
		}
		if bits > maxBits {
			maxBits = bits
		}
	}
	return minBits, maxBits
}

func jangPackedFormat(info *JANGQuantizationInfo) string {
	if info == nil {
		return ""
	}
	lower := core.Lower(core.Concat(info.WeightFormat, " ", info.Profile, " ", info.Method))
	switch {
	case core.Contains(lower, "mxtq"):
		return "mxtq"
	case core.Contains(lower, "jangtq"):
		return "jangtq"
	case core.Contains(lower, "jang"):
		return "jang"
	default:
		return core.Lower(info.WeightFormat)
	}
}

func jangValuesPerByte(bits int) int {
	if bits <= 0 {
		return 0
	}
	return 8 / bits
}

func jangShapeElements(shape []uint64) (uint64, error) {
	if len(shape) == 0 {
		return 0, core.NewError("mlx: JANG packed tensor shape is required")
	}
	elements := uint64(1)
	for _, dim := range shape {
		if dim == 0 {
			return 0, core.NewError("mlx: JANG packed tensor shape contains zero dimension")
		}
		if elements > ^uint64(0)/dim {
			return 0, core.NewError("mlx: JANG packed tensor shape overflows element count")
		}
		elements *= dim
	}
	return elements, nil
}

func validateJANGDescriptor(desc JANGPackedTensorDescriptor) error {
	if desc.Elements == 0 {
		return core.NewError(core.Sprintf("mlx: JANG packed tensor %q has no elements", desc.Name))
	}
	if err := validateJANGBits(desc.Bits, desc.Name); err != nil {
		return err
	}
	if desc.GroupSize <= 0 {
		return core.NewError(core.Sprintf("mlx: JANG packed tensor %q has invalid group size %d", desc.Name, desc.GroupSize))
	}
	if desc.PackedBytes <= 0 {
		return core.NewError(core.Sprintf("mlx: JANG packed tensor %q has invalid packed byte count %d", desc.Name, desc.PackedBytes))
	}
	if desc.ScaleCount <= 0 || desc.BiasCount <= 0 {
		return core.NewError(core.Sprintf("mlx: JANG packed tensor %q has invalid scale/bias counts", desc.Name))
	}
	return nil
}

func validateJANGBits(bits int, name string) error {
	switch bits {
	case 1, 2, 3, 4, 8:
		return nil
	default:
		return core.NewError(core.Sprintf("mlx: JANG packed tensor %q has unsupported %d-bit width", name, bits))
	}
}

func unpackJANGQuantizedValue(packed []byte, index, bits int) uint8 {
	bitOffset := index * bits
	remaining := bits
	shiftOut := 0
	value := uint16(0)
	for remaining > 0 {
		byteIndex := bitOffset / 8
		shiftIn := bitOffset % 8
		take := minJANGInt(remaining, 8-shiftIn)
		mask := uint16((1 << take) - 1)
		chunk := (uint16(packed[byteIndex]) >> shiftIn) & mask
		value |= chunk << shiftOut
		remaining -= take
		bitOffset += take
		shiftOut += take
	}
	return uint8(value)
}

func writeJANGQuantizedValue(out []byte, index, bits int, value uint8) {
	bitOffset := index * bits
	remaining := bits
	raw := uint16(value)
	for remaining > 0 {
		byteIndex := bitOffset / 8
		shift := bitOffset % 8
		take := minJANGInt(remaining, 8-shift)
		mask := uint16((1 << take) - 1)
		out[byteIndex] |= byte((raw & mask) << shift)
		raw >>= take
		remaining -= take
		bitOffset += take
	}
}

func ceilDivUint64(value, divisor uint64) uint64 {
	if divisor == 0 || value == 0 {
		return 0
	}
	quotient := value / divisor
	if value%divisor != 0 {
		quotient++
	}
	return quotient
}

func maxIntValue() int {
	return int(^uint(0) >> 1)
}

func minJANGInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func cloneJANGRoleBits(roleBits map[string]int) map[string]int {
	if len(roleBits) == 0 {
		return nil
	}
	cloned := make(map[string]int, len(roleBits))
	for key, value := range roleBits {
		cloned[key] = value
	}
	return cloned
}
