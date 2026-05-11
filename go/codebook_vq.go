// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import core "dappco.re/go"

const (
	CodebookQuantizationType = "codebook"
	CodebookFormatVQ         = "vq"
)

// CodebookQuantizationProfile describes vector-quantized tensor sidecars in a
// model pack. The runtime lane starts with unpacked integer codes and f32
// codebooks; packed code streams can layer on this metadata later.
type CodebookQuantizationProfile struct {
	Type         string                     `json:"type,omitempty"`
	Format       string                     `json:"format,omitempty"`
	CodebookSize int                        `json:"codebook_size,omitempty"`
	CodeDim      int                        `json:"code_dim,omitempty"`
	IndexBits    int                        `json:"index_bits,omitempty"`
	Source       string                     `json:"source,omitempty"`
	Tensors      []CodebookTensorDescriptor `json:"tensors,omitempty"`
}

// CodebookTensorDescriptor is the validated tensor-local shape contract for one
// VQ-compressed weight matrix.
type CodebookTensorDescriptor struct {
	Name          string   `json:"name,omitempty"`
	Format        string   `json:"format,omitempty"`
	Shape         []uint64 `json:"shape,omitempty"`
	Elements      uint64   `json:"elements,omitempty"`
	CodebookSize  int      `json:"codebook_size,omitempty"`
	CodeDim       int      `json:"code_dim,omitempty"`
	CodeCount     int      `json:"code_count,omitempty"`
	IndexBits     int      `json:"index_bits,omitempty"`
	IndexBytes    int      `json:"index_bytes,omitempty"`
	CodesName     string   `json:"codes_name,omitempty"`
	CodebookName  string   `json:"codebook_name,omitempty"`
	CodesShape    []uint64 `json:"codes_shape,omitempty"`
	CodebookShape []uint64 `json:"codebook_shape,omitempty"`
}

type codebookConfigProbe struct {
	Type         string `json:"type"`
	Format       string `json:"format"`
	CodebookSize int    `json:"codebook_size"`
	CodeDim      int    `json:"code_dim"`
	IndexBits    int    `json:"index_bits"`
	Source       string `json:"source"`
	Tensors      []struct {
		Name          string   `json:"name"`
		Shape         []uint64 `json:"shape"`
		CodesName     string   `json:"codes"`
		CodebookName  string   `json:"codebook"`
		CodesShape    []uint64 `json:"codes_shape"`
		CodebookShape []uint64 `json:"codebook_shape"`
		CodebookSize  int      `json:"codebook_size"`
		CodeDim       int      `json:"code_dim"`
		IndexBits     int      `json:"index_bits"`
	} `json:"tensors"`
}

// ParseCodebookQuantizationProfile parses codebook_config.json.
func ParseCodebookQuantizationProfile(data []byte) (*CodebookQuantizationProfile, error) {
	var probe codebookConfigProbe
	if result := core.JSONUnmarshal(data, &probe); !result.OK {
		return nil, result.Value.(error)
	}
	profile := CodebookQuantizationProfile{
		Type:         firstNonEmpty(probe.Type, CodebookQuantizationType),
		Format:       firstNonEmpty(probe.Format, CodebookFormatVQ),
		CodebookSize: probe.CodebookSize,
		CodeDim:      probe.CodeDim,
		IndexBits:    firstPositive(probe.IndexBits, 8),
		Source:       firstNonEmpty(probe.Source, "codebook_config.json"),
	}
	for _, tensor := range probe.Tensors {
		local := profile
		local.CodebookSize = firstPositive(tensor.CodebookSize, profile.CodebookSize)
		local.CodeDim = firstPositive(tensor.CodeDim, profile.CodeDim)
		local.IndexBits = firstPositive(tensor.IndexBits, profile.IndexBits)
		desc, err := NewCodebookTensorDescriptor(tensor.Name, tensor.Shape, local)
		if err != nil {
			return nil, err
		}
		desc.CodesName = firstNonEmpty(tensor.CodesName, defaultCodebookCodesName(desc.Name))
		desc.CodebookName = firstNonEmpty(tensor.CodebookName, defaultCodebookTableName(desc.Name))
		if len(tensor.CodesShape) > 0 {
			desc.CodesShape = append([]uint64(nil), tensor.CodesShape...)
		}
		if len(tensor.CodebookShape) > 0 {
			desc.CodebookShape = append([]uint64(nil), tensor.CodebookShape...)
		}
		profile.Tensors = append(profile.Tensors, desc)
	}
	if err := ValidateCodebookQuantizationProfile(profile); err != nil {
		return nil, err
	}
	return &profile, nil
}

// NewCodebookTensorDescriptor creates a validated descriptor for one VQ tensor.
func NewCodebookTensorDescriptor(name string, shape []uint64, profile CodebookQuantizationProfile) (CodebookTensorDescriptor, error) {
	if name == "" {
		return CodebookTensorDescriptor{}, core.NewError("mlx: codebook tensor name is required")
	}
	if profile.Format == "" {
		profile.Format = CodebookFormatVQ
	}
	if profile.Format != CodebookFormatVQ {
		return CodebookTensorDescriptor{}, core.NewError("mlx: unsupported codebook format: " + profile.Format)
	}
	if len(shape) != 2 || shape[0] == 0 || shape[1] == 0 {
		return CodebookTensorDescriptor{}, core.NewError("mlx: codebook tensor shape must be [out, in]")
	}
	if profile.CodebookSize <= 0 {
		return CodebookTensorDescriptor{}, core.NewError("mlx: codebook size must be positive")
	}
	if profile.CodeDim <= 0 {
		return CodebookTensorDescriptor{}, core.NewError("mlx: codebook code_dim must be positive")
	}
	if !validCodebookIndexBits(profile.IndexBits) {
		return CodebookTensorDescriptor{}, core.NewError(core.Sprintf("mlx: unsupported codebook index bits %d", profile.IndexBits))
	}
	elements := shape[0] * shape[1]
	if elements%uint64(profile.CodeDim) != 0 {
		return CodebookTensorDescriptor{}, core.NewError(core.Sprintf("mlx: codebook tensor elements %d must be divisible by code_dim %d", elements, profile.CodeDim))
	}
	codeCount := int(elements / uint64(profile.CodeDim))
	return CodebookTensorDescriptor{
		Name:          name,
		Format:        profile.Format,
		Shape:         append([]uint64(nil), shape...),
		Elements:      elements,
		CodebookSize:  profile.CodebookSize,
		CodeDim:       profile.CodeDim,
		CodeCount:     codeCount,
		IndexBits:     profile.IndexBits,
		IndexBytes:    (codeCount*profile.IndexBits + 7) / 8,
		CodesName:     defaultCodebookCodesName(name),
		CodebookName:  defaultCodebookTableName(name),
		CodesShape:    []uint64{uint64(codeCount)},
		CodebookShape: []uint64{uint64(profile.CodebookSize), uint64(profile.CodeDim)},
	}, nil
}

// ValidateCodebookQuantizationProfile checks global and tensor-local VQ metadata.
func ValidateCodebookQuantizationProfile(profile CodebookQuantizationProfile) error {
	if profile.Type != "" && profile.Type != CodebookQuantizationType {
		return core.NewError("mlx: unsupported codebook type: " + profile.Type)
	}
	if profile.Format != "" && profile.Format != CodebookFormatVQ {
		return core.NewError("mlx: unsupported codebook format: " + profile.Format)
	}
	if profile.CodebookSize <= 0 {
		return core.NewError("mlx: codebook size must be positive")
	}
	if profile.CodeDim <= 0 {
		return core.NewError("mlx: codebook code_dim must be positive")
	}
	if !validCodebookIndexBits(firstPositive(profile.IndexBits, 8)) {
		return core.NewError(core.Sprintf("mlx: unsupported codebook index bits %d", profile.IndexBits))
	}
	for _, tensor := range profile.Tensors {
		if err := ValidateCodebookTensorDescriptor(tensor); err != nil {
			return err
		}
	}
	return nil
}

// ValidateCodebookTensorDescriptor checks a tensor descriptor without payloads.
func ValidateCodebookTensorDescriptor(desc CodebookTensorDescriptor) error {
	if desc.Name == "" {
		return core.NewError("mlx: codebook tensor name is required")
	}
	if desc.Format != CodebookFormatVQ {
		return core.NewError("mlx: codebook tensor format must be vq")
	}
	if len(desc.Shape) != 2 || desc.Shape[0] == 0 || desc.Shape[1] == 0 {
		return core.NewError("mlx: codebook tensor shape must be [out, in]")
	}
	if desc.CodebookSize <= 0 || desc.CodeDim <= 0 || desc.CodeCount <= 0 {
		return core.NewError("mlx: codebook tensor requires codebook_size, code_dim, and code_count")
	}
	if !validCodebookIndexBits(desc.IndexBits) {
		return core.NewError(core.Sprintf("mlx: unsupported codebook index bits %d", desc.IndexBits))
	}
	if desc.Elements != desc.Shape[0]*desc.Shape[1] {
		return core.NewError("mlx: codebook tensor element count does not match shape")
	}
	if int(desc.Elements/uint64(desc.CodeDim)) != desc.CodeCount {
		return core.NewError("mlx: codebook tensor code count does not match code_dim")
	}
	return nil
}

// CodebookVQMatVec computes input @ dequantized(weight).T plus optional bias.
// Input is flattened rows of width desc.Shape[1]; output is flattened rows of
// width desc.Shape[0].
func CodebookVQMatVec(desc CodebookTensorDescriptor, input []float32, codes []uint32, codebook []float32, bias []float32) ([]float32, error) {
	if err := ValidateCodebookTensorPayload(desc, codes, codebook, bias); err != nil {
		return nil, err
	}
	outDim := int(desc.Shape[0])
	inDim := int(desc.Shape[1])
	if len(input) == 0 || len(input)%inDim != 0 {
		return nil, core.NewError(core.Sprintf("mlx: codebook matvec input length %d is not divisible by input width %d", len(input), inDim))
	}
	rows := len(input) / inDim
	out := make([]float32, rows*outDim)
	for row := 0; row < rows; row++ {
		for outCol := 0; outCol < outDim; outCol++ {
			sum := float32(0)
			for inCol := 0; inCol < inDim; inCol++ {
				weightIndex := outCol*inDim + inCol
				codeIndex := weightIndex / desc.CodeDim
				codeOffset := weightIndex % desc.CodeDim
				codeID := codes[codeIndex]
				weight := codebook[int(codeID)*desc.CodeDim+codeOffset]
				sum += input[row*inDim+inCol] * weight
			}
			if len(bias) > 0 {
				sum += bias[outCol]
			}
			out[row*outDim+outCol] = sum
		}
	}
	return out, nil
}

// ValidateCodebookTensorPayload checks VQ code/codebook/bias buffers.
func ValidateCodebookTensorPayload(desc CodebookTensorDescriptor, codes []uint32, codebook []float32, bias []float32) error {
	if err := ValidateCodebookTensorDescriptor(desc); err != nil {
		return err
	}
	if len(codes) != desc.CodeCount {
		return core.NewError(core.Sprintf("mlx: codebook code count %d, expected %d", len(codes), desc.CodeCount))
	}
	if len(codebook) != desc.CodebookSize*desc.CodeDim {
		return core.NewError(core.Sprintf("mlx: codebook value count %d, expected %d", len(codebook), desc.CodebookSize*desc.CodeDim))
	}
	for i, codeID := range codes {
		if codeID >= uint32(desc.CodebookSize) {
			return core.NewError(core.Sprintf("mlx: codebook code id %d at index %d exceeds codebook size %d", codeID, i, desc.CodebookSize))
		}
	}
	if len(bias) > 0 && len(bias) != int(desc.Shape[0]) {
		return core.NewError(core.Sprintf("mlx: codebook bias length %d, expected %d", len(bias), desc.Shape[0]))
	}
	return nil
}

func readCodebookQuantizationProfile(root string) (*CodebookQuantizationProfile, error) {
	read := core.ReadFile(core.PathJoin(root, "codebook_config.json"))
	if !read.OK {
		if core.IsNotExist(read.Value.(error)) {
			return nil, nil
		}
		return nil, read.Value.(error)
	}
	return ParseCodebookQuantizationProfile(read.Value.([]byte))
}

func cloneCodebookQuantizationProfile(profile *CodebookQuantizationProfile) *CodebookQuantizationProfile {
	if profile == nil {
		return nil
	}
	cloned := *profile
	cloned.Tensors = append([]CodebookTensorDescriptor(nil), profile.Tensors...)
	for i := range cloned.Tensors {
		cloned.Tensors[i].Shape = append([]uint64(nil), profile.Tensors[i].Shape...)
		cloned.Tensors[i].CodesShape = append([]uint64(nil), profile.Tensors[i].CodesShape...)
		cloned.Tensors[i].CodebookShape = append([]uint64(nil), profile.Tensors[i].CodebookShape...)
	}
	return &cloned
}

func validCodebookIndexBits(bits int) bool {
	switch bits {
	case 8, 16, 32:
		return true
	default:
		return false
	}
}

func defaultCodebookCodesName(name string) string {
	return name + ".codes"
}

func defaultCodebookTableName(name string) string {
	return name + ".codebook"
}
