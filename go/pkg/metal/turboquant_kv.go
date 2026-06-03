// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import core "dappco.re/go"

const (
	// TurboQuantKVLayoutVersion is the first on-disk/in-State physical schema
	// for compressed K/V pages. Older snapshot families must fail closed rather
	// than guess this layout.
	TurboQuantKVLayoutVersion = 1
	TurboQuantKVCodecName     = "turboquant-kv-v1"
)

type TurboQuantKVAlgorithm string

const (
	TurboQuantKVAlgorithmMSE  TurboQuantKVAlgorithm = "turboquantmse"
	TurboQuantKVAlgorithmProd TurboQuantKVAlgorithm = "turboquantprod"
)

const (
	TurboQuantKVOutlierPolicyHighHalfHeadDimV1 = "high-half-head-dim-v1"
	TurboQuantKVOutlierPolicyExplicitMaskV1    = "explicit-mask-v1"
)

const (
	TurboQuantKVNormPolicyExplicitVectorBF16V1         = "explicit-vector-norm-bf16-v1"
	TurboQuantKVResidualNormPolicyExplicitVectorBF16V1 = "explicit-vector-residual-norm-bf16-v1"
)

// TurboQuantKVShape is the logical MLX cache tensor shape. Compression changes
// the physical payload, not this rank-4 view.
type TurboQuantKVShape struct {
	Batch   int32 `json:"batch"`
	Heads   int32 `json:"heads"`
	SeqLen  int32 `json:"seq_len"`
	HeadDim int32 `json:"head_dim"`
}

func (shape TurboQuantKVShape) ElementCount() int64 {
	if !shape.Valid() {
		return 0
	}
	return int64(shape.Batch) * int64(shape.Heads) * int64(shape.SeqLen) * int64(shape.HeadDim)
}

func (shape TurboQuantKVShape) Valid() bool {
	return shape.Batch > 0 && shape.Heads > 0 && shape.SeqLen > 0 && shape.HeadDim > 0
}

// TurboQuantKVCodec describes one side of a compressed K/V page. Keys should
// use TurboQuantprod; values start with TurboQuantmse.
type TurboQuantKVCodec struct {
	Algorithm          TurboQuantKVAlgorithm `json:"algorithm"`
	NormalBits         int                   `json:"normal_bits"`
	OutlierBits        int                   `json:"outlier_bits,omitempty"`
	OutlierPolicy      string                `json:"outlier_policy,omitempty"`
	OutlierMask        []byte                `json:"outlier_mask,omitempty"`
	NormPolicy         string                `json:"norm_policy,omitempty"`
	ResidualNormPolicy string                `json:"residual_norm_policy,omitempty"`
	RotationSeed       uint64                `json:"rotation_seed"`
	QJLSeed            uint64                `json:"qjl_seed,omitempty"`
	CodebookID         string                `json:"codebook_id"`
}

func (codec TurboQuantKVCodec) Validate(kind string, headDim int32) error {
	if codec.Algorithm != TurboQuantKVAlgorithmMSE && codec.Algorithm != TurboQuantKVAlgorithmProd {
		return core.NewError("mlx: TurboQuant " + kind + " algorithm is invalid")
	}
	if codec.NormalBits <= 0 {
		return core.NewError("mlx: TurboQuant " + kind + " normal bit width is invalid")
	}
	if codec.NormalBits > 8 {
		return core.NewError("mlx: TurboQuant " + kind + " normal bit width exceeds byte storage")
	}
	if len(codec.OutlierMask) > 0 && codec.OutlierBits <= 0 {
		return core.NewError("mlx: TurboQuant " + kind + " outlier bit width is invalid")
	}
	if codec.OutlierBits > 8 {
		return core.NewError("mlx: TurboQuant " + kind + " outlier bit width exceeds byte storage")
	}
	if len(codec.OutlierMask) > 0 && codec.OutlierPolicy == "" {
		return core.NewError("mlx: TurboQuant " + kind + " outlier policy is missing")
	}
	if headDim <= 0 {
		return core.NewError("mlx: TurboQuant " + kind + " head dimension is invalid")
	}
	if len(codec.OutlierMask) > 0 && len(codec.OutlierMask) != turboQuantKVMaskBytes(headDim) {
		return core.NewError("mlx: TurboQuant " + kind + " outlier mask length is invalid")
	}
	if codec.OutlierPolicy != "" && codec.OutlierPolicy != TurboQuantKVOutlierPolicyHighHalfHeadDimV1 && codec.OutlierPolicy != TurboQuantKVOutlierPolicyExplicitMaskV1 {
		return core.NewError("mlx: TurboQuant " + kind + " outlier policy is unsupported")
	}
	if codec.OutlierPolicy == TurboQuantKVOutlierPolicyHighHalfHeadDimV1 {
		want := turboQuantKVOutlierMask(headDim, codec.OutlierChannels(headDim))
		if !turboQuantKVBytesEqual(codec.OutlierMask, want) {
			return core.NewError("mlx: TurboQuant " + kind + " outlier mask does not match high-half policy")
		}
	}
	if codec.NormPolicy == "" {
		return core.NewError("mlx: TurboQuant " + kind + " norm policy is missing")
	}
	if codec.NormPolicy != TurboQuantKVNormPolicyExplicitVectorBF16V1 {
		return core.NewError("mlx: TurboQuant " + kind + " norm policy is unsupported")
	}
	if codec.Algorithm == TurboQuantKVAlgorithmProd {
		if codec.ResidualNormPolicy == "" {
			return core.NewError("mlx: TurboQuant " + kind + " residual norm policy is missing")
		}
		if codec.ResidualNormPolicy != TurboQuantKVResidualNormPolicyExplicitVectorBF16V1 {
			return core.NewError("mlx: TurboQuant " + kind + " residual norm policy is unsupported")
		}
	} else if codec.ResidualNormPolicy != "" {
		return core.NewError("mlx: TurboQuant " + kind + " residual norm policy is only valid for TurboQuantprod")
	}
	if codec.RotationSeed == 0 {
		return core.NewError("mlx: TurboQuant " + kind + " rotation seed is missing")
	}
	if codec.Algorithm == TurboQuantKVAlgorithmProd && codec.QJLSeed == 0 {
		return core.NewError("mlx: TurboQuant " + kind + " QJL seed is missing")
	}
	if codec.CodebookID == "" {
		return core.NewError("mlx: TurboQuant " + kind + " codebook id is missing")
	}
	return nil
}

func (codec TurboQuantKVCodec) OutlierChannels(headDim int32) int32 {
	if headDim <= 0 || len(codec.OutlierMask) == 0 {
		return 0
	}
	var count int32
	for i := range headDim {
		if codec.OutlierMask[i/8]&(1<<uint(i%8)) != 0 {
			count++
		}
	}
	return count
}

func (codec TurboQuantKVCodec) EffectiveBitsMilli(headDim int32) int {
	if headDim <= 0 || codec.NormalBits <= 0 {
		return 0
	}
	outliers := int(codec.OutlierChannels(headDim))
	normal := int(headDim) - outliers
	outlierBits := codec.OutlierBits
	if outlierBits <= 0 {
		outlierBits = codec.NormalBits
	}
	totalMilli := (normal*codec.NormalBits + outliers*outlierBits) * 1000
	return totalMilli / int(headDim)
}

func (codec TurboQuantKVCodec) bitsForChannel(channel int32) int {
	if channel < 0 || len(codec.OutlierMask) == 0 {
		return codec.NormalBits
	}
	byteIndex := channel / 8
	bitIndex := uint(channel % 8)
	if int(byteIndex) < len(codec.OutlierMask) && codec.OutlierMask[byteIndex]&(1<<bitIndex) != 0 && codec.OutlierBits > 0 {
		return codec.OutlierBits
	}
	return codec.NormalBits
}

// TurboQuantKVPageLayout is the versioned metadata contract for one compressed
// K/V page. The payload bytes are deliberately separate so State files can index
// pages without materialising the full context.
type TurboQuantKVPageLayout struct {
	Version     int               `json:"version"`
	Codec       string            `json:"codec"`
	CacheIndex  int               `json:"cache_index"`
	Layer       int               `json:"layer"`
	LayerType   string            `json:"layer_type"`
	SharedOwner int               `json:"shared_owner"`
	Shape       TurboQuantKVShape `json:"shape"`
	TokenOffset int               `json:"token_offset"`
	PageTokens  int               `json:"page_tokens"`
	PageSize    int               `json:"page_size"`
	LocalWindow int               `json:"local_window,omitempty"`
	Key         TurboQuantKVCodec `json:"key"`
	Value       TurboQuantKVCodec `json:"value"`
}

// TurboQuantKVPagePayloadEstimate counts the compressed binary payload for one
// K/V page. It includes the side channels needed by the paper path (QJL signs
// and norms) so memory reports do not compare centroid bytes against fp16.
type TurboQuantKVPagePayloadEstimate struct {
	PageVectors          uint64  `json:"page_vectors"`
	PageElements         uint64  `json:"page_elements"`
	KeyCentroidBytes     uint64  `json:"key_centroid_bytes"`
	KeyQJLSignBytes      uint64  `json:"key_qjl_sign_bytes,omitempty"`
	KeyNormBytes         uint64  `json:"key_norm_bytes"`
	KeyResidualNormBytes uint64  `json:"key_residual_norm_bytes,omitempty"`
	ValueCentroidBytes   uint64  `json:"value_centroid_bytes"`
	ValueNormBytes       uint64  `json:"value_norm_bytes"`
	OutlierMaskBytes     uint64  `json:"outlier_mask_bytes,omitempty"`
	TotalBytes           uint64  `json:"total_bytes"`
	FP16BaselineBytes    uint64  `json:"fp16_baseline_bytes"`
	SavingsRatio         float64 `json:"savings_ratio,omitempty"`
}

func (layout TurboQuantKVPageLayout) PageVectorCount() uint64 {
	if !layout.Shape.Valid() || layout.PageTokens <= 0 {
		return 0
	}
	return uint64(layout.Shape.Batch) * uint64(layout.Shape.Heads) * uint64(layout.PageTokens)
}

func (layout TurboQuantKVPageLayout) PageElementCount() uint64 {
	vectors := layout.PageVectorCount()
	if vectors == 0 || layout.Shape.HeadDim <= 0 {
		return 0
	}
	return vectors * uint64(layout.Shape.HeadDim)
}

func (layout TurboQuantKVPageLayout) EstimatePayloadBytes() (TurboQuantKVPagePayloadEstimate, error) {
	if err := layout.Validate(); err != nil {
		return TurboQuantKVPagePayloadEstimate{}, err
	}
	vectors := layout.PageVectorCount()
	elements := layout.PageElementCount()
	keyCentroidBytesPerVector := turboQuantKVPackedBytes(layout.Key.centroidBitsPerVector(layout.Shape.HeadDim))
	keyQJLBytesPerVector := turboQuantKVPackedBytes(uint64(layout.Shape.HeadDim))
	valueCentroidBytesPerVector := turboQuantKVPackedBytes(layout.Value.centroidBitsPerVector(layout.Shape.HeadDim))
	estimate := TurboQuantKVPagePayloadEstimate{
		PageVectors:        vectors,
		PageElements:       elements,
		KeyCentroidBytes:   vectors * keyCentroidBytesPerVector,
		KeyNormBytes:       vectors * turboQuantKVNormBytesPerVector,
		ValueCentroidBytes: vectors * valueCentroidBytesPerVector,
		ValueNormBytes:     vectors * turboQuantKVNormBytesPerVector,
		OutlierMaskBytes:   uint64(len(layout.Key.OutlierMask) + len(layout.Value.OutlierMask)),
		FP16BaselineBytes:  elements * 2 * 2,
	}
	if layout.Key.Algorithm == TurboQuantKVAlgorithmProd {
		estimate.KeyQJLSignBytes = vectors * keyQJLBytesPerVector
		estimate.KeyResidualNormBytes = vectors * turboQuantKVNormBytesPerVector
	}
	estimate.TotalBytes = estimate.KeyCentroidBytes +
		estimate.KeyQJLSignBytes +
		estimate.KeyNormBytes +
		estimate.KeyResidualNormBytes +
		estimate.ValueCentroidBytes +
		estimate.ValueNormBytes +
		estimate.OutlierMaskBytes
	if estimate.FP16BaselineBytes > 0 {
		estimate.SavingsRatio = float64(estimate.TotalBytes) / float64(estimate.FP16BaselineBytes)
	}
	return estimate, nil
}

func (layout TurboQuantKVPageLayout) Validate() error {
	if layout.Version != TurboQuantKVLayoutVersion {
		return core.NewError(core.Sprintf("mlx: TurboQuant KV layout version %d is unsupported", layout.Version))
	}
	if layout.Codec != TurboQuantKVCodecName {
		return core.NewError("mlx: TurboQuant KV codec is invalid")
	}
	if layout.CacheIndex < 0 || layout.Layer < 0 || layout.SharedOwner < 0 {
		return core.NewError("mlx: TurboQuant KV layer identity is invalid")
	}
	if layout.LayerType == "" {
		return core.NewError("mlx: TurboQuant KV layer type is missing")
	}
	if !layout.Shape.Valid() {
		return core.NewError("mlx: TurboQuant KV shape is invalid")
	}
	if layout.TokenOffset < 0 || layout.PageTokens <= 0 || layout.PageSize <= 0 {
		return core.NewError("mlx: TurboQuant KV page range is invalid")
	}
	if layout.PageTokens > layout.PageSize || int32(layout.PageTokens) > layout.Shape.SeqLen {
		return core.NewError("mlx: TurboQuant KV page tokens exceed shape")
	}
	if layout.LocalWindow < 0 {
		return core.NewError("mlx: TurboQuant KV local window is invalid")
	}
	if layout.Key.Algorithm != TurboQuantKVAlgorithmProd {
		return core.NewError("mlx: TurboQuant KV keys require TurboQuantprod")
	}
	if err := layout.Key.Validate("key", layout.Shape.HeadDim); err != nil {
		return err
	}
	if layout.Value.Algorithm != TurboQuantKVAlgorithmMSE {
		return core.NewError("mlx: TurboQuant KV values require TurboQuantmse")
	}
	if err := layout.Value.Validate("value", layout.Shape.HeadDim); err != nil {
		return err
	}
	return nil
}

const turboQuantKVNormBytesPerVector = 2

func (codec TurboQuantKVCodec) centroidBitsPerVector(headDim int32) uint64 {
	if headDim <= 0 || codec.NormalBits <= 0 {
		return 0
	}
	outliers := uint64(codec.OutlierChannels(headDim))
	normal := uint64(headDim) - outliers
	outlierBits := codec.OutlierBits
	if outlierBits <= 0 {
		outlierBits = codec.NormalBits
	}
	return normal*uint64(codec.NormalBits) + outliers*uint64(outlierBits)
}

func turboQuantKVPackedBytes(bits uint64) uint64 {
	if bits == 0 {
		return 0
	}
	return (bits + 7) / 8
}

func turboQuantKVMaskBytes(headDim int32) int {
	if headDim <= 0 {
		return 0
	}
	return int((headDim + 7) / 8)
}

func turboQuantKVOutlierMask(headDim int32, outlierChannels int32) []byte {
	if headDim <= 0 || outlierChannels <= 0 {
		return nil
	}
	if outlierChannels > headDim {
		outlierChannels = headDim
	}
	mask := make([]byte, turboQuantKVMaskBytes(headDim))
	start := headDim - outlierChannels
	for channel := start; channel < headDim; channel++ {
		mask[channel/8] |= 1 << uint(channel%8)
	}
	return mask
}

func turboQuantKVBytesEqual(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for idx := range a {
		if a[idx] != b[idx] {
			return false
		}
	}
	return true
}
