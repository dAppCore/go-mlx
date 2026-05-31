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
	Algorithm    TurboQuantKVAlgorithm `json:"algorithm"`
	NormalBits   int                   `json:"normal_bits"`
	OutlierBits  int                   `json:"outlier_bits,omitempty"`
	OutlierMask  []byte                `json:"outlier_mask,omitempty"`
	RotationSeed uint64                `json:"rotation_seed"`
	QJLSeed      uint64                `json:"qjl_seed,omitempty"`
	CodebookID   string                `json:"codebook_id"`
}

func (codec TurboQuantKVCodec) Validate(kind string, headDim int32) error {
	if codec.Algorithm != TurboQuantKVAlgorithmMSE && codec.Algorithm != TurboQuantKVAlgorithmProd {
		return core.NewError("mlx: TurboQuant " + kind + " algorithm is invalid")
	}
	if codec.NormalBits <= 0 {
		return core.NewError("mlx: TurboQuant " + kind + " normal bit width is invalid")
	}
	if len(codec.OutlierMask) > 0 && codec.OutlierBits <= 0 {
		return core.NewError("mlx: TurboQuant " + kind + " outlier bit width is invalid")
	}
	if headDim <= 0 {
		return core.NewError("mlx: TurboQuant " + kind + " head dimension is invalid")
	}
	if len(codec.OutlierMask) > 0 && len(codec.OutlierMask) != turboQuantKVMaskBytes(headDim) {
		return core.NewError("mlx: TurboQuant " + kind + " outlier mask length is invalid")
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
	for i := int32(0); i < headDim; i++ {
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
	estimate := TurboQuantKVPagePayloadEstimate{
		PageVectors:        vectors,
		PageElements:       elements,
		KeyCentroidBytes:   turboQuantKVPackedBytes(vectors * layout.Key.centroidBitsPerVector(layout.Shape.HeadDim)),
		KeyNormBytes:       vectors * turboQuantKVNormBytesPerVector,
		ValueCentroidBytes: turboQuantKVPackedBytes(vectors * layout.Value.centroidBitsPerVector(layout.Shape.HeadDim)),
		ValueNormBytes:     vectors * turboQuantKVNormBytesPerVector,
		OutlierMaskBytes:   uint64(len(layout.Key.OutlierMask) + len(layout.Value.OutlierMask)),
		FP16BaselineBytes:  elements * 2 * 2,
	}
	if layout.Key.Algorithm == TurboQuantKVAlgorithmProd {
		estimate.KeyQJLSignBytes = turboQuantKVPackedBytes(elements)
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
