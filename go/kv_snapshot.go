// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"encoding/binary"
	"math"

	core "dappco.re/go"
)

const (
	// KVSnapshotVersion is the on-disk binary format version for KV snapshots.
	KVSnapshotVersion = 3

	kvSnapshotMagic = "MLXKV001"
)

// KVSnapshotEncoding controls how K/V tensors are represented on disk.
type KVSnapshotEncoding string

const (
	// KVSnapshotEncodingFloat32 preserves exact float32 K/V cache tensors.
	KVSnapshotEncodingFloat32 KVSnapshotEncoding = "float32"
	// KVSnapshotEncodingQ8 stores K/V cache tensors as symmetric int8 plus scale.
	KVSnapshotEncodingQ8 KVSnapshotEncoding = "q8"
)

// KVSnapshotSaveOptions controls the portable binary snapshot encoding.
type KVSnapshotSaveOptions struct {
	KVEncoding KVSnapshotEncoding
}

// KVSnapshot is a CPU-readable copy of model key/value cache tensors.
type KVSnapshot struct {
	Version       int
	Architecture  string
	Tokens        []int32
	Generated     []int32
	TokenOffset   int
	NumLayers     int
	NumHeads      int
	SeqLen        int
	HeadDim       int
	NumQueryHeads int
	LogitShape    []int32
	Logits        []float32
	Layers        []KVLayerSnapshot
}

// KVLayerSnapshot contains cache tensors for a logical transformer layer.
type KVLayerSnapshot struct {
	Layer      int
	CacheIndex int
	Heads      []KVHeadSnapshot
}

// KVHeadSnapshot contains flattened key/value tensors for one KV head.
type KVHeadSnapshot struct {
	Key   []float32
	Value []float32
}

// Head returns a defensive copy of the key/value tensors for layer and head.
func (s *KVSnapshot) Head(layer, head int) (KVHeadSnapshot, bool) {
	if s == nil || layer < 0 || head < 0 {
		return KVHeadSnapshot{}, false
	}
	layerSnapshot, ok := s.layer(layer)
	if !ok || head >= len(layerSnapshot.Heads) {
		return KVHeadSnapshot{}, false
	}
	return cloneKVHead(layerSnapshot.Heads[head]), true
}

func (s *KVSnapshot) layer(layer int) (KVLayerSnapshot, bool) {
	if layer < len(s.Layers) && s.Layers[layer].Layer == layer {
		return s.Layers[layer], true
	}
	for _, snapshot := range s.Layers {
		if snapshot.Layer == layer {
			return snapshot, true
		}
	}
	if layer < len(s.Layers) && s.Layers[layer].Layer == 0 {
		return s.Layers[layer], true
	}
	return KVLayerSnapshot{}, false
}

// Clone returns a deep copy of the snapshot.
func (s *KVSnapshot) Clone() *KVSnapshot {
	if s == nil {
		return nil
	}
	cloned := &KVSnapshot{
		Version:       s.Version,
		Architecture:  s.Architecture,
		Tokens:        append([]int32(nil), s.Tokens...),
		Generated:     append([]int32(nil), s.Generated...),
		TokenOffset:   s.TokenOffset,
		NumLayers:     s.NumLayers,
		NumHeads:      s.NumHeads,
		SeqLen:        s.SeqLen,
		HeadDim:       s.HeadDim,
		NumQueryHeads: s.NumQueryHeads,
		LogitShape:    append([]int32(nil), s.LogitShape...),
		Logits:        append([]float32(nil), s.Logits...),
		Layers:        cloneKVLayers(s.Layers),
	}
	return cloned
}

// Save writes the snapshot to path using the stable go-mlx KV binary format.
func (s *KVSnapshot) Save(path string) error {
	return s.SaveWithOptions(path, KVSnapshotSaveOptions{})
}

// SaveWithOptions writes the snapshot with explicit K/V tensor encoding.
func (s *KVSnapshot) SaveWithOptions(path string, opts KVSnapshotSaveOptions) error {
	if s == nil {
		return core.NewError("mlx: KV snapshot is nil")
	}
	data, err := s.bytesWithOptions(opts)
	if err != nil {
		return err
	}
	if result := core.WriteFile(path, data, 0o600); !result.OK {
		return core.E("KVSnapshot.Save", "write snapshot", kvSnapshotResultError(result))
	}
	return nil
}

// MarshalBinary returns the stable binary representation used by Save.
func (s *KVSnapshot) MarshalBinary() ([]byte, error) {
	if s == nil {
		return nil, core.NewError("mlx: KV snapshot is nil")
	}
	return s.bytesWithOptions(KVSnapshotSaveOptions{})
}

// UnmarshalBinary replaces the snapshot with data loaded from the stable binary format.
func (s *KVSnapshot) UnmarshalBinary(data []byte) error {
	if s == nil {
		return core.NewError("mlx: KV snapshot is nil")
	}
	loaded, err := parseKVSnapshot(data)
	if err != nil {
		return err
	}
	*s = *loaded
	return nil
}

// LoadKVSnapshot reads a KV snapshot saved by (*KVSnapshot).Save.
func LoadKVSnapshot(path string) (*KVSnapshot, error) {
	read := core.ReadFile(path)
	if !read.OK {
		return nil, core.E("LoadKVSnapshot", "read snapshot", kvSnapshotResultError(read))
	}
	data, ok := read.Value.([]byte)
	if !ok {
		return nil, core.E("LoadKVSnapshot", "read snapshot returned non-byte data", nil)
	}
	return parseKVSnapshot(data)
}

func (s *KVSnapshot) bytes() ([]byte, error) {
	return s.bytesWithOptions(KVSnapshotSaveOptions{})
}

func (s *KVSnapshot) bytesWithOptions(opts KVSnapshotSaveOptions) ([]byte, error) {
	encoding, err := normalizeKVSnapshotEncoding(opts.KVEncoding)
	if err != nil {
		return nil, err
	}
	data := []byte(kvSnapshotMagic)
	version := s.Version
	if version == 0 {
		version = KVSnapshotVersion
	}
	if encoding != KVSnapshotEncodingFloat32 && version < 3 {
		version = 3
	}
	if version <= 0 || version > KVSnapshotVersion {
		return nil, core.E("KVSnapshot.Save", "unsupported KV snapshot version", nil)
	}
	data = appendKVU32(data, uint32(version))
	if len(s.Architecture) > int(^uint32(0)) {
		return nil, core.E("KVSnapshot.Save", "architecture string too large", nil)
	}
	data = appendKVBytes(data, []byte(s.Architecture))
	data = appendKVU32(data, uint32(s.NumLayers))
	data = appendKVU32(data, uint32(s.NumHeads))
	data = appendKVU32(data, uint32(s.SeqLen))
	data = appendKVU32(data, uint32(s.HeadDim))
	data = appendKVU32(data, uint32(s.NumQueryHeads))
	if version >= 2 {
		tokenOffset := s.TokenOffset
		if tokenOffset == 0 {
			tokenOffset = len(s.Tokens)
		}
		data = appendKVU32(data, uint32(tokenOffset))
	}
	data = appendKVU32(data, uint32(len(s.Tokens)))
	for _, token := range s.Tokens {
		data = appendKVI32(data, token)
	}
	if version >= 2 {
		data = appendKVU32(data, uint32(len(s.Generated)))
		for _, token := range s.Generated {
			data = appendKVI32(data, token)
		}
	}
	data = appendKVU32(data, uint32(len(s.Layers)))
	for _, layer := range s.Layers {
		data = appendKVI32(data, int32(layer.Layer))
		data = appendKVI32(data, int32(layer.CacheIndex))
		data = appendKVU32(data, uint32(len(layer.Heads)))
		for _, head := range layer.Heads {
			if version >= 3 {
				data = appendKVEncodedF32s(data, head.Key, encoding)
				data = appendKVEncodedF32s(data, head.Value, encoding)
			} else {
				data = appendKVF32s(data, head.Key)
				data = appendKVF32s(data, head.Value)
			}
		}
	}
	if version >= 2 {
		data = appendKVU32(data, uint32(len(s.LogitShape)))
		for _, dim := range s.LogitShape {
			data = appendKVI32(data, dim)
		}
		data = appendKVF32s(data, s.Logits)
	}
	return data, nil
}

func normalizeKVSnapshotEncoding(encoding KVSnapshotEncoding) (KVSnapshotEncoding, error) {
	switch encoding {
	case "", KVSnapshotEncodingFloat32:
		return KVSnapshotEncodingFloat32, nil
	case KVSnapshotEncodingQ8:
		return KVSnapshotEncodingQ8, nil
	default:
		return "", core.E("KVSnapshot.Save", "unsupported KV snapshot encoding", nil)
	}
}

func parseKVSnapshot(data []byte) (*KVSnapshot, error) {
	reader := kvSnapshotReader{data: data}
	if magic := string(reader.read(len(kvSnapshotMagic))); magic != kvSnapshotMagic {
		return nil, core.E("LoadKVSnapshot", "invalid KV snapshot magic", nil)
	}
	version := int(reader.u32())
	if version <= 0 || version > KVSnapshotVersion {
		return nil, core.E("LoadKVSnapshot", "unsupported KV snapshot version", nil)
	}
	snapshot := &KVSnapshot{
		Version:       version,
		Architecture:  reader.string(),
		NumLayers:     int(reader.u32()),
		NumHeads:      int(reader.u32()),
		SeqLen:        int(reader.u32()),
		HeadDim:       int(reader.u32()),
		NumQueryHeads: int(reader.u32()),
	}
	if snapshot.Version >= 2 {
		snapshot.TokenOffset = int(reader.u32())
	}
	tokenCount := int(reader.u32())
	if tokenCount > 0 {
		snapshot.Tokens = make([]int32, tokenCount)
		for i := range snapshot.Tokens {
			snapshot.Tokens[i] = reader.i32()
		}
	}
	if snapshot.Version >= 2 {
		generatedCount := int(reader.u32())
		if generatedCount > 0 {
			snapshot.Generated = make([]int32, generatedCount)
			for i := range snapshot.Generated {
				snapshot.Generated[i] = reader.i32()
			}
		}
	}
	layerCount := int(reader.u32())
	if layerCount > 0 {
		snapshot.Layers = make([]KVLayerSnapshot, layerCount)
		for layerIdx := range snapshot.Layers {
			layer := &snapshot.Layers[layerIdx]
			layer.Layer = int(reader.i32())
			layer.CacheIndex = int(reader.i32())
			headCount := int(reader.u32())
			if headCount > 0 {
				layer.Heads = make([]KVHeadSnapshot, headCount)
				for headIdx := range layer.Heads {
					if snapshot.Version >= 3 {
						layer.Heads[headIdx].Key = reader.encodedF32s()
						layer.Heads[headIdx].Value = reader.encodedF32s()
					} else {
						layer.Heads[headIdx].Key = reader.f32s()
						layer.Heads[headIdx].Value = reader.f32s()
					}
				}
			}
		}
	}
	if snapshot.Version >= 2 {
		shapeCount := int(reader.u32())
		if shapeCount > 0 {
			snapshot.LogitShape = make([]int32, shapeCount)
			for i := range snapshot.LogitShape {
				snapshot.LogitShape[i] = reader.i32()
			}
		}
		snapshot.Logits = reader.f32s()
	}
	if reader.err != nil {
		return nil, core.E("LoadKVSnapshot", "parse snapshot", reader.err)
	}
	if snapshot.TokenOffset == 0 {
		snapshot.TokenOffset = len(snapshot.Tokens)
	}
	return snapshot, nil
}

func appendKVBytes(dst, src []byte) []byte {
	dst = appendKVU32(dst, uint32(len(src)))
	return append(dst, src...)
}

func appendKVU32(dst []byte, value uint32) []byte {
	var buf [4]byte
	binary.LittleEndian.PutUint32(buf[:], value)
	return append(dst, buf[:]...)
}

func appendKVI32(dst []byte, value int32) []byte {
	return appendKVU32(dst, uint32(value))
}

func appendKVF32s(dst []byte, values []float32) []byte {
	dst = appendKVU32(dst, uint32(len(values)))
	return appendKVF32Raw(dst, values)
}

func appendKVF32Raw(dst []byte, values []float32) []byte {
	for _, value := range values {
		dst = appendKVU32(dst, math.Float32bits(value))
	}
	return dst
}

func appendKVEncodedF32s(dst []byte, values []float32, encoding KVSnapshotEncoding) []byte {
	if encoding == KVSnapshotEncodingQ8 && kvSnapshotCanQuantizeQ8(values) {
		scale, quantized := quantizeKVSnapshotQ8(values)
		dst = appendKVU32(dst, 1)
		dst = appendKVU32(dst, uint32(len(values)))
		dst = appendKVU32(dst, math.Float32bits(scale))
		return append(dst, quantized...)
	}
	dst = appendKVU32(dst, 0)
	dst = appendKVU32(dst, uint32(len(values)))
	return appendKVF32Raw(dst, values)
}

func kvSnapshotCanQuantizeQ8(values []float32) bool {
	for _, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return false
		}
	}
	return true
}

func quantizeKVSnapshotQ8(values []float32) (float32, []byte) {
	var maxAbs float32
	for _, value := range values {
		abs := float32(math.Abs(float64(value)))
		if abs > maxAbs {
			maxAbs = abs
		}
	}
	scale := float32(1)
	if maxAbs > 0 {
		scale = maxAbs / 127
	}
	quantized := make([]byte, len(values))
	for i, value := range values {
		q := int(math.Round(float64(value / scale)))
		if q > 127 {
			q = 127
		}
		if q < -127 {
			q = -127
		}
		quantized[i] = byte(int8(q))
	}
	return scale, quantized
}

type kvSnapshotReader struct {
	data   []byte
	offset int
	err    error
}

func (r *kvSnapshotReader) read(n int) []byte {
	if r.err != nil {
		return nil
	}
	if n < 0 || len(r.data)-r.offset < n {
		r.err = core.NewError("mlx: truncated KV snapshot")
		return nil
	}
	chunk := r.data[r.offset : r.offset+n]
	r.offset += n
	return chunk
}

func (r *kvSnapshotReader) u32() uint32 {
	chunk := r.read(4)
	if chunk == nil {
		return 0
	}
	return binary.LittleEndian.Uint32(chunk)
}

func (r *kvSnapshotReader) i32() int32 {
	return int32(r.u32())
}

func (r *kvSnapshotReader) string() string {
	size := int(r.u32())
	return string(r.read(size))
}

func (r *kvSnapshotReader) f32s() []float32 {
	size := int(r.u32())
	values := make([]float32, size)
	for i := range values {
		values[i] = math.Float32frombits(r.u32())
	}
	return values
}

func (r *kvSnapshotReader) encodedF32s() []float32 {
	encoding := r.u32()
	size := int(r.u32())
	switch encoding {
	case 0:
		values := make([]float32, size)
		for i := range values {
			values[i] = math.Float32frombits(r.u32())
		}
		return values
	case 1:
		scale := math.Float32frombits(r.u32())
		raw := r.read(size)
		values := make([]float32, size)
		for i, value := range raw {
			values[i] = float32(int8(value)) * scale
		}
		return values
	default:
		r.err = core.NewError("mlx: unsupported KV tensor encoding")
		return nil
	}
}

func cloneKVLayers(src []KVLayerSnapshot) []KVLayerSnapshot {
	if len(src) == 0 {
		return nil
	}
	cloned := make([]KVLayerSnapshot, len(src))
	for i, layer := range src {
		cloned[i] = KVLayerSnapshot{
			Layer:      layer.Layer,
			CacheIndex: layer.CacheIndex,
			Heads:      cloneKVHeads(layer.Heads),
		}
	}
	return cloned
}

func cloneKVHeads(src []KVHeadSnapshot) []KVHeadSnapshot {
	if len(src) == 0 {
		return nil
	}
	cloned := make([]KVHeadSnapshot, len(src))
	for i, head := range src {
		cloned[i] = cloneKVHead(head)
	}
	return cloned
}

func cloneKVHead(src KVHeadSnapshot) KVHeadSnapshot {
	return KVHeadSnapshot{
		Key:   append([]float32(nil), src.Key...),
		Value: append([]float32(nil), src.Value...),
	}
}

func kvSnapshotResultError(result core.Result) error {
	if err, ok := result.Value.(error); ok {
		return err
	}
	if text, ok := result.Value.(string); ok {
		return core.NewError(text)
	}
	return core.NewError("unknown filesystem error")
}
