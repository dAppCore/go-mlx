// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"encoding/binary"
	stdio "io"
	"math"

	core "dappco.re/go"
	"dappco.re/go/mlx/safetensors"
)

const (
	// SnapshotVersion is the on-disk binary format version for KV snapshots.
	SnapshotVersion = 4

	kvSnapshotMagic = "MLXKV001"
)

// Encoding controls how K/V tensors are represented on disk.
type Encoding string

const (
	// KVSnapshotEncodingFloat32 preserves exact float32 K/V cache tensors.
	KVSnapshotEncodingFloat32 Encoding = "float32"
	// EncodingQ8 stores K/V cache tensors as symmetric int8 plus scale.
	EncodingQ8 Encoding = "q8"
	// EncodingNative stores K/V tensors in their captured dtype when
	// native dtype bytes are present, falling back to float32 otherwise.
	EncodingNative Encoding = "native"
)

// SaveOptions controls the portable binary snapshot encoding.
type SaveOptions struct {
	KVEncoding Encoding
}

// LoadOptions controls how portable binary snapshots are decoded.
type LoadOptions struct {
	// RawKVOnly preserves native K/V tensor bytes without decoding float32
	// side slices. Float32 and Q8 snapshot encodings still decode to float32.
	RawKVOnly bool
}

// CaptureOptions controls native K/V capture.
type CaptureOptions struct {
	// RawKVOnly captures native K/V dtype bytes without retaining float32
	// key/value slices when the native backend can provide raw tensors.
	RawKVOnly bool
}

// Snapshot is a CPU-readable copy of model key/value cache tensors.
type Snapshot struct {
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
	Layers        []LayerSnapshot
}

// LayerSnapshot contains cache tensors for a logical transformer layer.
type LayerSnapshot struct {
	Layer      int
	CacheIndex int
	KeyDType   string
	KeyBytes   []byte
	KeyShape   []int32
	ValueDType string
	ValueBytes []byte
	ValueShape []int32
	Heads      []HeadSnapshot
}

// HeadSnapshot contains flattened key/value tensors for one KV head.
type HeadSnapshot struct {
	Key        []float32
	KeyDType   string
	KeyBytes   []byte
	Value      []float32
	ValueDType string
	ValueBytes []byte
}

// Head returns a defensive copy of the key/value tensors for layer and head.
func (s *Snapshot) Head(layer, head int) (HeadSnapshot, bool) {
	if s == nil || layer < 0 || head < 0 {
		return HeadSnapshot{}, false
	}
	layerSnapshot, ok := s.layer(layer)
	if !ok || head >= len(layerSnapshot.Heads) {
		return HeadSnapshot{}, false
	}
	return cloneKVHead(layerSnapshot.Heads[head]), true
}

func (s *Snapshot) layer(layer int) (LayerSnapshot, bool) {
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
	return LayerSnapshot{}, false
}

// Clone returns a deep copy of the snapshot.
func (s *Snapshot) Clone() *Snapshot {
	if s == nil {
		return nil
	}
	cloned := &Snapshot{
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
func (s *Snapshot) Save(path string) error {
	return s.SaveWithOptions(path, SaveOptions{})
}

// SaveWithOptions writes the snapshot with explicit K/V tensor encoding.
func (s *Snapshot) SaveWithOptions(path string, opts SaveOptions) error {
	if s == nil {
		return core.NewError("mlx: KV snapshot is nil")
	}
	data, err := s.bytesWithOptions(opts)
	if err != nil {
		return err
	}
	if result := core.WriteFile(path, data, 0o600); !result.OK {
		return core.E("Snapshot.Save", "write snapshot", ResultError(result))
	}
	return nil
}

// MarshalBinary returns the stable binary representation used by Save.
func (s *Snapshot) MarshalBinary() ([]byte, error) {
	if s == nil {
		return nil, core.NewError("mlx: KV snapshot is nil")
	}
	return s.bytesWithOptions(SaveOptions{})
}

// UnmarshalBinary replaces the snapshot with data loaded from the stable binary format.
func (s *Snapshot) UnmarshalBinary(data []byte) error {
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

// Load reads a KV snapshot saved by (*Snapshot).Save.
func Load(path string) (*Snapshot, error) {
	return LoadWithOptions(path, LoadOptions{})
}

// LoadWithOptions reads a KV snapshot with explicit decode options.
func LoadWithOptions(path string, opts LoadOptions) (*Snapshot, error) {
	read := core.ReadFile(path)
	if !read.OK {
		return nil, core.E("Load", "read snapshot", ResultError(read))
	}
	data, ok := read.Value.([]byte)
	if !ok {
		return nil, core.E("Load", "read snapshot returned non-byte data", nil)
	}
	return parseKVSnapshotWithOptions(data, opts)
}

func (s *Snapshot) bytes() ([]byte, error) {
	return s.bytesWithOptions(SaveOptions{})
}

func (s *Snapshot) encodedSizeWithOptions(opts SaveOptions) (int, error) {
	encoding, err := normalizeKVSnapshotEncoding(opts.KVEncoding)
	if err != nil {
		return 0, err
	}
	version := effectiveVersion(s, encoding)
	if version <= 0 || version > SnapshotVersion {
		return 0, core.E("Snapshot.Save", "unsupported KV snapshot version", nil)
	}
	if len(s.Architecture) > int(^uint32(0)) {
		return 0, core.E("Snapshot.Save", "architecture string too large", nil)
	}
	size := len(kvSnapshotMagic)
	size += 4                       // version
	size += 4 + len(s.Architecture) // architecture
	size += 5 * 4                   // layers, heads, seq len, head dim, query heads
	size += 4 + len(s.Tokens)*4     // tokens
	size += 4                       // layer count
	if version >= 2 {
		size += 4                      // token offset
		size += 4 + len(s.Generated)*4 // generated tokens
	}
	for _, layer := range s.Layers {
		size += 12 // layer, cache index, head count
		if version >= 4 {
			keySize, err := kvSnapshotEncodedTensorSize(nil, layer.KeyDType, layer.KeyBytes, encoding)
			if err != nil {
				return 0, core.E("Snapshot.Save", "encode layer key tensor", err)
			}
			valueSize, err := kvSnapshotEncodedTensorSize(nil, layer.ValueDType, layer.ValueBytes, encoding)
			if err != nil {
				return 0, core.E("Snapshot.Save", "encode layer value tensor", err)
			}
			size += 4 + len(layer.KeyShape)*4
			size += keySize
			size += 4 + len(layer.ValueShape)*4
			size += valueSize
		}
		for _, head := range layer.Heads {
			if version >= 3 {
				keySize, err := kvSnapshotEncodedTensorSize(head.Key, head.KeyDType, head.KeyBytes, encoding)
				if err != nil {
					return 0, core.E("Snapshot.Save", "encode key tensor", err)
				}
				valueSize, err := kvSnapshotEncodedTensorSize(head.Value, head.ValueDType, head.ValueBytes, encoding)
				if err != nil {
					return 0, core.E("Snapshot.Save", "encode value tensor", err)
				}
				size += keySize + valueSize
			} else {
				size += 4 + len(head.Key)*4
				size += 4 + len(head.Value)*4
			}
		}
	}
	if version >= 2 {
		size += 4 + len(s.LogitShape)*4
		size += 4 + len(s.Logits)*4
	}
	return size, nil
}

func (s *Snapshot) bytesWithOptions(opts SaveOptions) ([]byte, error) {
	encoding, err := normalizeKVSnapshotEncoding(opts.KVEncoding)
	if err != nil {
		return nil, err
	}
	size, err := s.encodedSizeWithOptions(opts)
	if err != nil {
		return nil, err
	}
	data := make([]byte, 0, size)
	data = append(data, kvSnapshotMagic...)
	version := effectiveVersion(s, encoding)
	if version <= 0 || version > SnapshotVersion {
		return nil, core.E("Snapshot.Save", "unsupported KV snapshot version", nil)
	}
	data = appendKVU32(data, uint32(version))
	if len(s.Architecture) > int(^uint32(0)) {
		return nil, core.E("Snapshot.Save", "architecture string too large", nil)
	}
	data = appendKVBytes(data, core.AsBytes(s.Architecture))
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
		if version >= 4 {
			data = appendKVI32s(data, layer.KeyShape)
			data, err = appendKVEncodedTensor(data, nil, layer.KeyDType, layer.KeyBytes, encoding)
			if err != nil {
				return nil, core.E("Snapshot.Save", "encode layer key tensor", err)
			}
			data = appendKVI32s(data, layer.ValueShape)
			data, err = appendKVEncodedTensor(data, nil, layer.ValueDType, layer.ValueBytes, encoding)
			if err != nil {
				return nil, core.E("Snapshot.Save", "encode layer value tensor", err)
			}
		}
		for _, head := range layer.Heads {
			if version >= 3 {
				data, err = appendKVEncodedTensor(data, head.Key, head.KeyDType, head.KeyBytes, encoding)
				if err != nil {
					return nil, core.E("Snapshot.Save", "encode key tensor", err)
				}
				data, err = appendKVEncodedTensor(data, head.Value, head.ValueDType, head.ValueBytes, encoding)
				if err != nil {
					return nil, core.E("Snapshot.Save", "encode value tensor", err)
				}
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

func (s *Snapshot) writeWithOptions(writer stdio.Writer, opts SaveOptions) error {
	encoding, err := normalizeKVSnapshotEncoding(opts.KVEncoding)
	if err != nil {
		return err
	}
	if _, err := s.encodedSizeWithOptions(opts); err != nil {
		return err
	}
	version := effectiveVersion(s, encoding)
	stream := kvSnapshotStreamWriter{writer: writer}
	stream.bytes(core.AsBytes(kvSnapshotMagic))
	stream.u32(uint32(version))
	stream.bytesWithLength(core.AsBytes(s.Architecture))
	stream.u32(uint32(s.NumLayers))
	stream.u32(uint32(s.NumHeads))
	stream.u32(uint32(s.SeqLen))
	stream.u32(uint32(s.HeadDim))
	stream.u32(uint32(s.NumQueryHeads))
	if version >= 2 {
		tokenOffset := s.TokenOffset
		if tokenOffset == 0 {
			tokenOffset = len(s.Tokens)
		}
		stream.u32(uint32(tokenOffset))
	}
	stream.u32(uint32(len(s.Tokens)))
	for _, token := range s.Tokens {
		stream.i32(token)
	}
	if version >= 2 {
		stream.u32(uint32(len(s.Generated)))
		for _, token := range s.Generated {
			stream.i32(token)
		}
	}
	stream.u32(uint32(len(s.Layers)))
	for _, layer := range s.Layers {
		stream.i32(int32(layer.Layer))
		stream.i32(int32(layer.CacheIndex))
		stream.u32(uint32(len(layer.Heads)))
		if version >= 4 {
			stream.i32s(layer.KeyShape)
			if err := stream.encodedTensor(nil, layer.KeyDType, layer.KeyBytes, encoding); err != nil {
				return core.E("Snapshot.Save", "encode layer key tensor", err)
			}
			stream.i32s(layer.ValueShape)
			if err := stream.encodedTensor(nil, layer.ValueDType, layer.ValueBytes, encoding); err != nil {
				return core.E("Snapshot.Save", "encode layer value tensor", err)
			}
		}
		for _, head := range layer.Heads {
			if version >= 3 {
				if err := stream.encodedTensor(head.Key, head.KeyDType, head.KeyBytes, encoding); err != nil {
					return core.E("Snapshot.Save", "encode key tensor", err)
				}
				if err := stream.encodedTensor(head.Value, head.ValueDType, head.ValueBytes, encoding); err != nil {
					return core.E("Snapshot.Save", "encode value tensor", err)
				}
			} else {
				stream.f32s(head.Key)
				stream.f32s(head.Value)
			}
		}
	}
	if version >= 2 {
		stream.u32(uint32(len(s.LogitShape)))
		for _, dim := range s.LogitShape {
			stream.i32(dim)
		}
		stream.f32s(s.Logits)
	}
	return stream.err
}

func normalizeKVSnapshotEncoding(encoding Encoding) (Encoding, error) {
	switch encoding {
	case "", KVSnapshotEncodingFloat32:
		return KVSnapshotEncodingFloat32, nil
	case EncodingQ8, EncodingNative:
		return encoding, nil
	default:
		return "", core.E("Snapshot.Save", "unsupported KV snapshot encoding", nil)
	}
}

func parseKVSnapshot(data []byte) (*Snapshot, error) {
	return parseKVSnapshotWithOptions(data, LoadOptions{})
}

func parseKVSnapshotWithOptions(data []byte, opts LoadOptions) (*Snapshot, error) {
	reader := kvSnapshotReader{data: data}
	if magic := string(reader.read(len(kvSnapshotMagic))); magic != kvSnapshotMagic {
		return nil, core.E("Load", "invalid KV snapshot magic", nil)
	}
	version := int(reader.u32())
	if version <= 0 || version > SnapshotVersion {
		return nil, core.E("Load", "unsupported KV snapshot version", nil)
	}
	snapshot := &Snapshot{
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
		// Batch the i32 block read so bounds check is paid once.
		chunk := reader.read(tokenCount * 4)
		if chunk != nil {
			snapshot.Tokens = make([]int32, tokenCount)
			for i := range snapshot.Tokens {
				snapshot.Tokens[i] = int32(binary.LittleEndian.Uint32(chunk[i*4:]))
			}
		}
	}
	if snapshot.Version >= 2 {
		generatedCount := int(reader.u32())
		if generatedCount > 0 {
			chunk := reader.read(generatedCount * 4)
			if chunk != nil {
				snapshot.Generated = make([]int32, generatedCount)
				for i := range snapshot.Generated {
					snapshot.Generated[i] = int32(binary.LittleEndian.Uint32(chunk[i*4:]))
				}
			}
		}
	}
	layerCount := int(reader.u32())
	if layerCount > 0 {
		snapshot.Layers = make([]LayerSnapshot, layerCount)
		for layerIdx := range snapshot.Layers {
			layer := &snapshot.Layers[layerIdx]
			layer.Layer = int(reader.i32())
			layer.CacheIndex = int(reader.i32())
			headCount := int(reader.u32())
			if snapshot.Version >= 4 {
				layer.KeyShape = reader.i32s()
				key := reader.encodedTensor(LoadOptions{RawKVOnly: true})
				layer.KeyDType = key.DType
				layer.KeyBytes = key.Bytes
				layer.ValueShape = reader.i32s()
				value := reader.encodedTensor(LoadOptions{RawKVOnly: true})
				layer.ValueDType = value.DType
				layer.ValueBytes = value.Bytes
			}
			if headCount > 0 {
				layer.Heads = make([]HeadSnapshot, headCount)
				for headIdx := range layer.Heads {
					if snapshot.Version >= 3 {
						key := reader.encodedTensor(opts)
						value := reader.encodedTensor(opts)
						layer.Heads[headIdx].Key = key.Values
						layer.Heads[headIdx].KeyDType = key.DType
						layer.Heads[headIdx].KeyBytes = key.Bytes
						layer.Heads[headIdx].Value = value.Values
						layer.Heads[headIdx].ValueDType = value.DType
						layer.Heads[headIdx].ValueBytes = value.Bytes
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
			chunk := reader.read(shapeCount * 4)
			if chunk != nil {
				snapshot.LogitShape = make([]int32, shapeCount)
				for i := range snapshot.LogitShape {
					snapshot.LogitShape[i] = int32(binary.LittleEndian.Uint32(chunk[i*4:]))
				}
			}
		}
		snapshot.Logits = reader.f32s()
	}
	if reader.err != nil {
		return nil, core.E("Load", "parse snapshot", reader.err)
	}
	if snapshot.TokenOffset == 0 {
		snapshot.TokenOffset = len(snapshot.Tokens)
	}
	return snapshot, nil
}

func parseKVSnapshotTokens(data []byte) ([]int32, error) {
	reader := kvSnapshotReader{data: data}
	if magic := string(reader.read(len(kvSnapshotMagic))); magic != kvSnapshotMagic {
		return nil, core.E("Load", "invalid KV snapshot magic", nil)
	}
	version := int(reader.u32())
	if version <= 0 || version > SnapshotVersion {
		return nil, core.E("Load", "unsupported KV snapshot version", nil)
	}
	architectureLength := int(reader.u32())
	reader.read(architectureLength)
	for range 5 {
		reader.u32()
	}
	if version >= 2 {
		reader.u32()
	}
	tokenCount := int(reader.u32())
	if tokenCount < 0 || tokenCount > (len(reader.data)-reader.offset)/4 {
		return nil, core.NewError("mlx: State token block token count is invalid")
	}
	tokens := make([]int32, tokenCount)
	if tokenCount > 0 {
		// Batch the token block read so bounds check is paid once
		// regardless of token count.
		chunk := reader.read(tokenCount * 4)
		if chunk != nil {
			for i := range tokens {
				tokens[i] = int32(binary.LittleEndian.Uint32(chunk[i*4:]))
			}
		}
	}
	if reader.err != nil {
		return nil, core.E("Load", "parse State tokens", reader.err)
	}
	return tokens, nil
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

func appendKVI32s(dst []byte, values []int32) []byte {
	dst = appendKVU32(dst, uint32(len(values)))
	for _, value := range values {
		dst = appendKVI32(dst, value)
	}
	return dst
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

func appendKVEncodedTensor(dst []byte, values []float32, dtype string, raw []byte, encoding Encoding) ([]byte, error) {
	if encoding == EncodingNative {
		if raw, dtype, elements, ok, err := normalizeKVSnapshotNativeTensor(values, dtype, raw); err != nil {
			return nil, err
		} else if ok {
			dst = appendKVU32(dst, 2)
			dst = appendKVU32(dst, uint32(elements))
			dst = appendKVBytes(dst, core.AsBytes(dtype))
			return appendKVBytes(dst, raw), nil
		}
	}
	if len(values) == 0 && len(raw) > 0 {
		return nil, core.NewError("mlx: KV snapshot raw tensor requires native encoding")
	}
	if encoding == EncodingQ8 && kvSnapshotCanQuantizeQ8(values) {
		scale, quantized := quantizeKVSnapshotQ8(values)
		dst = appendKVU32(dst, 1)
		dst = appendKVU32(dst, uint32(len(values)))
		dst = appendKVU32(dst, math.Float32bits(scale))
		return append(dst, quantized...), nil
	}
	dst = appendKVU32(dst, 0)
	dst = appendKVU32(dst, uint32(len(values)))
	return appendKVF32Raw(dst, values), nil
}

func appendKVEncodedF32s(dst []byte, values []float32, encoding Encoding) []byte {
	out, err := appendKVEncodedTensor(dst, values, "", nil, encoding)
	if err != nil {
		return dst
	}
	return out
}

func kvSnapshotEncodedTensorSize(values []float32, dtype string, raw []byte, encoding Encoding) (int, error) {
	if encoding == EncodingNative {
		normalisedDType, _, rawBytes, ok, err := kvSnapshotNativeTensorInfo(values, dtype, raw)
		if err != nil {
			return 0, err
		}
		if ok {
			return 16 + len(normalisedDType) + rawBytes, nil
		}
	}
	if len(values) == 0 && len(raw) > 0 {
		return 0, core.NewError("mlx: KV snapshot raw tensor requires native encoding")
	}
	if encoding == EncodingQ8 && kvSnapshotCanQuantizeQ8(values) {
		return 12 + len(values), nil
	}
	return 8 + len(values)*4, nil
}

func normalizeKVSnapshotNativeTensor(values []float32, dtype string, raw []byte) ([]byte, string, int, bool, error) {
	dtype, elements, rawBytes, ok, err := kvSnapshotNativeTensorInfo(values, dtype, raw)
	if err != nil {
		return nil, "", 0, false, err
	}
	if len(raw) > 0 {
		return raw, dtype, elements, true, nil
	}
	if !ok {
		return nil, "", 0, false, nil
	}
	raw = make([]byte, 0, rawBytes)
	for _, value := range values {
		var buf [4]byte
		binary.LittleEndian.PutUint32(buf[:], math.Float32bits(value))
		raw = append(raw, buf[:]...)
	}
	return raw, "float32", len(values), true, nil
}

func kvSnapshotNativeTensorInfo(values []float32, dtype string, raw []byte) (string, int, int, bool, error) {
	if len(raw) > 0 {
		dtype, bytesPerValue := normalizeKVSnapshotTensorDType(dtype)
		if dtype == "" || bytesPerValue <= 0 {
			return "", 0, 0, false, core.NewError("mlx: unsupported KV snapshot native tensor dtype")
		}
		if len(raw)%bytesPerValue != 0 {
			return "", 0, 0, false, core.NewError("mlx: KV native tensor byte length mismatch")
		}
		elements := len(raw) / bytesPerValue
		if len(values) > 0 && elements != len(values) {
			return "", 0, 0, false, core.NewError("mlx: KV native tensor element count mismatch")
		}
		return dtype, elements, len(raw), true, nil
	}
	if len(values) == 0 {
		return "", 0, 0, false, nil
	}
	return "float32", len(values), len(values) * 4, true, nil
}

func normalizeKVSnapshotTensorDType(dtype string) (string, int) {
	switch dtype {
	case "float32", "F32":
		return "float32", 4
	case "float16", "F16":
		return "float16", 2
	case "bfloat16", "BF16":
		return "bfloat16", 2
	default:
		return "", 0
	}
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

type kvSnapshotStreamWriter struct {
	writer stdio.Writer
	err    error
	buf    [4]byte
}

func (w *kvSnapshotStreamWriter) bytes(data []byte) {
	if w.err != nil {
		return
	}
	n, err := w.writer.Write(data)
	if err != nil {
		w.err = err
		return
	}
	if n != len(data) {
		w.err = stdio.ErrShortWrite
	}
}

func (w *kvSnapshotStreamWriter) bytesWithLength(data []byte) {
	w.u32(uint32(len(data)))
	w.bytes(data)
}

func (w *kvSnapshotStreamWriter) u32(value uint32) {
	binary.LittleEndian.PutUint32(w.buf[:], value)
	w.bytes(w.buf[:])
}

func (w *kvSnapshotStreamWriter) i32(value int32) {
	w.u32(uint32(value))
}

func (w *kvSnapshotStreamWriter) i32s(values []int32) {
	w.u32(uint32(len(values)))
	for _, value := range values {
		w.i32(value)
	}
}

func (w *kvSnapshotStreamWriter) f32s(values []float32) {
	w.u32(uint32(len(values)))
	for _, value := range values {
		w.u32(math.Float32bits(value))
	}
}

func (w *kvSnapshotStreamWriter) encodedTensor(values []float32, dtype string, raw []byte, encoding Encoding) error {
	if encoding == EncodingNative {
		if raw, dtype, elements, ok, err := normalizeKVSnapshotNativeTensor(values, dtype, raw); err != nil {
			return err
		} else if ok {
			w.u32(2)
			w.u32(uint32(elements))
			w.bytesWithLength(core.AsBytes(dtype))
			w.bytesWithLength(raw)
			return w.err
		}
	}
	if len(values) == 0 && len(raw) > 0 {
		return core.NewError("mlx: KV snapshot raw tensor requires native encoding")
	}
	if encoding == EncodingQ8 && kvSnapshotCanQuantizeQ8(values) {
		scale, quantized := quantizeKVSnapshotQ8(values)
		w.u32(1)
		w.u32(uint32(len(values)))
		w.u32(math.Float32bits(scale))
		w.bytes(quantized)
		return w.err
	}
	w.u32(0)
	w.u32(uint32(len(values)))
	for _, value := range values {
		w.u32(math.Float32bits(value))
	}
	return w.err
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

func (r *kvSnapshotReader) i32s() []int32 {
	size := int(r.u32())
	if size <= 0 {
		return nil
	}
	// Single bounds check + direct decode amortises the per-element
	// read+slice overhead the per-call r.u32() loop incurred.
	chunk := r.read(size * 4)
	if chunk == nil {
		return nil
	}
	values := make([]int32, size)
	for i := range values {
		values[i] = int32(binary.LittleEndian.Uint32(chunk[i*4:]))
	}
	return values
}

func (r *kvSnapshotReader) bytes() []byte {
	size := int(r.u32())
	raw := r.read(size)
	if raw == nil {
		return nil
	}
	return raw
}

func (r *kvSnapshotReader) f32s() []float32 {
	size := int(r.u32())
	if size <= 0 {
		return nil
	}
	// Single bounds check + direct decode amortises the per-element
	// read+slice overhead the per-call r.u32() loop incurred.
	chunk := r.read(size * 4)
	if chunk == nil {
		return nil
	}
	values := make([]float32, size)
	for i := range values {
		values[i] = math.Float32frombits(binary.LittleEndian.Uint32(chunk[i*4:]))
	}
	return values
}

type kvSnapshotEncodedTensor struct {
	Values []float32
	DType  string
	Bytes  []byte
}

func (r *kvSnapshotReader) encodedF32s() []float32 {
	return r.encodedTensor(LoadOptions{}).Values
}

func (r *kvSnapshotReader) encodedTensor(opts LoadOptions) kvSnapshotEncodedTensor {
	encoding := r.u32()
	size := int(r.u32())
	switch encoding {
	case 0:
		if size <= 0 {
			return kvSnapshotEncodedTensor{Values: []float32{}}
		}
		// Single bounds check via batched read avoids per-element bounds work.
		chunk := r.read(size * 4)
		if chunk == nil {
			return kvSnapshotEncodedTensor{}
		}
		values := make([]float32, size)
		for i := range values {
			values[i] = math.Float32frombits(binary.LittleEndian.Uint32(chunk[i*4:]))
		}
		return kvSnapshotEncodedTensor{Values: values}
	case 1:
		scale := math.Float32frombits(r.u32())
		raw := r.read(size)
		values := make([]float32, size)
		for i, value := range raw {
			values[i] = float32(int8(value)) * scale
		}
		return kvSnapshotEncodedTensor{Values: values}
	case 2:
		dtype := r.string()
		raw := r.bytes()
		dtype, err := validateKVSnapshotNativeTensor(dtype, raw, size)
		if err != nil {
			r.err = err
			return kvSnapshotEncodedTensor{}
		}
		if opts.RawKVOnly {
			return kvSnapshotEncodedTensor{
				DType: dtype,
				Bytes: raw,
			}
		}
		values, err := decodeKVSnapshotNativeTensor(dtype, raw, size)
		if err != nil {
			r.err = err
			return kvSnapshotEncodedTensor{}
		}
		return kvSnapshotEncodedTensor{
			Values: values,
			DType:  dtype,
			Bytes:  raw,
		}
	default:
		r.err = core.NewError("mlx: unsupported KV tensor encoding")
		return kvSnapshotEncodedTensor{}
	}
}

func validateKVSnapshotNativeTensor(dtype string, raw []byte, elements int) (string, error) {
	dtype, bytesPerValue := normalizeKVSnapshotTensorDType(dtype)
	if dtype == "" || bytesPerValue <= 0 {
		return "", core.NewError("mlx: unsupported KV native tensor dtype")
	}
	if elements < 0 || len(raw) != elements*bytesPerValue {
		return "", core.NewError("mlx: KV native tensor byte length mismatch")
	}
	return dtype, nil
}

func decodeKVSnapshotNativeTensor(dtype string, raw []byte, elements int) ([]float32, error) {
	dtype, err := validateKVSnapshotNativeTensor(dtype, raw, elements)
	if err != nil {
		return nil, err
	}
	values := make([]float32, elements)
	switch dtype {
	case "float32":
		for i := range values {
			values[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
	case "float16":
		for i := range values {
			values[i] = safetensors.Float16ToFloat32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
	case "bfloat16":
		for i := range values {
			values[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[i*2:])) << 16)
		}
	default:
		return nil, core.NewError("mlx: unsupported KV native tensor dtype")
	}
	return values, nil
}

func cloneKVLayers(src []LayerSnapshot) []LayerSnapshot {
	if len(src) == 0 {
		return nil
	}
	cloned := make([]LayerSnapshot, len(src))
	for i, layer := range src {
		cloned[i] = LayerSnapshot{
			Layer:      layer.Layer,
			CacheIndex: layer.CacheIndex,
			KeyDType:   layer.KeyDType,
			KeyBytes:   append([]byte(nil), layer.KeyBytes...),
			KeyShape:   append([]int32(nil), layer.KeyShape...),
			ValueDType: layer.ValueDType,
			ValueBytes: append([]byte(nil), layer.ValueBytes...),
			ValueShape: append([]int32(nil), layer.ValueShape...),
			Heads:      cloneKVHeads(layer.Heads),
		}
	}
	return cloned
}

func cloneKVHeads(src []HeadSnapshot) []HeadSnapshot {
	if len(src) == 0 {
		return nil
	}
	cloned := make([]HeadSnapshot, len(src))
	for i, head := range src {
		cloned[i] = cloneKVHead(head)
	}
	return cloned
}

func cloneKVHead(src HeadSnapshot) HeadSnapshot {
	return HeadSnapshot{
		Key:        append([]float32(nil), src.Key...),
		KeyDType:   src.KeyDType,
		KeyBytes:   append([]byte(nil), src.KeyBytes...),
		Value:      append([]float32(nil), src.Value...),
		ValueDType: src.ValueDType,
		ValueBytes: append([]byte(nil), src.ValueBytes...),
	}
}

func DropFloat32(snapshot *Snapshot) {
	if snapshot == nil {
		return
	}
	for layerIndex := range snapshot.Layers {
		for headIndex := range snapshot.Layers[layerIndex].Heads {
			head := &snapshot.Layers[layerIndex].Heads[headIndex]
			if len(head.KeyBytes) > 0 {
				head.Key = nil
			}
			if len(head.ValueBytes) > 0 {
				head.Value = nil
			}
		}
	}
}

func ResultError(result core.Result) error {
	if err, ok := result.Value.(error); ok {
		return err
	}
	if text, ok := result.Value.(string); ok {
		return core.NewError(text)
	}
	return core.NewError("unknown filesystem error")
}

const defaultCacheBlockSize = 512

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if core.Trim(value) != "" {
			return value
		}
	}
	return ""
}

func normalizeSnapshot(snapshot *Snapshot) {
	if snapshot == nil {
		return
	}
	if snapshot.Version == 0 {
		snapshot.Version = SnapshotVersion
	}
	if snapshot.TokenOffset == 0 {
		snapshot.TokenOffset = len(snapshot.Tokens)
	}
}

func requiresNativeEncoding(snapshot *Snapshot) bool {
	if snapshot == nil {
		return false
	}
	if snapshotHasLayerNativeTensors(snapshot) {
		return true
	}
	for _, layer := range snapshot.Layers {
		for _, head := range layer.Heads {
			if len(head.Key) == 0 && len(head.KeyBytes) > 0 {
				return true
			}
			if len(head.Value) == 0 && len(head.ValueBytes) > 0 {
				return true
			}
		}
	}
	return false
}

func snapshotHasLayerNativeTensors(snapshot *Snapshot) bool {
	if snapshot == nil {
		return false
	}
	for _, layer := range snapshot.Layers {
		if len(layer.KeyBytes) > 0 || len(layer.ValueBytes) > 0 {
			return true
		}
	}
	return false
}

// HashSnapshot computes a stable hash of a normalised Snapshot for use as
// a content-addressed identifier.
//
//	hash, err := kv.HashSnapshot(snap)
func HashSnapshot(snapshot *Snapshot) (string, error) {
	if snapshot == nil {
		return "", core.NewError("mlx: KV snapshot is nil")
	}
	// bytesWithOptions is read-only — version derivation and token-offset
	// defaulting are applied inline during encoding, so the defensive clone
	// + normalize round-trip is pure waste. Direct encode is hash-stable
	// because the writer always emits len(Tokens) when TokenOffset is zero.
	opts := SaveOptions{}
	if requiresNativeEncoding(snapshot) {
		opts.KVEncoding = EncodingNative
	}
	data, err := snapshot.bytesWithOptions(opts)
	if err != nil {
		return "", err
	}
	return core.SHA256Hex(data), nil
}
