// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the model-merge pure-math + plan-construction core.
// Per AX-11 — Packs is a slow IO-bound action overall, but the inner
// kernels (linearMerge, slerpMerge, writeFloat32Values, mergeTensorValues,
// normalizedWeights, buildMergedHeader, sameUint64Slice, clampFloat64)
// run per-tensor per-chunk and are the surface a budget pass can
// actually optimise. The chunked write paths (writeLinearChunks,
// writeSLERPChunks) are exercised at small sizes to surface the
// per-chunk overhead without making the bench IO-dominated.
//
// Run:    go test -bench='BenchmarkMerge|BenchmarkLinearMerge|BenchmarkSLERPMerge|BenchmarkMergeTensorValues|BenchmarkNormalizedWeights|BenchmarkBuildMergedHeader|BenchmarkSameUint64Slice|BenchmarkClampFloat64|BenchmarkWriteFloat32Values|BenchmarkWriteLinearChunks|BenchmarkWriteSLERPChunks' -benchmem -run='^$' ./go/merge

package merge

import (
	"context"
	"encoding/binary"
	"math"
	"testing"
	"unsafe"

	core "dappco.re/go"
	mp "dappco.re/go/mlx/pack"
	"dappco.re/go/mlx/safetensors"
)

// Sinks defeat compiler DCE.
var (
	benchMergeF32    []float32
	benchMergeF64    []float64
	benchMergeErr    error
	benchMergeBool   bool
	benchMergeFloat  float64
	benchMergeBytes  []byte
	benchMergeHeader map[string]safetensors.HeaderEntry
	benchMergeResult *Result
)

// benchTensorValues builds two synthetic source slices the merge math
// can chew on. Same shape, same length — every value finite.
func benchTensorValues(n int) [][]float32 {
	left := make([]float32, n)
	right := make([]float32, n)
	for i := range left {
		left[i] = float32(i%256) * 0.0125
		right[i] = float32((i+1)%256) * 0.0125
	}
	return [][]float32{left, right}
}

// --- linearMerge — per-tensor inner loop ---

func BenchmarkLinearMerge_1024Elements(b *testing.B) {
	values := benchTensorValues(1024)
	weights := []float64{0.25, 0.75}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchMergeF32, benchMergeErr = linearMerge(values, weights)
	}
}

func BenchmarkLinearMerge_1048576Elements(b *testing.B) {
	values := benchTensorValues(1 << 20) // 1 MiB float32 elements per source
	weights := []float64{0.5, 0.5}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchMergeF32, benchMergeErr = linearMerge(values, weights)
	}
}

// --- slerpMerge — adds dot/norm scan over both tensors before linear ---

func BenchmarkSLERPMerge_1024Elements(b *testing.B) {
	values := benchTensorValues(1024)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchMergeF32, benchMergeErr = slerpMerge(values, 0.5)
	}
}

func BenchmarkSLERPMerge_1048576Elements(b *testing.B) {
	values := benchTensorValues(1 << 20)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchMergeF32, benchMergeErr = slerpMerge(values, 0.5)
	}
}

// SLERP falls back to linearMerge when norms are zero — bench the
// degenerate path separately.
func BenchmarkSLERPMerge_ZeroNormFallback(b *testing.B) {
	zero := make([]float32, 1024)
	right := make([]float32, 1024)
	for i := range right {
		right[i] = float32(i)
	}
	values := [][]float32{zero, right}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchMergeF32, benchMergeErr = slerpMerge(values, 0.5)
	}
}

// --- mergeTensorValues — public dispatcher fires per tensor ---

func BenchmarkMergeTensorValues_Linear(b *testing.B) {
	values := benchTensorValues(1024)
	weights := []float64{0.25, 0.75}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchMergeF32, benchMergeErr = mergeTensorValues(values, MethodLinear, 0, weights)
	}
}

func BenchmarkMergeTensorValues_SLERP(b *testing.B) {
	values := benchTensorValues(1024)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchMergeF32, benchMergeErr = mergeTensorValues(values, MethodSLERP, 0.5, nil)
	}
}

// --- normalizedWeights — fires once per merge but is on the prepare
// path; cheap so easy to spot regressions. ---

func BenchmarkNormalizedWeights_EqualSplit(b *testing.B) {
	sources := []Source{{}, {}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchMergeF64, benchMergeErr = normalizedWeights(sources)
	}
}

func BenchmarkNormalizedWeights_Explicit3(b *testing.B) {
	sources := []Source{{Weight: 0.25}, {Weight: 0.5}, {Weight: 0.25}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchMergeF64, benchMergeErr = normalizedWeights(sources)
	}
}

// --- buildMergedHeader — runs once per merge but scales with tensor
// count. Real qwen3-class checkpoints have 200+ tensor entries. ---

func benchSafetensorsIndex(tensorCount int) safetensors.Index {
	names := make([]string, 0, tensorCount)
	tensors := make(map[string]safetensors.TensorRef, tensorCount)
	var offset int64
	for i := range tensorCount {
		name := "blk." + core.Itoa(i/4) + ".w." + core.Itoa(i%4)
		shape := []uint64{4096, 4096}
		elements := 4096 * 4096
		byteLen := int64(elements * 4)
		tensors[name] = safetensors.TensorRef{
			Name:      name,
			DType:     "F32",
			Shape:     shape,
			Elements:  elements,
			DataStart: offset,
			ByteLen:   byteLen,
		}
		offset += byteLen
		names = append(names, name)
	}
	return safetensors.Index{Names: names, Tensors: tensors}
}

func BenchmarkBuildMergedHeader_50Tensors(b *testing.B) {
	index := benchSafetensorsIndex(50)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchMergeHeader = buildMergedHeader(index)
	}
}

func BenchmarkBuildMergedHeader_200Tensors(b *testing.B) {
	index := benchSafetensorsIndex(200)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchMergeHeader = buildMergedHeader(index)
	}
}

// --- sameUint64Slice — per-tensor shape match; runs in validateTensorIndexes
// + readTensorRefs + readTensorValues ---

func BenchmarkSameUint64Slice_Match4D(b *testing.B) {
	a := []uint64{4, 28, 2048, 64}
	c := []uint64{4, 28, 2048, 64}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchMergeBool = sameUint64Slice(a, c)
	}
}

func BenchmarkSameUint64Slice_DifferentLength(b *testing.B) {
	a := []uint64{4, 28, 2048, 64}
	c := []uint64{4, 28, 2048}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchMergeBool = sameUint64Slice(a, c)
	}
}

func BenchmarkSameUint64Slice_LastDimMismatch(b *testing.B) {
	a := []uint64{4, 28, 2048, 64}
	c := []uint64{4, 28, 2048, 128}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchMergeBool = sameUint64Slice(a, c)
	}
}

// --- clampFloat64 — fires per SLERP angle clamp ---

func BenchmarkClampFloat64_InRange(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchMergeFloat = clampFloat64(0.5, -1, 1)
	}
}

func BenchmarkClampFloat64_ClampHigh(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchMergeFloat = clampFloat64(2.0, -1, 1)
	}
}

// --- writeFloat32Values — fires per chunk write; the encode loop ---

type discardFile struct {
	written int
}

// noopWriter is a minimal *core.OSFile substitute path: the merge
// writers expect *core.OSFile so we run a small temp file to keep the
// signature satisfied without touching disk for huge slices.

func BenchmarkWriteFloat32Values_1024(b *testing.B) {
	dir := b.TempDir()
	created := core.Create(core.PathJoin(dir, "out.bin"))
	if !created.OK {
		b.Fatal(created.Error())
	}
	file := created.Value.(*core.OSFile)
	defer file.Close()

	values := make([]float32, 1024)
	for i := range values {
		values[i] = float32(i)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchMergeErr = writeFloat32Values(file, values)
	}
}

func BenchmarkWriteFloat32Values_98304(b *testing.B) {
	dir := b.TempDir()
	created := core.Create(core.PathJoin(dir, "out.bin"))
	if !created.OK {
		b.Fatal(created.Error())
	}
	file := created.Value.(*core.OSFile)
	defer file.Close()

	values := make([]float32, 98304)
	for i := range values {
		values[i] = float32(i % 1024)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchMergeErr = writeFloat32Values(file, values)
	}
}

// BenchmarkWriteFloat32ValuesScratch_1M sizes the slice large enough
// that the float-serialisation loop dominates over alloc + the file
// write syscall. Exposes the unsafe reinterpret-cast win on the merge
// writer's hot path — single memcpy vs per-element PutUint32 +
// Float32bits. Reuses scratch across iterations so allocation cost is
// paid once (mirrors the chunked-merge IO callers).
func BenchmarkWriteFloat32ValuesScratch_1M(b *testing.B) {
	dir := b.TempDir()
	created := core.Create(core.PathJoin(dir, "out.bin"))
	if !created.OK {
		b.Fatal(created.Error())
	}
	file := created.Value.(*core.OSFile)
	defer file.Close()

	values := make([]float32, 1<<20)
	for i := range values {
		values[i] = float32(i % 1024)
	}
	scratch := make([]byte, len(values)*4)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		scratch, benchMergeErr = writeFloat32ValuesScratch(file, values, scratch)
	}
}

// BenchmarkWriteFloat32ValuesEncode_1M_* measures only the float→byte
// serialisation step — the inner kernel writeFloat32ValuesScratch
// exists to replace. Isolates the unsafe reinterpret-cast win from
// the file.Write syscall floor. The LoopForm variant is the legacy
// per-element binary.LittleEndian.PutUint32(buf, math.Float32bits(v))
// path; the UnsafeForm variant is what ships in writeFloat32Values
// Scratch. Direct apples-to-apples — same fixture, same scratch
// reuse. Mirrors the comparison W8-A2 made in go/kv/snapshot.go
// f32sRaw.
func BenchmarkWriteFloat32ValuesEncode_1M_LoopForm(b *testing.B) {
	values := make([]float32, 1<<20)
	for i := range values {
		values[i] = float32(i % 1024)
	}
	scratch := make([]byte, len(values)*4)
	b.ReportAllocs()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		for i, value := range values {
			binary.LittleEndian.PutUint32(scratch[i*4:], math.Float32bits(value))
		}
	}
	benchMergeBytes = scratch
}

func BenchmarkWriteFloat32ValuesEncode_1M_UnsafeForm(b *testing.B) {
	values := make([]float32, 1<<20)
	for i := range values {
		values[i] = float32(i % 1024)
	}
	scratch := make([]byte, len(values)*4)
	b.ReportAllocs()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		src := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(values))), len(values)*4)
		copy(scratch, src)
	}
	benchMergeBytes = scratch
}

// --- writeLinearChunks — chunked merge IO path ---

// benchWriteSafetensorsF32 lays down a small safetensors file in temp
// so the chunk readers have something to seek over. Mirrors
// writeTestSafetensorsF32 in helpers_test.go but takes *testing.B.
func benchWriteSafetensorsF32(b *testing.B, path string, name string, shape []int, values []float32) {
	b.Helper()
	type entry struct {
		DType       string `json:"dtype"`
		Shape       []int  `json:"shape"`
		DataOffsets []int  `json:"data_offsets"`
	}
	header := map[string]entry{
		name: {DType: "F32", Shape: shape, DataOffsets: []int{0, len(values) * 4}},
	}
	encoded := core.JSONMarshal(header)
	if !encoded.OK {
		b.Fatalf("marshal safetensors header: %v", encoded.Value)
	}
	headerBytes := encoded.Value.([]byte)
	out := make([]byte, 8+len(headerBytes)+len(values)*4)
	// little-endian uint64 header length
	for i := range 8 {
		out[i] = byte(uint64(len(headerBytes)) >> (8 * i))
	}
	copy(out[8:], headerBytes)
	body := out[8+len(headerBytes):]
	for i, v := range values {
		bits := math.Float32bits(v)
		body[i*4+0] = byte(bits)
		body[i*4+1] = byte(bits >> 8)
		body[i*4+2] = byte(bits >> 16)
		body[i*4+3] = byte(bits >> 24)
	}
	if result := core.WriteFile(path, out, 0o644); !result.OK {
		b.Fatalf("write safetensors: %v", result.Value)
	}
}

func BenchmarkWriteLinearChunks_4KChunks(b *testing.B) {
	dir := b.TempDir()
	name := "blk.0.w.0"
	leftValues := make([]float32, 4096)
	rightValues := make([]float32, 4096)
	for i := range leftValues {
		leftValues[i] = float32(i)
		rightValues[i] = float32(i) * 0.5
	}
	leftPath := core.PathJoin(dir, "left.safetensors")
	rightPath := core.PathJoin(dir, "right.safetensors")
	benchWriteSafetensorsF32(b, leftPath, name, []int{4096}, leftValues)
	benchWriteSafetensorsF32(b, rightPath, name, []int{4096}, rightValues)
	leftIndex, err := safetensors.IndexFiles([]string{leftPath})
	if err != nil {
		b.Fatal(err)
	}
	rightIndex, err := safetensors.IndexFiles([]string{rightPath})
	if err != nil {
		b.Fatal(err)
	}
	refs := []safetensors.TensorRef{leftIndex.Tensors[name], rightIndex.Tensors[name]}
	weights := []float64{0.25, 0.75}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		outPath := core.PathJoin(dir, "out.bin")
		created := core.Create(outPath)
		if !created.OK {
			b.Fatal(created.Error())
		}
		file := created.Value.(*core.OSFile)
		b.StartTimer()
		benchMergeErr = writeLinearChunks(context.Background(), file, refs, weights, 1024)
		b.StopTimer()
		_ = file.Close()
		b.StartTimer()
	}
}

func BenchmarkWriteSLERPChunks_4KChunks(b *testing.B) {
	dir := b.TempDir()
	name := "blk.0.w.0"
	leftValues := make([]float32, 4096)
	rightValues := make([]float32, 4096)
	for i := range leftValues {
		// Set up a non-degenerate angle so the dot/norm path runs to the
		// real SLERP formula, not the cosTheta>0.9995 shortcut.
		leftValues[i] = float32(math.Sin(float64(i) * 0.01))
		rightValues[i] = float32(math.Cos(float64(i) * 0.01))
	}
	leftPath := core.PathJoin(dir, "left.safetensors")
	rightPath := core.PathJoin(dir, "right.safetensors")
	benchWriteSafetensorsF32(b, leftPath, name, []int{4096}, leftValues)
	benchWriteSafetensorsF32(b, rightPath, name, []int{4096}, rightValues)
	leftIndex, err := safetensors.IndexFiles([]string{leftPath})
	if err != nil {
		b.Fatal(err)
	}
	rightIndex, err := safetensors.IndexFiles([]string{rightPath})
	if err != nil {
		b.Fatal(err)
	}
	refs := []safetensors.TensorRef{leftIndex.Tensors[name], rightIndex.Tensors[name]}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		outPath := core.PathJoin(dir, "out.bin")
		created := core.Create(outPath)
		if !created.OK {
			b.Fatal(created.Error())
		}
		file := created.Value.(*core.OSFile)
		b.StartTimer()
		benchMergeErr = writeSLERPChunks(context.Background(), file, refs, 0.5, 1024)
		b.StopTimer()
		_ = file.Close()
		b.StartTimer()
	}
}

// --- validateTensorIndexes — runs once per merge across all source
// indexes. The base-vs-source name scan is the inner loop. ---

func BenchmarkValidateTensorIndexes_AllMatch(b *testing.B) {
	left := benchSafetensorsIndex(200)
	right := benchSafetensorsIndex(200)
	indexes := []safetensors.Index{left, right}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchMergeErr = validateTensorIndexes(indexes, false)
	}
}

// --- Packs end-to-end — the per-tensor write loop in
// writeMergedSafetensors. A real pack is many tensors in a few shard
// files; the loop opens the source shards once per tensor via
// OpenReaders, so the file-open allocs scale with tensor count even
// though only a handful of distinct shards exist. This bench writes a
// multi-tensor single-shard pack per source so that per-tensor reopen
// cost is the visible alloc surface (mirrors the fileCache win already
// in ComparePacks). ---

// benchWritePackTensors lays down one safetensors shard holding many
// small tensors and returns a pack pointed at it. Mirrors
// benchCompareScratchPack but lives on the merge side and keeps the
// per-tensor element count small so the bench stays write-loop-bound,
// not IO-bound.
func benchWritePackTensors(b *testing.B, names []string, perTensorElements int) mp.ModelPack {
	b.Helper()
	dir := b.TempDir()
	cfg := `{"model_type":"qwen3","vocab_size":151936,"hidden_size":2048,"num_hidden_layers":28,"max_position_embeddings":40960}`
	if result := core.WriteFile(core.PathJoin(dir, "config.json"), []byte(cfg), 0o644); !result.OK {
		b.Fatalf("write config: %v", result.Value)
	}
	tok := `{"model":{"type":"BPE","vocab":{"a":0},"merges":[]}}`
	if result := core.WriteFile(core.PathJoin(dir, "tokenizer.json"), []byte(tok), 0o644); !result.OK {
		b.Fatalf("write tokenizer: %v", result.Value)
	}
	values := make([]float32, perTensorElements)
	for i := range values {
		values[i] = float32(i%128) * 0.01
	}
	type entry struct {
		DType       string `json:"dtype"`
		Shape       []int  `json:"shape"`
		DataOffsets []int  `json:"data_offsets"`
	}
	header := map[string]entry{}
	var body []byte
	for _, name := range names {
		start := len(body)
		buf := make([]byte, perTensorElements*4)
		for i, v := range values {
			bits := math.Float32bits(v)
			buf[i*4+0] = byte(bits)
			buf[i*4+1] = byte(bits >> 8)
			buf[i*4+2] = byte(bits >> 16)
			buf[i*4+3] = byte(bits >> 24)
		}
		body = append(body, buf...)
		header[name] = entry{DType: "F32", Shape: []int{perTensorElements}, DataOffsets: []int{start, len(body)}}
	}
	encoded := core.JSONMarshal(header)
	if !encoded.OK {
		b.Fatalf("marshal header: %v", encoded.Value)
	}
	headerBytes := encoded.Value.([]byte)
	out := make([]byte, 8+len(headerBytes)+len(body))
	hl := uint64(len(headerBytes))
	for i := range 8 {
		out[i] = byte(hl >> (8 * i))
	}
	copy(out[8:], headerBytes)
	copy(out[8+len(headerBytes):], body)
	tensorPath := core.PathJoin(dir, "model.safetensors")
	if result := core.WriteFile(tensorPath, out, 0o644); !result.OK {
		b.Fatalf("write safetensors: %v", result.Value)
	}
	return mp.ModelPack{
		Root:          dir,
		Path:          dir,
		Format:        mp.ModelPackFormatSafetensors,
		WeightFiles:   []string{tensorPath},
		TokenizerPath: core.PathJoin(dir, "tokenizer.json"),
		Architecture:  "qwen3",
	}
}

func benchMergePackNames(tensorCount int) []string {
	names := make([]string, 0, tensorCount)
	for i := range tensorCount {
		names = append(names, "model.layers."+core.Itoa(i/4)+".w."+core.Itoa(i%4)+".weight")
	}
	return names
}

func BenchmarkPacksLinear_32Tensors(b *testing.B) {
	names := benchMergePackNames(32)
	left := benchWritePackTensors(b, names, 256)
	right := benchWritePackTensors(b, names, 256)
	outDir := b.TempDir()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out := core.PathJoin(outDir, "merged-"+core.Itoa(i))
		benchMergeResult, benchMergeErr = Packs(context.Background(), Options{
			OutputPath: out,
			Method:     MethodLinear,
			Sources: []Source{
				{Pack: left, Weight: 0.5},
				{Pack: right, Weight: 0.5},
			},
		})
	}
}

func BenchmarkPacksSLERP_32Tensors(b *testing.B) {
	names := benchMergePackNames(32)
	left := benchWritePackTensors(b, names, 256)
	right := benchWritePackTensors(b, names, 256)
	outDir := b.TempDir()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out := core.PathJoin(outDir, "merged-"+core.Itoa(i))
		benchMergeResult, benchMergeErr = Packs(context.Background(), Options{
			OutputPath: out,
			Method:     MethodSLERP,
			T:          0.5,
			Sources: []Source{
				{Pack: left},
				{Pack: right},
			},
		})
	}
}
