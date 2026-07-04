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
	"dappco.re/go/inference/safetensors"
)

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
