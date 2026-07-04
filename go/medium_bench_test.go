// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for medium.go — the io.Medium staging surface.
// Per AX-11 — stagePathFromMedium fires once per LoadModelFromMedium
// call (model load, hundreds-of-MB streams), so the per-tree pass is
// the cost. walkMedium recurses N times for an N-entry tree; the per-
// entry cost (PathJoin + mediumRelativePath + PathJoin) is the
// dominant alloc shape.
//
// mediumModelRoot / cleanMediumPath fire on the cold open-path side
// once per call, but mediumRelativePath fires once per visited entry
// inside the walkMedium recursion — its hot-suffix branch is the
// load-bearing inner loop.
//
// Run:    go test -bench='BenchmarkMedium' -benchmem -run='^$' ./go

package mlx

import (
	"io/fs"
	"testing"

	coreio "dappco.re/go/io"
)

// Sinks defeat compiler DCE.
var (
	mediumBenchSinkString string
	mediumBenchSinkErr    error
)

// --- mediumRelativePath ---
// Hot path: walkMedium feeds visit callback with paths shaped as
// `root + "/" + suffix`. The hot-suffix branch returns the suffix
// directly; bench it on the shape it actually sees.

func BenchmarkMedium_RelativePath_HotSuffix(b *testing.B) {
	root := "models/gemma-3-1b"
	target := "models/gemma-3-1b/model.safetensors"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mediumBenchSinkString = mediumRelativePath(root, target)
	}
}

// Nested suffix — same shape as a model bundle's sub/tokenizer.json
// shape; ensures the hot-suffix branch handles deep relative paths
// without falling through to PathRel.

func BenchmarkMedium_RelativePath_HotSuffixNested(b *testing.B) {
	root := "models/qwen3-7b"
	target := "models/qwen3-7b/sub/tokenizer.json"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mediumBenchSinkString = mediumRelativePath(root, target)
	}
}

// Empty root — falls through TrimPrefix path; bench it for the
// stage-with-implicit-root callers.

func BenchmarkMedium_RelativePath_EmptyRoot(b *testing.B) {
	target := "/models/gemma-3-1b/model.safetensors"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mediumBenchSinkString = mediumRelativePath("", target)
	}
}

// Identical root == target — early-return path.

func BenchmarkMedium_RelativePath_RootEqualsTarget(b *testing.B) {
	root := "models/gemma-3-1b"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mediumBenchSinkString = mediumRelativePath(root, root)
	}
}

// --- cleanMediumPath ---
// Trim + Clean entry — cold-ish (called once per stage), but the
// shape is small + tidy so we want the floor pinned.

func BenchmarkMedium_CleanMediumPath_Clean(b *testing.B) {
	p := "models/gemma-3-1b"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mediumBenchSinkString = cleanMediumPath(p)
	}
}

func BenchmarkMedium_CleanMediumPath_WithWhitespace(b *testing.B) {
	p := "  models/gemma-3-1b/  "
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mediumBenchSinkString = cleanMediumPath(p)
	}
}

// --- mediumModelRoot ---
// Once per stage call; weight-file shape (one HasSuffix hit) vs
// directory shape (fall-through).

func BenchmarkMedium_ModelRoot_SafetensorsFile(b *testing.B) {
	p := "models/gemma-3-1b/model.safetensors"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mediumBenchSinkString = mediumModelRoot(p)
	}
}

func BenchmarkMedium_ModelRoot_Directory(b *testing.B) {
	p := "models/gemma-3-1b"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mediumBenchSinkString = mediumModelRoot(p)
	}
}

// --- fromSlashPath ---
// On POSIX the early-return branch is taken; ensure no surprise alloc.

func BenchmarkMedium_FromSlashPath(b *testing.B) {
	p := "models/gemma-3-1b/sub/tokenizer.json"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mediumBenchSinkString = fromSlashPath(p)
	}
}

// --- walkMedium end-to-end ---
// Stages a small synthetic model tree into a MemoryMedium and walks
// it, counting visited paths. Captures the *per-tree* cost — every
// real LoadModelFromMedium call drives this loop end-to-end.

func benchMediumPopulate(b *testing.B) *coreio.MemoryMedium {
	b.Helper()
	medium := coreio.NewMemoryMedium()
	files := []string{
		"models/demo/config.json",
		"models/demo/tokenizer.json",
		"models/demo/special_tokens_map.json",
		"models/demo/sub/tokenizer.json",
		"models/demo/model.safetensors",
	}
	for _, file := range files {
		if err := medium.Write(file, "x"); err != nil {
			b.Fatalf("populate medium %q: %v", file, err)
		}
	}
	return medium
}

func BenchmarkMedium_WalkMedium_Small(b *testing.B) {
	medium := benchMediumPopulate(b)
	root := "models/demo"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		visitCount := 0
		err := walkMedium(medium, root, func(p string, _ fs.DirEntry) error {
			visitCount++
			_ = p
			return nil
		})
		if err != nil {
			b.Fatalf("walkMedium: %v", err)
		}
		mediumBenchSinkErr = err
	}
}
