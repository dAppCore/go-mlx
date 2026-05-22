// SPDX-Licence-Identifier: EUPL-1.2

//go:build !nomlxlm

// Benchmarks for the hot-path helpers in backend.go that don't require
// a live subprocess: jsonlinereader.ReadLine's indexByte scan, the
// stringSliceOption clone path, and the argv-build path inside
// startMLXLMProcess.
//
// Run: go test -bench='BenchmarkBackend' -benchmem -run='^$' ./go/mlxlm

package mlxlm

import (
	"bytes"
	"testing"

	core "dappco.re/go"
)

var (
	backendBenchSinkInt    int
	backendBenchSinkArgs   []string
	backendBenchSinkArgv   []string
	backendBenchSinkLine   []byte
	backendBenchSinkErr    error
	backendBenchSinkOpts   core.Options
	backendBenchSinkResult []string
)

// --- bytes.IndexByte — called once per ReadLine to locate the JSON
// line boundary. Backend reads 32 KB chunks; scan dominates ReadLine
// cost. bytes.IndexByte uses arm64 SIMD; the prior hand-rolled loop
// did not.

func benchmarkIndexByte(b *testing.B, size int, terminatorAt int) {
	data := make([]byte, size)
	for i := range data {
		data[i] = 'x'
	}
	if terminatorAt >= 0 && terminatorAt < size {
		data[terminatorAt] = '\n'
	}
	b.ReportAllocs()
	b.SetBytes(int64(size))
	for i := 0; i < b.N; i++ {
		backendBenchSinkInt = bytes.IndexByte(data, '\n')
	}
}

func BenchmarkBackend_IndexByte_NotFound_1K(b *testing.B)  { benchmarkIndexByte(b, 1024, -1) }
func BenchmarkBackend_IndexByte_NotFound_32K(b *testing.B) { benchmarkIndexByte(b, 32*1024, -1) }
func BenchmarkBackend_IndexByte_FoundEnd_1K(b *testing.B)  { benchmarkIndexByte(b, 1024, 1023) }
func BenchmarkBackend_IndexByte_FoundEnd_32K(b *testing.B) { benchmarkIndexByte(b, 32*1024, 32*1024-1) }
func BenchmarkBackend_IndexByte_FoundMid_32K(b *testing.B) { benchmarkIndexByte(b, 32*1024, 16*1024) }

// --- stringSliceOption — clones the args slice once per process start.

func BenchmarkBackend_StringSliceOption_Small(b *testing.B) {
	opts := core.NewOptions(core.Option{Key: "args", Value: []string{"-u", "bridge.py"}})
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		backendBenchSinkArgs, backendBenchSinkErr = stringSliceOption(opts, "args")
	}
}

func BenchmarkBackend_StringSliceOption_Large(b *testing.B) {
	args := make([]string, 32)
	for i := range args {
		args[i] = "arg-with-some-length-" + core.Itoa(i)
	}
	opts := core.NewOptions(core.Option{Key: "args", Value: args})
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		backendBenchSinkArgs, backendBenchSinkErr = stringSliceOption(opts, "args")
	}
}

// --- argv build — mirrors `append([]string{command}, args...)` inside
// startMLXLMProcess.

func argvBuildAppend(command string, args []string) []string {
	return append([]string{command}, args...)
}

func argvBuildMake(command string, args []string) []string {
	argv := make([]string, 1+len(args))
	argv[0] = command
	copy(argv[1:], args)
	return argv
}

func BenchmarkBackend_ArgvBuild_AppendLiteral_Small(b *testing.B) {
	args := []string{"-u", "bridge.py"}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		backendBenchSinkArgv = argvBuildAppend("python3", args)
	}
}

func BenchmarkBackend_ArgvBuild_MakeCopy_Small(b *testing.B) {
	args := []string{"-u", "bridge.py"}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		backendBenchSinkArgv = argvBuildMake("python3", args)
	}
}

func BenchmarkBackend_ArgvBuild_AppendLiteral_Large(b *testing.B) {
	args := make([]string, 16)
	for i := range args {
		args[i] = "arg-with-some-length-" + core.Itoa(i)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		backendBenchSinkArgv = argvBuildAppend("python3", args)
	}
}

func BenchmarkBackend_ArgvBuild_MakeCopy_Large(b *testing.B) {
	args := make([]string, 16)
	for i := range args {
		args[i] = "arg-with-some-length-" + core.Itoa(i)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		backendBenchSinkArgv = argvBuildMake("python3", args)
	}
}

// --- ReadLine over a synthetic stdout stream of N short JSON lines.
// Replaces the actual subprocess and isolates the parser cost.

type fixedReader struct {
	data []byte
	pos  int
}

func (r *fixedReader) Read(p []byte) (int, error) {
	if r.pos >= len(r.data) {
		return 0, nil
	}
	n := copy(p, r.data[r.pos:])
	r.pos += n
	return n, nil
}

func makeJSONLines(numLines, lineBytes int) []byte {
	out := make([]byte, 0, numLines*(lineBytes+1))
	line := make([]byte, lineBytes)
	for i := range line {
		line[i] = 'x'
	}
	for i := 0; i < numLines; i++ {
		out = append(out, line...)
		out = append(out, '\n')
	}
	return out
}

func BenchmarkBackend_ReadLine_Short(b *testing.B) {
	const lines = 32
	const lineSize = 64
	data := makeJSONLines(lines, lineSize)
	b.ReportAllocs()
	b.SetBytes(int64(len(data)))
	for i := 0; i < b.N; i++ {
		reader := newJSONLineReader(&fixedReader{data: data})
		for j := 0; j < lines; j++ {
			backendBenchSinkLine, _ = reader.ReadLine()
		}
	}
}

func BenchmarkBackend_ReadLine_Long(b *testing.B) {
	const lines = 8
	const lineSize = 16 * 1024
	data := makeJSONLines(lines, lineSize)
	b.ReportAllocs()
	b.SetBytes(int64(len(data)))
	for i := 0; i < b.N; i++ {
		reader := newJSONLineReader(&fixedReader{data: data})
		for j := 0; j < lines; j++ {
			backendBenchSinkLine, _ = reader.ReadLine()
		}
	}
}

// Silence unused-var lints when only a subset of benches is run.
var _ = backendBenchSinkOpts
var _ = backendBenchSinkResult
