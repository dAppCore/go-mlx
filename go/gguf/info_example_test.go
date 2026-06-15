// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"encoding/binary"

	core "dappco.re/go"
)

// ExampleReadInfo summarises a GGUF file's architecture and tensor count
// without materialising weights into MLX.
func ExampleReadInfo() {
	path, cleanup := writeExampleGGUF()
	defer cleanup()

	info, err := ReadInfo(path)
	if err != nil {
		core.Println(err.Error())
		return
	}
	core.Println(info.Architecture, info.TensorCount, info.Valid())
	// Output: gemma4 1 true
}

// ExampleDiscoverModels walks a directory tree and reports the loadable models
// it finds, classified by on-disk format.
func ExampleDiscoverModels() {
	path, cleanup := writeExampleGGUF()
	defer cleanup()

	models := DiscoverModels(core.PathDir(path))
	core.Println(len(models), models[0].Format)
	// Output: 1 gguf
}

// ExampleInfo_Valid shows the validation gate on a parsed Info: a clean GGUF
// reads back Valid, so callers can trust its tensor metadata before loading.
// Valid reports false only when a GGUFValidationError-severity issue is present
// (warnings do not fail it).
func ExampleInfo_Valid() {
	path, cleanup := writeExampleGGUF()
	defer cleanup()

	info, err := ReadInfo(path)
	if err != nil {
		core.Println(err.Error())
		return
	}
	core.Println(info.Valid())

	// An Info carrying an error-severity validation issue is not Valid; one
	// carrying only a warning still is.
	withError := Info{ValidationIssues: []ValidationIssue{{Severity: GGUFValidationError, Code: "unknown_tensor_type"}}}
	withWarning := Info{ValidationIssues: []ValidationIssue{{Severity: GGUFValidationWarning, Code: "missing_alignment"}}}
	core.Println(withError.Valid(), withWarning.Valid())
	// Output:
	// true
	// false true
}

// writeMultiTypeExampleGGUF emits a header-only GGUF carrying three metadata
// values of different types (string, uint32, bool) and no tensors — enough to
// demonstrate Metadata's per-type decode. T-free, like writeExampleGGUF.
func writeMultiTypeExampleGGUF() (string, func()) {
	dirResult := core.MkdirTemp("", "go-mlx-gguf-meta-example-*")
	if !dirResult.OK {
		panic(dirResult.Value)
	}
	dir := dirResult.Value.(string)
	path := core.PathJoin(dir, "model.gguf")

	created := core.Create(path)
	if !created.OK {
		core.RemoveAll(dir)
		panic(created.Value)
	}
	file := created.Value.(*core.OSFile)
	fail := func(v any) {
		file.Close()
		core.RemoveAll(dir)
		panic(v)
	}
	write := func(value any) {
		if err := binary.Write(file, binary.LittleEndian, value); err != nil {
			fail(err)
		}
	}
	writeString := func(value string) {
		write(uint64(len(value)))
		if _, err := file.Write([]byte(value)); err != nil {
			fail(err)
		}
	}

	if _, err := file.Write([]byte("GGUF")); err != nil {
		fail(err)
	}
	write(uint32(3)) // version
	write(uint64(0)) // tensor count
	write(uint64(3)) // metadata count

	writeString("general.architecture")
	write(uint32(ValueTypeString))
	writeString("qwen3")

	writeString("qwen3.block_count")
	write(uint32(ValueTypeUint32))
	write(uint32(28))

	writeString("general.thinking")
	write(uint32(7)) // ggufValueTypeBool
	write(uint8(1))  // true

	file.Close()
	return path, func() { core.RemoveAll(dir) }
}

// writeExampleGGUF emits a minimal but structurally valid GGUF (one string
// metadata key, one quantised tensor) to a fresh temp dir and returns its path
// plus a cleanup func. Mirrors writeTestGGUF without the *testing.T dependency.
func writeExampleGGUF() (string, func()) {
	dirResult := core.MkdirTemp("", "go-mlx-gguf-example-*")
	if !dirResult.OK {
		panic(dirResult.Value)
	}
	dir := dirResult.Value.(string)
	path := core.PathJoin(dir, "model.gguf")

	created := core.Create(path)
	if !created.OK {
		core.RemoveAll(dir)
		panic(created.Value)
	}
	file := created.Value.(*core.OSFile)

	fail := func(v any) {
		file.Close()
		core.RemoveAll(dir)
		panic(v)
	}
	write := func(value any) {
		if err := binary.Write(file, binary.LittleEndian, value); err != nil {
			fail(err)
		}
	}
	writeString := func(value string) {
		write(uint64(len(value)))
		if _, err := file.Write([]byte(value)); err != nil {
			fail(err)
		}
	}

	if _, err := file.Write([]byte("GGUF")); err != nil {
		fail(err)
	}
	write(uint32(3)) // version
	write(uint64(1)) // tensor count
	write(uint64(1)) // metadata count
	writeString("general.architecture")
	write(uint32(ValueTypeString))
	writeString("gemma4")
	// One quantised tensor: name, ndim, dims, type, offset.
	writeString("blk.0.attn_q.weight")
	write(uint32(2))
	write(uint64(256))
	write(uint64(128))
	write(uint32(TensorTypeQ8_0))
	write(uint64(0))
	file.Close()

	return path, func() { core.RemoveAll(dir) }
}
