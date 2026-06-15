// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"encoding/binary"

	core "dappco.re/go"
)

// ExampleNormalizeQuantType shows how a free-form quant-type label is folded
// to the canonical lower-snake form the rest of the package keys on.
func ExampleNormalizeQuantType() {
	core.Println(NormalizeQuantType("Q4_K_M"))
	core.Println(NormalizeQuantType("Q5-K M"))
	core.Println(NormalizeQuantType("  BF16  "))
	// Output:
	// q4_k_m
	// q5_k_m
	// bf16
}

// ExampleValidationSummary renders a one-line summary of GGUF validation
// findings; tensor-scoped issues print as code:tensor.
func ExampleValidationSummary() {
	summary := ValidationSummary([]ValidationIssue{
		{Severity: GGUFValidationError, Code: "shape_mismatch", Tensor: "blk.0.attn_q.weight"},
		{Severity: GGUFValidationWarning, Code: "missing_alignment"},
	})
	core.Println(summary)
	core.Println(ValidationSummary(nil))
	// Output:
	// shape_mismatch:blk.0.attn_q.weight, missing_alignment
	// unknown validation failure
}

// ExampleMetadata reads a .gguf file's key/value metadata without loading any
// tensor data — here from a tiny synthetic file written to a temp dir.
func ExampleMetadata() {
	path, cleanup := writeExampleGGUF()
	defer cleanup()

	meta, err := Metadata(path)
	if err != nil {
		core.Println(err.Error())
		return
	}
	arch, _ := meta["general.architecture"].(string)
	core.Println(arch)
	// Output: gemma4
}

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
