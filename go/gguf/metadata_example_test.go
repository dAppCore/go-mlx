// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	core "dappco.re/go"
)

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

// ExampleMetadata_valueTypes shows that Metadata decodes each GGUF value type
// to its native Go type: a string arch, a uint32 count, and a bool flag all
// come back ready to type-assert. This is the read path engines use to pull
// architecture-specific config keys (e.g. the gemma4-assistant.* drafter keys)
// without materialising any tensor data.
func ExampleMetadata_valueTypes() {
	path, cleanup := writeMultiTypeExampleGGUF()
	defer cleanup()

	meta, err := Metadata(path)
	if err != nil {
		core.Println(err.Error())
		return
	}
	arch, _ := meta["general.architecture"].(string)
	blocks, _ := meta["qwen3.block_count"].(uint32)
	thinking, _ := meta["general.thinking"].(bool)
	core.Println(arch, blocks, thinking)
	// Output: qwen3 28 true
}
