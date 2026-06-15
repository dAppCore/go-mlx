// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	core "dappco.re/go"
)

// ExampleLoad shows the file round-trip: Save writes a snapshot to a path and
// Load reads it back, recovering the architecture.
func ExampleLoad() {
	dir := core.MkdirTemp("", "kv-load-example-*").Value.(string)
	path := core.PathJoin(dir, "snapshot.kvbin")
	if err := testSnapshot().Save(path); err != nil {
		core.Println("save error:", err)
		return
	}

	loaded, err := Load(path)
	if err != nil {
		core.Println("load error:", err)
		return
	}
	core.Println("architecture:", loaded.Architecture)
	// Output:
	// architecture: gemma4_text
}
