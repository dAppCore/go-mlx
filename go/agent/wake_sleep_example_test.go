// SPDX-Licence-Identifier: EUPL-1.2

package agent

import (
	"fmt"

	state "dappco.re/go/inference/state"
	"dappco.re/go/mlx/kv"
)

// ExampleSleepURIs shows how a single EntryURI is expanded into the
// derived bundle and index URIs used by a sleep round.
func ExampleSleepURIs() {
	entryURI, bundleURI, indexURI, err := SleepURIs(SleepOptions{EntryURI: "mlx://agent/session-1"})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(entryURI)
	fmt.Println(bundleURI)
	fmt.Println(indexURI)
	// Output:
	// mlx://agent/session-1
	// mlx://agent/session-1/bundle
	// mlx://agent/session-1/index
}

// ExampleSleepBlockOptions shows the defaulting applied to an empty
// BlockOptions: native KV encoding, a derived blocks URI, and the
// canonical "state" label appended.
func ExampleSleepBlockOptions() {
	blockOpts := SleepBlockOptions(SleepOptions{Title: "session-1"}, "mlx://agent/session-1/bundle")
	fmt.Println(blockOpts.KVEncoding)
	fmt.Println(blockOpts.URI)
	fmt.Println(blockOpts.Labels)
	// Output:
	// native
	// mlx://agent/session-1/bundle/blocks
	// [state]
}

// ExampleWakeReportFromSleep shows converting a SleepReport into the
// WakeReport a caller continues with in-process (no reload needed).
func ExampleWakeReportFromSleep() {
	wake := WakeReportFromSleep(&SleepReport{
		EntryURI:   "mlx://agent/session-1",
		Title:      "session-1",
		TokenCount: 2048,
		BlockSize:  512,
	})
	fmt.Println(wake.EntryURI)
	fmt.Println(wake.PrefixTokens)
	fmt.Println(wake.BlocksRead)
	// Output:
	// mlx://agent/session-1
	// 2048
	// 0
}

// ExampleCloneWakeReport shows that the clone is an independent copy:
// mutating it leaves the original untouched.
func ExampleCloneWakeReport() {
	original := &WakeReport{Title: "session-1", PrefixTokens: 2048}
	clone := CloneWakeReport(original)
	clone.Title = "mutated"
	fmt.Println(original.Title)
	fmt.Println(clone.Title)
	// Output:
	// session-1
	// mutated
}

// ExampleNewSleepReport assembles a durable sleep report from a freshly
// built index and bundle, printing the load-bearing scalar fields.
func ExampleNewSleepReport() {
	bundle := &kv.StateBlockBundle{
		Version:      kv.MemvidBlockVersion,
		Kind:         kv.MemvidBlockBundleKind,
		SnapshotHash: "snap",
		KVEncoding:   kv.EncodingNative,
		Architecture: "qwen3",
		TokenCount:   4,
		BlockSize:    2,
		NumLayers:    1,
		NumHeads:     1,
		SeqLen:       4,
		HeadDim:      2,
		Blocks: []kv.StateBlockRef{
			{Index: 0, TokenStart: 0, TokenCount: 2},
			{Index: 1, TokenStart: 2, TokenCount: 2},
		},
	}
	opts := SleepOptions{Title: "session-1"}
	idx, err := NewSleepIndex(bundle, opts, "mlx://agent/session-1", "mlx://agent/session-1/bundle")
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	report := NewSleepReport(idx, bundle, opts,
		"mlx://agent/session-1", "mlx://agent/session-1/bundle", "mlx://agent/session-1/index",
		state.ChunkRef{}, state.ChunkRef{})
	fmt.Println(report.EntryURI)
	fmt.Println(report.TokenCount)
	fmt.Println(report.BlocksWritten)
	// Output:
	// mlx://agent/session-1
	// 4
	// 2
}
