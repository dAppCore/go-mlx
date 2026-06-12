// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

// Probe derivation from a validation set shaped exactly like the LEM
// gold-full rows: the first user turn of each line, fixed count, blanks
// and malformed lines skipped.
func TestSFT_ProbesFromValid_Good(t *testing.T) {
	path := core.JoinPath(t.TempDir(), "valid.jsonl")
	rows := `{"messages": [{"role": "user", "content": "design an offline-first social graph"}, {"role": "assistant", "content": "..."}]}

not json
{"messages": [{"role": "system", "content": "be kind"}, {"role": "user", "content": "what survives device seizure?"}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "user", "content": "third probe"}, {"role": "assistant", "content": "..."}]}
`
	if err := coreio.Local.Write(path, rows); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	prompts, err := sftProbesFromValid(path, 2)
	if err != nil {
		t.Fatalf("sftProbesFromValid: %v", err)
	}
	if len(prompts) != 2 {
		t.Fatalf("prompts = %d, want 2 (fixed probe count)", len(prompts))
	}
	if prompts[0] != "design an offline-first social graph" {
		t.Fatalf("prompt[0] = %q, want the first user turn", prompts[0])
	}
	if prompts[1] != "what survives device seizure?" {
		t.Fatalf("prompt[1] = %q, want the user turn past the system message", prompts[1])
	}
}

func TestSFT_ProbesFromValid_Bad(t *testing.T) {
	path := core.JoinPath(t.TempDir(), "valid.jsonl")
	if err := coreio.Local.Write(path, `{"messages": [{"role": "assistant", "content": "no user turns here"}]}`); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	if _, err := sftProbesFromValid(path, 4); err == nil {
		t.Fatal("a set with no user turns must refuse probe derivation")
	}
	if _, err := sftProbesFromValid(core.JoinPath(t.TempDir(), "missing.jsonl"), 4); err == nil {
		t.Fatal("a missing file must error")
	}
}
