// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// writeTuningProfile returns the mkdir error when the profile's parent
// directory cannot be created (the parent path is a regular file) — main.go
// ~487.
func TestWriteTuningProfile_MkdirFails_Bad(t *testing.T) {
	tmp := t.TempDir()
	fileAsDir := core.PathJoin(tmp, "afile")
	if r := core.WriteFile(fileAsDir, []byte("x"), 0o644); !r.OK {
		t.Fatalf("seed file: %v", r.Value)
	}
	err := writeTuningProfile(core.PathJoin(fileAsDir, "sub", "profile.json"), inference.TuningProfile{})
	if err == nil {
		t.Fatal("want a mkdir error when the profile parent path is a regular file")
	}
}

// writeTuningProfile returns the write error when the target path is an
// existing directory — main.go ~490.
func TestWriteTuningProfile_WriteFails_Bad(t *testing.T) {
	tmp := t.TempDir()
	dirAsFile := core.PathJoin(tmp, "adir")
	if r := core.MkdirAll(dirAsFile, 0o755); !r.OK {
		t.Fatalf("seed dir: %v", r.Value)
	}
	if err := writeTuningProfile(dirAsFile, inference.TuningProfile{}); err == nil {
		t.Fatal("want a write error when the profile target path is a directory")
	}
}
