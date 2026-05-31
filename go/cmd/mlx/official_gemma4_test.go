// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"testing"

	core "dappco.re/go"
	mlx "dappco.re/go/mlx"
)

func TestRunCommand_OfficialGemma4VerifyJSON_Good(t *testing.T) {
	lock, dir := officialGemma4VerifyTestSnapshot(t)
	originalLookup := officialGemma4VerifyLockByRole
	officialGemma4VerifyLockByRole = func(role string) (mlx.OfficialGemma4E2BLock, bool) {
		if role != lock.Role {
			return mlx.OfficialGemma4E2BLock{}, false
		}
		return lock, true
	}
	t.Cleanup(func() { officialGemma4VerifyLockByRole = originalLookup })

	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"official-gemma4-verify", "-json", "-role", lock.Role, dir}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if stderr.String() != "" {
		t.Fatalf("stderr = %q, want empty", stderr.String())
	}
	out := stdout.String()
	if !core.Contains(out, `"verified": true`) || !core.Contains(out, `"role": "target"`) || !core.Contains(out, `"model_id": "google/gemma-4-E2B-it"`) {
		t.Fatalf("stdout = %q, want verified official Gemma 4 JSON report", out)
	}
}

func TestRunCommand_OfficialGemma4VerifyInvalidRole_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"official-gemma4-verify", "-role", "draft", "/models/snapshot"}, stdout, stderr)
	if code != 2 {
		t.Fatalf("exit code = %d, want usage error 2; stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !core.Contains(stderr.String(), "unknown official Gemma 4 E2B role") || !core.Contains(stderr.String(), "draft") {
		t.Fatalf("stderr = %q, want invalid official Gemma 4 role", stderr.String())
	}
}

func TestRunCommand_OfficialGemma4VerifyHashMismatch_Bad(t *testing.T) {
	lock, dir := officialGemma4VerifyTestSnapshot(t)
	writeOfficialGemma4VerifyTestFile(t, dir, "config.json", []byte("changed"))
	originalLookup := officialGemma4VerifyLockByRole
	officialGemma4VerifyLockByRole = func(role string) (mlx.OfficialGemma4E2BLock, bool) {
		if role != lock.Role {
			return mlx.OfficialGemma4E2BLock{}, false
		}
		return lock, true
	}
	t.Cleanup(func() { officialGemma4VerifyLockByRole = originalLookup })

	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"official-gemma4-verify", "-role", lock.Role, dir}, stdout, stderr)
	if code != 1 {
		t.Fatalf("exit code = %d, want verification failure 1; stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if stdout.String() != "" {
		t.Fatalf("stdout = %q, want empty", stdout.String())
	}
	if !core.Contains(stderr.String(), "config.json") || !core.Contains(stderr.String(), "SHA-256") {
		t.Fatalf("stderr = %q, want config SHA-256 mismatch", stderr.String())
	}
}

func officialGemma4VerifyTestSnapshot(t *testing.T) (mlx.OfficialGemma4E2BLock, string) {
	t.Helper()
	lock := mlx.OfficialGemma4E2BLock{
		Role:                   mlx.OfficialGemma4E2BRoleTarget,
		ModelID:                "google/gemma-4-E2B-it",
		Revision:               "test-revision",
		ConfigSHA256:           core.SHA256Hex([]byte("config")),
		TokenizerSHA256:        core.SHA256Hex([]byte("tokenizer")),
		TokenizerConfigSHA256:  core.SHA256Hex([]byte("tokenizer-config")),
		GenerationConfigSHA256: core.SHA256Hex([]byte("generation-config")),
		ChatTemplateSHA256:     core.SHA256Hex([]byte("chat-template")),
		WeightFile:             "model.safetensors",
		WeightSHA256:           core.SHA256Hex([]byte("weights")),
		WeightBytes:            uint64(len("weights")),
	}
	dir := core.PathJoin(t.TempDir(), lock.Revision)
	if result := core.MkdirAll(dir, 0o755); !result.OK {
		t.Fatalf("MkdirAll snapshot: %v", result.Value)
	}
	writeOfficialGemma4VerifyTestFile(t, dir, "config.json", []byte("config"))
	writeOfficialGemma4VerifyTestFile(t, dir, "tokenizer.json", []byte("tokenizer"))
	writeOfficialGemma4VerifyTestFile(t, dir, "tokenizer_config.json", []byte("tokenizer-config"))
	writeOfficialGemma4VerifyTestFile(t, dir, "generation_config.json", []byte("generation-config"))
	writeOfficialGemma4VerifyTestFile(t, dir, "chat_template.jinja", []byte("chat-template"))
	writeOfficialGemma4VerifyTestFile(t, dir, lock.WeightFile, []byte("weights"))
	return lock, dir
}

func writeOfficialGemma4VerifyTestFile(t *testing.T, dir, name string, data []byte) {
	t.Helper()
	if result := core.WriteFile(core.PathJoin(dir, name), data, 0o644); !result.OK {
		t.Fatalf("WriteFile %s: %v", name, result.Value)
	}
}
