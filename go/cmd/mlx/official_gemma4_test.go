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
	if !core.Contains(out, `"architecture_ok": true`) || !core.Contains(out, `"native_loadable": true`) || !core.Contains(out, `"architecture": "gemma4_text"`) {
		t.Fatalf("stdout = %q, want official Gemma 4 pack preflight in JSON report", out)
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
	writeOfficialGemma4VerifyTestFile(t, dir, "config.json", []byte(`{
		"model_type": "gemma4",
		"architectures": ["Gemma4ForConditionalGeneration"],
		"text_config": {
			"model_type": "gemma4_text",
			"vocab_size": 262208,
			"hidden_size": 4096,
			"num_hidden_layers": 26,
			"max_position_embeddings": 131072
		},
		"quantization_config": {"bits": 6, "group_size": 64}
	}`))
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
	config := []byte(`{
		"model_type": "gemma4",
		"architectures": ["Gemma4ForConditionalGeneration"],
		"text_config": {
			"model_type": "gemma4_text",
			"vocab_size": 262208,
			"hidden_size": 2048,
			"num_hidden_layers": 26,
			"max_position_embeddings": 131072
		},
		"quantization_config": {"bits": 6, "group_size": 64}
	}`)
	tokenizer := []byte(`{
		"model": {
			"type": "BPE",
			"vocab": {"h": 0, "e": 1, "l": 2, "o": 3},
			"merges": ["h e"],
			"byte_fallback": false
		},
		"added_tokens": [
			{"id": 100, "content": "<bos>", "special": true},
			{"id": 101, "content": "<eos>", "special": true}
		]
	}`)
	tokenizerConfig := []byte(`{"model_max_length": 131072}`)
	generationConfig := []byte(`{"max_new_tokens": 8192}`)
	chatTemplate := []byte(`{{ bos_token }}{% for message in messages %}{{ message["content"] }}{% endfor %}`)
	weights := []byte("weights")
	lock := mlx.OfficialGemma4E2BLock{
		Role:                   mlx.OfficialGemma4E2BRoleTarget,
		ModelID:                "google/gemma-4-E2B-it",
		Revision:               "test-revision",
		ConfigSHA256:           core.SHA256Hex(config),
		TokenizerSHA256:        core.SHA256Hex(tokenizer),
		TokenizerConfigSHA256:  core.SHA256Hex(tokenizerConfig),
		GenerationConfigSHA256: core.SHA256Hex(generationConfig),
		ChatTemplateSHA256:     core.SHA256Hex(chatTemplate),
		WeightFile:             "model.safetensors",
		WeightSHA256:           core.SHA256Hex(weights),
		WeightBytes:            uint64(len(weights)),
	}
	dir := core.PathJoin(t.TempDir(), lock.Revision)
	if result := core.MkdirAll(dir, 0o755); !result.OK {
		t.Fatalf("MkdirAll snapshot: %v", result.Value)
	}
	writeOfficialGemma4VerifyTestFile(t, dir, "config.json", config)
	writeOfficialGemma4VerifyTestFile(t, dir, "tokenizer.json", tokenizer)
	writeOfficialGemma4VerifyTestFile(t, dir, "tokenizer_config.json", tokenizerConfig)
	writeOfficialGemma4VerifyTestFile(t, dir, "generation_config.json", generationConfig)
	writeOfficialGemma4VerifyTestFile(t, dir, "chat_template.jinja", chatTemplate)
	writeOfficialGemma4VerifyTestFile(t, dir, lock.WeightFile, weights)
	return lock, dir
}

func writeOfficialGemma4VerifyTestFile(t *testing.T, dir, name string, data []byte) {
	t.Helper()
	if result := core.WriteFile(core.PathJoin(dir, name), data, 0o644); !result.OK {
		t.Fatalf("WriteFile %s: %v", name, result.Value)
	}
}
