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

func TestRunCommand_OfficialGemma4VerifyCacheRootJSON_Good(t *testing.T) {
	lock, cacheRoot, snapshotDir := officialGemma4VerifyTestCacheRoot(t)
	originalLookup := officialGemma4VerifyLockByRole
	officialGemma4VerifyLockByRole = func(role string) (mlx.OfficialGemma4E2BLock, bool) {
		if role != lock.Role {
			return mlx.OfficialGemma4E2BLock{}, false
		}
		return lock, true
	}
	t.Cleanup(func() { officialGemma4VerifyLockByRole = originalLookup })

	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"official-gemma4-verify", "-json", "-role", lock.Role, cacheRoot}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if stderr.String() != "" {
		t.Fatalf("stderr = %q, want empty", stderr.String())
	}
	out := stdout.String()
	if !core.Contains(out, core.Sprintf(`"snapshot_dir": %q`, snapshotDir)) || !core.Contains(out, `"verified": true`) {
		t.Fatalf("stdout = %q, want verified JSON report for resolved locked snapshot %q", out, snapshotDir)
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

func TestRunCommand_OfficialGemma4PairVerifyJSON_Good(t *testing.T) {
	targetLock, targetDir := officialGemma4VerifyTestSnapshot(t)
	assistantLock, assistantDir := officialGemma4VerifyAssistantTestSnapshot(t)
	originalInspect := officialGemma4PairInspect
	officialGemma4PairInspect = func(targetDir, assistantDir string) (mlx.OfficialGemma4E2BPairReport, error) {
		return mlx.InspectOfficialGemma4E2BPairLocalSnapshots(targetDir, assistantDir, targetLock, assistantLock)
	}
	t.Cleanup(func() { officialGemma4PairInspect = originalInspect })

	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"official-gemma4-pair-verify", "-json", targetDir, assistantDir}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if stderr.String() != "" {
		t.Fatalf("stderr = %q, want empty", stderr.String())
	}
	out := stdout.String()
	if !core.Contains(out, `"pair_ok": true`) || !core.Contains(out, `"assistant_attachable": true`) {
		t.Fatalf("stdout = %q, want verified official target+assistant pair report", out)
	}
	if !core.Contains(out, `"assistant_ordered_embeddings": true`) || !core.Contains(out, `"assistant_num_centroids": 2048`) || !core.Contains(out, `"assistant_centroid_intermediate_top_k": 32`) {
		t.Fatalf("stdout = %q, want official ordered-embedding assistant metadata", out)
	}
}

func TestRunCommand_OfficialGemma4PairVerifyCacheRootJSON_Good(t *testing.T) {
	targetLock, targetCacheRoot, targetSnapshotDir := officialGemma4VerifyTestCacheRoot(t)
	assistantLock, assistantCacheRoot, assistantSnapshotDir := officialGemma4VerifyAssistantTestCacheRoot(t)
	originalInspect := officialGemma4PairInspect
	officialGemma4PairInspect = func(targetDir, assistantDir string) (mlx.OfficialGemma4E2BPairReport, error) {
		return mlx.InspectOfficialGemma4E2BPairLocalSnapshots(targetDir, assistantDir, targetLock, assistantLock)
	}
	t.Cleanup(func() { officialGemma4PairInspect = originalInspect })

	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"official-gemma4-pair-verify", "-json", targetCacheRoot, assistantCacheRoot}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if stderr.String() != "" {
		t.Fatalf("stderr = %q, want empty", stderr.String())
	}
	out := stdout.String()
	if !core.Contains(out, `"pair_ok": true`) || !core.Contains(out, core.Sprintf(`"target_path": %q`, targetSnapshotDir)) || !core.Contains(out, core.Sprintf(`"assistant_path": %q`, assistantSnapshotDir)) {
		t.Fatalf("stdout = %q, want resolved cache-root target+assistant pair report", out)
	}
}

func TestRunCommand_OfficialGemma4VerifyHashMismatch_Bad(t *testing.T) {
	lock, dir := officialGemma4VerifyTestSnapshot(t)
	writeOfficialGemma4VerifyTestFile(t, dir, "config.json", []byte(`{
		"model_type": "gemma4",
		"architectures": ["Gemma4ForConditionalGeneration"],
		"text_config": {
			"model_type": "gemma4_text",
			"vocab_size": 262144,
			"hidden_size": 4096,
			"num_hidden_layers": 35,
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
			"vocab_size": 262144,
			"hidden_size": 1536,
			"hidden_size_per_layer_input": 256,
			"num_hidden_layers": 35,
			"num_attention_heads": 8,
			"num_key_value_heads": 1,
			"num_kv_shared_layers": 20,
			"head_dim": 256,
			"global_head_dim": 512,
			"max_position_embeddings": 131072,
			"sliding_window": 512,
			"layer_types": [
				"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
				"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
				"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
				"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
				"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
				"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
				"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"
			],
			"rope_parameters": {
				"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
				"sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
			}
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

func officialGemma4VerifyTestCacheRoot(t *testing.T) (mlx.OfficialGemma4E2BLock, string, string) {
	t.Helper()
	lock, sourceDir := officialGemma4VerifyTestSnapshot(t)
	return officialGemma4VerifyTestCacheRootFrom(t, lock, sourceDir)
}

func officialGemma4VerifyAssistantTestCacheRoot(t *testing.T) (mlx.OfficialGemma4E2BLock, string, string) {
	t.Helper()
	lock, sourceDir := officialGemma4VerifyAssistantTestSnapshot(t)
	return officialGemma4VerifyTestCacheRootFrom(t, lock, sourceDir)
}

func officialGemma4VerifyTestCacheRootFrom(t *testing.T, lock mlx.OfficialGemma4E2BLock, sourceDir string) (mlx.OfficialGemma4E2BLock, string, string) {
	t.Helper()
	cacheRoot := core.PathJoin(t.TempDir(), "models--google--gemma-4-E2B-it")
	snapshotDir := core.PathJoin(cacheRoot, "snapshots", lock.Revision)
	if result := core.MkdirAll(snapshotDir, 0o755); !result.OK {
		t.Fatalf("MkdirAll cache snapshot: %v", result.Value)
	}
	for _, name := range []string{
		"config.json",
		"tokenizer.json",
		"tokenizer_config.json",
		"generation_config.json",
		lock.WeightFile,
	} {
		read := core.ReadFile(core.PathJoin(sourceDir, name))
		if !read.OK {
			t.Fatalf("ReadFile %s: %v", name, read.Value)
		}
		writeOfficialGemma4VerifyTestFile(t, snapshotDir, name, read.Value.([]byte))
	}
	if lock.ChatTemplateSHA256 != "" {
		read := core.ReadFile(core.PathJoin(sourceDir, "chat_template.jinja"))
		if !read.OK {
			t.Fatalf("ReadFile chat_template.jinja: %v", read.Value)
		}
		writeOfficialGemma4VerifyTestFile(t, snapshotDir, "chat_template.jinja", read.Value.([]byte))
	}
	return lock, cacheRoot, snapshotDir
}

func officialGemma4VerifyAssistantTestSnapshot(t *testing.T) (mlx.OfficialGemma4E2BLock, string) {
	t.Helper()
	config := []byte(`{
		"model_type": "gemma4_assistant",
		"architectures": ["Gemma4AssistantForCausalLM"],
		"backbone_hidden_size": 1536,
		"num_centroids": 2048,
		"centroid_intermediate_top_k": 32,
		"use_ordered_embeddings": true,
		"text_config": {
			"model_type": "gemma4_text",
			"vocab_size": 262144,
			"hidden_size": 256,
			"num_hidden_layers": 4,
			"num_attention_heads": 4,
			"num_key_value_heads": 1,
			"num_kv_shared_layers": 4,
			"head_dim": 256,
			"global_head_dim": 512,
			"max_position_embeddings": 131072,
			"sliding_window": 512,
			"layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
			"rope_parameters": {
				"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
				"sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
			}
		}
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
	weights := []byte("assistant-weights")
	lock := mlx.OfficialGemma4E2BLock{
		Role:                   mlx.OfficialGemma4E2BRoleAssistant,
		ModelID:                "google/gemma-4-E2B-it-assistant",
		Revision:               "test-assistant-revision",
		ConfigSHA256:           core.SHA256Hex(config),
		TokenizerSHA256:        core.SHA256Hex(tokenizer),
		TokenizerConfigSHA256:  core.SHA256Hex(tokenizerConfig),
		GenerationConfigSHA256: core.SHA256Hex(generationConfig),
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
	writeOfficialGemma4VerifyTestFile(t, dir, lock.WeightFile, weights)
	return lock, dir
}

func writeOfficialGemma4VerifyTestFile(t *testing.T, dir, name string, data []byte) {
	t.Helper()
	if result := core.WriteFile(core.PathJoin(dir, name), data, 0o644); !result.OK {
		t.Fatalf("WriteFile %s: %v", name, result.Value)
	}
}
