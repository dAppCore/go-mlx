// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"encoding/binary"
	"testing"

	core "dappco.re/go"
	mlx "dappco.re/go/mlx"
)

func TestRunCommand_OfficialGemma4LocksJSON_Good(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"official-gemma4-locks", "-json"}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if stderr.String() != "" {
		t.Fatalf("stderr = %q, want empty", stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{
		`"kind": "official-gemma4-e2b-source-lock"`,
		`"source_checked_at": "2026-05-31"`,
		`"model_id": "google/gemma-4-E2B-it"`,
		`"revision": "905e84b50c4d2a365ebde34e685027578e6728db"`,
		`"model_id": "google/gemma-4-E2B-it-assistant"`,
		`"revision": "5810c41a67974da9c7bd6f3e6c69d5d13854d9f0"`,
		`"quantized_target_locks": [`,
		`"model_id": "mlx-community/gemma-4-e2b-it-8bit"`,
		`"model_id": "mlx-community/gemma-4-e2b-it-6bit"`,
		`"model_id": "mlx-community/gemma-4-e2b-it-4bit"`,
		`"quant_bits": 6`,
		`"platform_api_locks": [`,
		`"minimum_os": "macOS 26.0"`,
		`"introduced_in": "macOS 26.0"`,
		`"source_url": "https://developer.apple.com/documentation/macos-release-notes/macos-26-release-notes"`,
		`"source_url": "https://developer.apple.com/documentation/packagedescription/supportedplatform/macosversion/v26"`,
		`"source_url": "https://developer.apple.com/macos/whats-new/"`,
		`"source_url": "https://developer.apple.com/metal/whats-new/"`,
		`"source_url": "https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf"`,
		`"licence": "apache-2.0"`,
		`"gated": false`,
		`"config_sha256":`,
		`"tokenizer_sha256":`,
		`"safetensors_index_present": false`,
		`"safetensors_index_notes": "HF snapshot lists a single model.safetensors file and no model.safetensors.index.json."`,
	} {
		if !core.Contains(out, want) {
			t.Fatalf("stdout = %q, want %s", out, want)
		}
	}
}

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
	if core.Contains(out, "{{ bos_token }}") {
		t.Fatalf("stdout = %q, want default report to omit raw chat template body", out)
	}
}

func TestRunCommand_OfficialGemma4VerifyJSONIncludesChatTemplateWhenRequested_Good(t *testing.T) {
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
	code := runCommand(context.Background(), []string{"official-gemma4-verify", "-json", "-include-chat-template", "-role", lock.Role, dir}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if stderr.String() != "" {
		t.Fatalf("stderr = %q, want empty", stderr.String())
	}
	if !core.Contains(stdout.String(), "{{ bos_token }}") {
		t.Fatalf("stdout = %q, want raw chat template when explicitly requested", stdout.String())
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
	for _, want := range []string{
		`"assistant_layer_count": 4`,
		`"assistant_four_layer_drafter": true`,
		`"assistant_projection_tensors_ok": true`,
		`"assistant_ordered_embedding_tensors_ok": true`,
		`"target_kv_layer_types": [`,
		`"assistant_layer_types": [`,
		`"assistant_layer_types_covered_by_target": true`,
	} {
		if !core.Contains(out, want) {
			t.Fatalf("stdout = %q, want %s", out, want)
		}
	}
	if core.Contains(out, "{{ bos_token }}") {
		t.Fatalf("stdout = %q, want default pair report to omit raw chat template body", out)
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

func TestRunCommand_OfficialGemma4ControlCompareJSON_Good(t *testing.T) {
	targetLock, targetDir := officialGemma4VerifyTestSnapshot(t)
	controlDir := officialGemma4VerifyControlSnapshotFromTarget(t, targetDir)
	originalCompare := officialGemma4ControlCompare
	officialGemma4ControlCompare = func(targetDir, controlDir string) (mlx.OfficialGemma4E2BControlComparison, error) {
		return mlx.CompareOfficialGemma4E2BControlSnapshots(targetDir, controlDir, targetLock)
	}
	t.Cleanup(func() { officialGemma4ControlCompare = originalCompare })

	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"official-gemma4-control-compare", "-json", targetDir, controlDir}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if stderr.String() != "" {
		t.Fatalf("stderr = %q, want empty", stderr.String())
	}
	out := stdout.String()
	if !core.Contains(out, `"compatible": true`) || !core.Contains(out, `"quantization_differs": true`) {
		t.Fatalf("stdout = %q, want compatible official-vs-q4 comparison JSON", out)
	}
	if !core.Contains(out, `"retained_state_compatible": true`) || !core.Contains(out, `"prompt_cache_compatible": true`) {
		t.Fatalf("stdout = %q, want retained-State and prompt-cache compatibility flags", out)
	}
	if !core.Contains(out, `"model_id": "google/gemma-4-E2B-it"`) || !core.Contains(out, `"model_id": "mlx-community/gemma-4-e2b-it-4bit"`) {
		t.Fatalf("stdout = %q, want official target and archived q4 model IDs", out)
	}
	if !core.Contains(out, `"full_attention_interval": 5`) || !core.Contains(out, `"proportional_rope": true`) {
		t.Fatalf("stdout = %q, want attention and p-RoPE comparison fields", out)
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

func officialGemma4VerifyControlSnapshotFromTarget(t *testing.T, targetDir string) string {
	t.Helper()
	controlDir := core.PathJoin(t.TempDir(), "q4-control")
	if result := core.MkdirAll(controlDir, 0o755); !result.OK {
		t.Fatalf("MkdirAll control snapshot: %v", result.Value)
	}
	for _, name := range []string{
		"tokenizer.json",
		"tokenizer_config.json",
		"generation_config.json",
		"chat_template.jinja",
		"model.safetensors",
	} {
		read := core.ReadFile(core.PathJoin(targetDir, name))
		if !read.OK {
			t.Fatalf("ReadFile %s: %v", name, read.Value)
		}
		writeOfficialGemma4VerifyTestFile(t, controlDir, name, read.Value.([]byte))
	}
	read := core.ReadFile(core.PathJoin(targetDir, "config.json"))
	if !read.OK {
		t.Fatalf("ReadFile config.json: %v", read.Value)
	}
	config := core.Replace(core.AsString(read.Value.([]byte)), `"bits": 6`, `"bits": 4`)
	writeOfficialGemma4VerifyTestFile(t, controlDir, "config.json", []byte(config))
	return controlDir
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
	weights := officialGemma4VerifyAssistantTensorFixture(t)
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

func officialGemma4VerifyAssistantTensorFixture(t *testing.T) []byte {
	t.Helper()
	return officialGemma4VerifySafetensorsHeaderOnly(t, map[string][]int64{
		"pre_projection.weight":             {256, 3072},
		"post_projection.weight":            {1536, 256},
		"masked_embedding.centroids.weight": {2048, 256},
		"masked_embedding.token_ordering":   {2048, 128},
	})
}

func officialGemma4VerifySafetensorsHeaderOnly(t *testing.T, shapes map[string][]int64) []byte {
	t.Helper()
	type headerEntry struct {
		DType       string  `json:"dtype"`
		Shape       []int64 `json:"shape"`
		DataOffsets []int64 `json:"data_offsets"`
	}
	header := make(map[string]headerEntry, len(shapes))
	for name, shape := range shapes {
		header[name] = headerEntry{DType: "F32", Shape: shape, DataOffsets: []int64{0, 0}}
	}
	encoded := core.JSONMarshal(header)
	if !encoded.OK {
		t.Fatalf("JSONMarshal safetensors fixture: %v", encoded.Value)
	}
	headerBytes := encoded.Value.([]byte)
	out := make([]byte, 8+len(headerBytes))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(headerBytes)))
	copy(out[8:], headerBytes)
	return out
}

func writeOfficialGemma4VerifyTestFile(t *testing.T, dir, name string, data []byte) {
	t.Helper()
	if result := core.WriteFile(core.PathJoin(dir, name), data, 0o644); !result.OK {
		t.Fatalf("WriteFile %s: %v", name, result.Value)
	}
}
