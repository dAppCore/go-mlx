// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"testing"

	core "dappco.re/go"
)

func TestLoRAFusePairName_Good(t *testing.T) {
	pair, suffix, ok := loraFusePairName("model.layers.0.self_attn.q_proj.lora_a")
	if !ok || pair != "model.layers.0.self_attn.q_proj" || suffix != "a" {
		t.Fatalf("pair=%q suffix=%q ok=%v, want q_proj/a/true", pair, suffix, ok)
	}
	if got := loraFuseBaseWeightKey(pair); got != "model.layers.0.self_attn.q_proj.weight" {
		t.Fatalf("base weight key = %q", got)
	}

	pair, suffix, ok = loraFusePairName("model.layers.0.self_attn.q_proj.lora_B.weight")
	if !ok || pair != "model.layers.0.self_attn.q_proj" || suffix != "b" {
		t.Fatalf("PEFT pair=%q suffix=%q ok=%v, want q_proj/b/true", pair, suffix, ok)
	}

	for _, name := range []string{
		"layer.lora_a.weight",
		"layer.lora_A.weight",
		"layer.lora_A",
		"layer.lora_b.weight",
		"layer.lora_B",
	} {
		pair, suffix, ok := loraFusePairName(name)
		if !ok || pair != "layer" || (suffix != "a" && suffix != "b") {
			t.Fatalf("loraFusePairName(%q) = pair:%q suffix:%q ok:%v", name, pair, suffix, ok)
		}
	}
	if pair, suffix, ok := loraFusePairName("layer.weight"); ok || pair != "" || suffix != "" {
		t.Fatalf("loraFusePairName(non-lora) = pair:%q suffix:%q ok:%v", pair, suffix, ok)
	}
}

func TestPrepareLoRAFuse_OutputMustBePackDirectory_Bad(t *testing.T) {
	_, err := prepareLoRAFuse(context.Background(), FuseLoRAOptions{
		ModelPath:   "/tmp/source",
		AdapterPath: "/tmp/adapter",
		OutputPath:  "/tmp/fused.safetensors",
	})
	if err == nil {
		t.Fatal("expected output directory error")
	}
	if !core.Contains(err.Error(), "directory") {
		t.Fatalf("error = %v, want directory context", err)
	}
}

func TestPrepareLoRAFuse_ValidationErrors_Bad(t *testing.T) {
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := prepareLoRAFuse(cancelled, FuseLoRAOptions{}); err != context.Canceled {
		t.Fatalf("prepareLoRAFuse(cancelled) = %v, want context.Canceled", err)
	}
	if _, err := prepareLoRAFuse(context.Background(), FuseLoRAOptions{}); err == nil {
		t.Fatal("expected missing model path error")
	}
	if _, err := prepareLoRAFuse(context.Background(), FuseLoRAOptions{ModelPath: "/tmp/model"}); err == nil {
		t.Fatal("expected missing adapter path error")
	}
	if _, err := prepareLoRAFuse(context.Background(), FuseLoRAOptions{ModelPath: "/tmp/model", AdapterPath: "/tmp/adapter"}); err == nil {
		t.Fatal("expected missing output path error")
	}
}

func TestLoRAFuseDestinationAndMetadata_Good(t *testing.T) {
	base := t.TempDir()
	output := core.PathJoin(t.TempDir(), "fused")
	if result := core.MkdirAll(output, 0o755); !result.OK {
		t.Fatalf("mkdir output: %v", result.Value)
	}
	files := map[string]string{
		"config.json":              `{"model_type":"qwen3"}`,
		"tokenizer.json":           modelPackTokenizerJSON,
		"adapter_provenance.json":  `{"skip":true}`,
		"model.safetensors.index":  "skip",
		"notes.txt":                "keep",
		"tokenizer.model":          "keep model",
		"ignored.gguf":             "skip",
		"ignored.safetensors":      "skip",
		"model.safetensors.index2": "skip because contains",
	}
	for name, content := range files {
		writeModelPackFile(t, core.PathJoin(base, name), content)
	}

	if err := copyModelPackMetadata(base, output); err != nil {
		t.Fatalf("copyModelPackMetadata: %v", err)
	}
	for _, name := range []string{"config.json", "tokenizer.json", "notes.txt", "tokenizer.model"} {
		if stat := core.Stat(core.PathJoin(output, name)); !stat.OK {
			t.Fatalf("%s was not copied: %v", name, stat.Value)
		}
	}
	for _, name := range []string{"adapter_provenance.json", "ignored.gguf", "ignored.safetensors", "model.safetensors.index"} {
		if stat := core.Stat(core.PathJoin(output, name)); stat.OK {
			t.Fatalf("%s should not have been copied", name)
		}
	}
	if err := ensureEmptyFuseWeightDestination(core.PathJoin(t.TempDir(), "missing")); err != nil {
		t.Fatalf("missing destination should be accepted: %v", err)
	}
	if !samePath(base, base) {
		t.Fatal("samePath(base, base) = false, want true")
	}
}

func TestLoRAFuseDestinationAndMetadata_Bad(t *testing.T) {
	dir := t.TempDir()
	if result := core.WriteFile(core.PathJoin(dir, "model.safetensors"), []byte("weights"), 0o644); !result.OK {
		t.Fatalf("write weights: %v", result.Value)
	}
	if err := ensureEmptyFuseWeightDestination(dir); err == nil || !core.Contains(err.Error(), "already contains") {
		t.Fatalf("ensureEmptyFuseWeightDestination() error = %v", err)
	}
	if !isModelWeightMetadataCopySkip("MODEL.GGUF") || !isModelWeightMetadataCopySkip("adapter_provenance.json") {
		t.Fatal("expected model weight metadata files to be skipped")
	}
	if isModelWeightMetadataCopySkip("tokenizer.json") {
		t.Fatal("tokenizer.json should not be skipped")
	}
	if err := copyLocalFile(core.PathJoin(dir, "missing.json"), core.PathJoin(dir, "out.json")); err == nil {
		t.Fatal("expected copyLocalFile missing source error")
	}
}

func TestLoRAFuseAdapterWeightFiles_Good(t *testing.T) {
	dir := t.TempDir()
	a := core.PathJoin(dir, "b.safetensors")
	b := core.PathJoin(dir, "a.safetensors")
	for _, path := range []string{a, b} {
		if result := core.WriteFile(path, []byte("weights"), 0o644); !result.OK {
			t.Fatalf("write adapter weight: %v", result.Value)
		}
	}
	files, err := loraFuseAdapterWeightFiles(dir)
	if err != nil {
		t.Fatalf("loraFuseAdapterWeightFiles(dir): %v", err)
	}
	if len(files) != 2 || files[0] != b || files[1] != a {
		t.Fatalf("adapter files = %+v, want sorted", files)
	}
	files, err = loraFuseAdapterWeightFiles(a)
	if err != nil {
		t.Fatalf("loraFuseAdapterWeightFiles(file): %v", err)
	}
	if len(files) != 1 || files[0] != a {
		t.Fatalf("adapter file result = %+v, want %q", files, a)
	}
	if _, err := loraFuseAdapterWeightFiles(core.PathJoin(t.TempDir(), "empty")); err == nil {
		t.Fatal("expected no adapter safetensors error")
	}
}

func TestWriteLoRAFuseProvenance_Ugly(t *testing.T) {
	path := core.PathJoin(t.TempDir(), LoRAFuseProvenanceFile)
	err := writeLoRAFuseProvenance(path, LoRAFuseProvenance{
		Version:         1,
		OutputWeight:    "model.safetensors",
		FusedWeightKeys: []string{"z.weight", "a.weight"},
		Labels:          map[string]string{"run": "probe"},
	})
	if err != nil {
		t.Fatalf("writeLoRAFuseProvenance() error = %v", err)
	}
	read := core.ReadFile(path)
	if !read.OK {
		t.Fatalf("ReadFile provenance: %v", read.Value)
	}
	text := string(read.Value.([]byte))
	if !core.Contains(text, "model.safetensors") || !core.Contains(text, "probe") {
		t.Fatalf("provenance missing expected fields: %s", text)
	}
	parts := core.Split(text, "a.weight")
	if len(parts) < 2 || !core.Contains(parts[1], "z.weight") {
		t.Fatalf("fused keys are not sorted: %s", text)
	}
}
