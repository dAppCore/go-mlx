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
