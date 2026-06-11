package mlx

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/internal/metaltest"
	"dappco.re/go/mlx/memory"
)

func TestDecodeDeterminism_LiveModel(t *testing.T) {
	if !metaltest.RunModelEvalTests {
		t.Skip("model-eval test")
	}
	dir := metaltest.HFModelPath(t, "mlx-community/gemma-4-e2b-it-4bit")
	m, err := LoadModel(dir, WithKVCacheMode(memory.KVCacheModePaged), WithContextLength(4096))
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer m.Close()
	ctx := context.Background()
	run := func() string {
		sess, err := m.NewSession()
		if err != nil {
			t.Fatalf("NewSession: %v", err)
		}
		defer sess.Close()
		if err := sess.Prefill("Write a long, detailed story about a clockmaker who repairs time itself."); err != nil {
			t.Fatalf("Prefill: %v", err)
		}
		text := core.NewBuilder()
		for tok := range sess.GenerateStream(ctx, WithMaxTokens(640), WithTemperature(0)) {
			text.WriteString(tok.Text)
		}
		if err := sess.Err(); err != nil {
			t.Fatalf("generate: %v", err)
		}
		return text.String()
	}
	a, b := run(), run()
	if a != b {
		i := 0
		for i < len(a) && i < len(b) && a[i] == b[i] {
			i++
		}
		t.Errorf("same-lane decode is non-deterministic; first byte diff at %d:\n  a %q\n  b %q", i, a[max(0,i-40):min(len(a),i+40)], b[max(0,i-40):min(len(b),i+40)])
	}
}
