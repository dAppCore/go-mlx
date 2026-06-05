// SPDX-Licence-Identifier: EUPL-1.2

package loraadapter

import "testing"

func TestParseConfig_Aliases_Good(t *testing.T) {
	cfg, err := ParseConfig([]byte(`{"r":4,"lora_alpha":12,"target_modules":["q_proj","v_proj"]}`))
	if err != nil {
		t.Fatalf("ParseConfig() error = %v", err)
	}
	if cfg.Rank != 4 || cfg.Alpha != 12 || cfg.Scale != 3 {
		t.Fatalf("config rank/alpha/scale = %d/%f/%f, want 4/12/3", cfg.Rank, cfg.Alpha, cfg.Scale)
	}
	if !sameStrings(cfg.TargetKeys, []string{"q_proj", "v_proj"}) {
		t.Fatalf("TargetKeys = %v, want target_modules alias", cfg.TargetKeys)
	}

	missing, err := ParseConfig([]byte(`{}`))
	if err != nil {
		t.Fatalf("ParseConfig(missing) error = %v", err)
	}
	if missing.Rank != 0 || missing.Alpha != 0 || missing.Scale != 0 {
		t.Fatalf("missing rank/alpha/scale = %d/%f/%f, want zero metadata", missing.Rank, missing.Alpha, missing.Scale)
	}
}

func TestNormalizeForNativeLoad_Defaults_Good(t *testing.T) {
	cfg := NormalizeForNativeLoad(Config{})
	if cfg.Rank != 8 || cfg.Alpha != 16 || cfg.Scale != 2 {
		t.Fatalf("default rank/alpha/scale = %d/%f/%f, want 8/16/2", cfg.Rank, cfg.Alpha, cfg.Scale)
	}

	cfg = NormalizeForNativeLoad(Config{Rank: 4, Scale: 1.5})
	if cfg.Alpha != 6 || cfg.Scale != 1.5 {
		t.Fatalf("scale-derived native alpha/scale = %f/%f, want 6/1.5", cfg.Alpha, cfg.Scale)
	}
}

func TestParseConfig_TargetPrecedence_Good(t *testing.T) {
	cfg, err := ParseConfig([]byte(`{
		"target_keys":["explicit"],
		"target_modules":["peft"],
		"lora_layers":["mlx-lm"]
	}`))
	if err != nil {
		t.Fatalf("ParseConfig() error = %v", err)
	}
	if !sameStrings(cfg.TargetKeys, []string{"explicit"}) {
		t.Fatalf("TargetKeys = %v, want explicit target_keys precedence", cfg.TargetKeys)
	}

	cfg, err = ParseConfig([]byte(`{
		"target_modules":["peft"],
		"lora_layers":["mlx-lm"]
	}`))
	if err != nil {
		t.Fatalf("ParseConfig(peft) error = %v", err)
	}
	if !sameStrings(cfg.TargetKeys, []string{"peft"}) {
		t.Fatalf("TargetKeys = %v, want PEFT target_modules before lora_layers", cfg.TargetKeys)
	}

	cfg, err = ParseConfig([]byte(`{"lora_layers":["mlx-lm"]}`))
	if err != nil {
		t.Fatalf("ParseConfig(mlx-lm) error = %v", err)
	}
	if !sameStrings(cfg.TargetKeys, []string{"mlx-lm"}) {
		t.Fatalf("TargetKeys = %v, want lora_layers fallback", cfg.TargetKeys)
	}
}

func TestParseConfig_BadInvalidJSON(t *testing.T) {
	if _, err := ParseConfig([]byte(`{broken`)); err == nil {
		t.Fatal("expected invalid JSON error")
	}
}

func sameStrings(got, want []string) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range want {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}
