// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"testing"

	core "dappco.re/go"
)

func TestMain_RunCommand_Good_Help(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"--help"}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q", code, stderr.String())
	}
	if !core.Contains(stdout.String(), "Usage: violet [flags]") {
		t.Fatalf("stdout = %q, want usage", stdout.String())
	}
}

func TestMain_RunCommand_Bad_UnknownFlag(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"--unknown"}, stdout, stderr)
	if code == 0 {
		t.Fatalf("exit code = %d, want non-zero", code)
	}
	if !core.Contains(stderr.String(), "flag provided but not defined") {
		t.Fatalf("stderr = %q, want unknown flag error", stderr.String())
	}
}

func TestMain_RunCommand_Bad_UnexpectedArg(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"serve"}, stdout, stderr)
	if code == 0 {
		t.Fatalf("exit code = %d, want non-zero", code)
	}
	if !core.Contains(stderr.String(), "unexpected argument") {
		t.Fatalf("stderr = %q, want unexpected argument error", stderr.String())
	}
}

func TestMain_RunCommand_Good_Version(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"--version"}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q", code, stderr.String())
	}
	if !core.Contains(stdout.String(), "violet ") {
		t.Fatalf("stdout = %q, want version", stdout.String())
	}
}

func TestMain_LoadRuntimeConfig_Good_FileAndEnvFallbacks(t *testing.T) {
	setCoreEnv(t, "VIOLET_MODEL_PATH", "/models/env-default")
	setCoreEnv(t, "VIOLET_EMBED_MODEL_PATH", "/models/env-embed")
	setCoreEnv(t, "VIOLET_SCORE_MODEL_PATH", "/models/env-score")
	setCoreEnv(t, "VIOLET_GENERATE_MODEL_PATH", "/models/env-generate")
	path := core.PathJoin(t.TempDir(), "violet.toml")
	if result := core.WriteFile(path, []byte(`
socket = "/tmp/violet.sock" # trailing comments are stripped
model = "/models/file-default"

[models]
generate = "/models/file-generate"
research = 'qwen#literal'
`), 0o644); !result.OK {
		t.Fatalf("write config: %v", result.Value)
	}

	cfg, err := loadRuntimeConfig(path)
	if err != nil {
		t.Fatalf("loadRuntimeConfig() error = %v", err)
	}
	if cfg.SocketPath != "/tmp/violet.sock" {
		t.Fatalf("SocketPath = %q", cfg.SocketPath)
	}
	if cfg.Models["default"] != "/models/file-default" {
		t.Fatalf("default model = %q", cfg.Models["default"])
	}
	if cfg.Models["embed"] != "/models/env-embed" || cfg.Models["score"] != "/models/env-score" {
		t.Fatalf("env fallback models = %+v", cfg.Models)
	}
	if cfg.Models["generate"] != "/models/file-generate" {
		t.Fatalf("generate model = %q, want config to beat env", cfg.Models["generate"])
	}
	if cfg.Models["research"] != "qwen#literal" {
		t.Fatalf("single quoted model = %q", cfg.Models["research"])
	}
}

func TestMain_ReadConfigFile_Bad_InvalidSyntax(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "violet.toml")
	if result := core.WriteFile(path, []byte("socket /tmp/violet.sock\n"), 0o644); !result.OK {
		t.Fatalf("write config: %v", result.Value)
	}

	err := readConfigFile(path, &runtimeConfig{Models: make(map[string]string)})
	if err == nil {
		t.Fatal("expected syntax error")
	}
	if !core.Contains(err.Error(), "expected key = value") {
		t.Fatalf("error = %v, want key/value context", err)
	}
}

func TestMain_ParseConfigValue_Bad_UnmatchedSingleQuote(t *testing.T) {
	_, err := parseConfigValue("'unterminated")
	if err == nil {
		t.Fatal("expected unmatched single quote error")
	}
}

func TestMain_ParseConfigValue_Good_QuotesAndComments(t *testing.T) {
	value, err := parseConfigValue(`"a # value"`)
	if err != nil {
		t.Fatalf("parseConfigValue(double) error = %v", err)
	}
	if value != "a # value" {
		t.Fatalf("double quoted value = %q", value)
	}
	value, err = parseConfigValue(`'literal # value'`)
	if err != nil {
		t.Fatalf("parseConfigValue(single) error = %v", err)
	}
	if value != "literal # value" {
		t.Fatalf("single quoted value = %q", value)
	}
	if got := stripConfigComment(`model = "a # b" # comment`); got != `model = "a # b" ` {
		t.Fatalf("stripConfigComment() = %q", got)
	}
}

func TestMain_CommandResultError_Ugly(t *testing.T) {
	err := commandResultError(core.Result{OK: false, Value: "not an error"})
	if err == nil || !core.Contains(err.Error(), "violet command operation failed") {
		t.Fatalf("commandResultError() = %v", err)
	}
}

func setCoreEnv(t *testing.T, key, value string) {
	t.Helper()
	old, had := core.LookupEnv(key)
	if result := core.Setenv(key, value); !result.OK {
		t.Fatalf("set %s: %v", key, result.Value)
	}
	t.Cleanup(func() {
		if had {
			core.Setenv(key, old)
			return
		}
		core.Unsetenv(key)
	})
}
