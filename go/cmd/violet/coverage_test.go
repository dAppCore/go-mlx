// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"os"
	"os/exec"
	"strings"
	"testing"

	core "dappco.re/go"
)

// violetSpikeMainEnv gates the in-child re-execution of main(). The parent test
// spawns the coverage-instrumented test binary with this set so main()'s own
// statements register against the parent coverage profile; the child calls
// main() with harmless arguments and exits before the test framework resumes.
const violetSpikeMainEnv = "VIOLET_TEST_RUN_MAIN"

// TestMain_Main_Good_SubprocessVersion exercises func main() itself. main()
// ends in core.Exit, so it cannot run in-process under the test harness; the
// parent re-execs the instrumented test binary, the child runs main()
// --version (a clean exit-0 path that binds no socket and loads no model), and
// the child's coverage flushes into the shared profile.
func TestMain_Main_Good_SubprocessVersion(t *testing.T) {
	if os.Getenv(violetSpikeMainEnv) == "1" {
		os.Args = []string{"violet", "--version"}
		main()
		return
	}

	cmd := exec.Command(os.Args[0], "-test.run=TestMain_Main_Good_SubprocessVersion")
	cmd.Env = append(os.Environ(), violetSpikeMainEnv+"=1")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("child main() --version failed: %v\noutput:\n%s", err, out)
	}
	if !strings.Contains(string(out), "violet ") {
		t.Fatalf("child output = %q, want version banner", out)
	}
}

// TestMain_RunCommand_Good_DaemonStartsAndStops drives runCommand all the way
// into daemon.NewServer + ListenAndServe with an already-cancelled context and
// an empty model set. The server binds the socket, observes the dead context,
// and returns a nil error, so runCommand returns 0. Empty Models keeps the
// generate backend nil so no model is loaded under metal_runtime.
func TestMain_RunCommand_Good_DaemonStartsAndStops(t *testing.T) {
	clearVioletEnv(t)
	socketPath := shortVioletSocket(t)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(ctx, []string{"--socket", socketPath}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q", code, stderr.String())
	}
}

// TestMain_RunCommand_Bad_ListenFails forces the daemon error return. A regular
// file already sitting at the socket path makes ListenAndServe refuse to
// replace a non-socket, so runCommand reports the error and returns 1.
func TestMain_RunCommand_Bad_ListenFails(t *testing.T) {
	clearVioletEnv(t)
	socketPath := core.PathJoin(t.TempDir(), "violet.sock")
	if result := core.WriteFile(socketPath, []byte("not a socket"), 0o600); !result.OK {
		t.Fatalf("seed non-socket file: %v", result.Value)
	}

	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"--socket", socketPath}, stdout, stderr)
	if code != 1 {
		t.Fatalf("exit code = %d, want 1; stderr=%q", code, stderr.String())
	}
	if !core.Contains(stderr.String(), "violet:") {
		t.Fatalf("stderr = %q, want violet error prefix", stderr.String())
	}
}

// TestMain_RunCommand_Bad_ConfigLoadFails covers the loadRuntimeConfig error
// return: an explicit -config pointing at a malformed file fails to parse, so
// runCommand prints the error and returns 1 before ever touching the daemon.
func TestMain_RunCommand_Bad_ConfigLoadFails(t *testing.T) {
	clearVioletEnv(t)
	path := core.PathJoin(t.TempDir(), "violet.toml")
	if result := core.WriteFile(path, []byte("socket /no/equals/sign\n"), 0o644); !result.OK {
		t.Fatalf("write malformed config: %v", result.Value)
	}

	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"--config", path}, stdout, stderr)
	if code != 1 {
		t.Fatalf("exit code = %d, want 1; stderr=%q", code, stderr.String())
	}
	if !core.Contains(stderr.String(), "expected key = value") {
		t.Fatalf("stderr = %q, want config syntax error", stderr.String())
	}
}

// TestMain_RunCommand_Good_EnvSocketUsedWhenConfigEmpty covers the
// VIOLET_SOCKET_PATH fallback branch: with no config socket and no -socket
// flag, the environment socket is adopted, then the daemon starts and stops on
// the pre-cancelled context.
func TestMain_RunCommand_Good_EnvSocketUsedWhenConfigEmpty(t *testing.T) {
	clearVioletEnv(t)
	socketPath := shortVioletSocket(t)
	setCoreEnv(t, "VIOLET_SOCKET_PATH", socketPath)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(ctx, nil, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q", code, stderr.String())
	}
}

// TestMain_RunCommand_Bad_DefaultSocketUnresolvable covers the branch where no
// socket is configured anywhere and daemon.DefaultSocketPath fails. With HOME
// and XDG_CONFIG_HOME blanked, the default config path is empty (no file read)
// and DefaultSocketPath cannot resolve the home directory on darwin, so
// runCommand returns 1.
func TestMain_RunCommand_Bad_DefaultSocketUnresolvable(t *testing.T) {
	clearVioletEnv(t)
	setCoreEnv(t, "HOME", "")
	setCoreEnv(t, "XDG_CONFIG_HOME", "")
	setCoreEnv(t, "XDG_RUNTIME_DIR", "")

	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), nil, stdout, stderr)
	if code != 1 {
		t.Fatalf("exit code = %d, want 1; stderr=%q", code, stderr.String())
	}
	if !core.Contains(stderr.String(), "violet:") {
		t.Fatalf("stderr = %q, want violet error prefix", stderr.String())
	}
}

// TestMain_LoadRuntimeConfig_Good_DefaultPathMissingIsSwallowed exercises the
// default-config-path branch of loadRuntimeConfig (explicitPath == ""). The
// resolved default file does not exist; because the path was implicit, the
// not-exist error is swallowed and an empty config is returned without error.
func TestMain_LoadRuntimeConfig_Good_DefaultPathMissingIsSwallowed(t *testing.T) {
	clearVioletEnv(t)
	// Point the default config path at a guaranteed-missing file under a temp
	// XDG_CONFIG_HOME so the implicit read fails with not-exist and is ignored.
	setCoreEnv(t, "XDG_CONFIG_HOME", t.TempDir())

	cfg, err := loadRuntimeConfig("")
	if err != nil {
		t.Fatalf("loadRuntimeConfig(\"\") error = %v, want nil (missing default swallowed)", err)
	}
	if cfg.SocketPath != "" {
		t.Fatalf("SocketPath = %q, want empty", cfg.SocketPath)
	}
}

// TestMain_LoadRuntimeConfig_Bad_ExplicitPathMissing covers the sibling branch:
// an explicit path that does not exist must surface the error rather than being
// swallowed.
func TestMain_LoadRuntimeConfig_Bad_ExplicitPathMissing(t *testing.T) {
	clearVioletEnv(t)
	missing := core.PathJoin(t.TempDir(), "does-not-exist.toml")

	_, err := loadRuntimeConfig(missing)
	if err == nil {
		t.Fatal("loadRuntimeConfig(missing explicit path) = nil, want error")
	}
}

// TestMain_DefaultConfigPath_Good_AllBranches drives defaultConfigPath through
// its three outcomes: XDG_CONFIG_HOME set, HOME set (XDG unset), and neither
// set (home unresolvable -> empty string).
func TestMain_DefaultConfigPath_Good_AllBranches(t *testing.T) {
	clearVioletEnv(t)

	// XDG_CONFIG_HOME wins when present.
	setCoreEnv(t, "XDG_CONFIG_HOME", "/xdg/cfg")
	if got := defaultConfigPath(); got != core.PathJoin("/xdg/cfg", "ofm", "violet.toml") {
		t.Fatalf("defaultConfigPath() with XDG = %q", got)
	}

	// With XDG cleared, HOME drives the path.
	setCoreEnv(t, "XDG_CONFIG_HOME", "")
	setCoreEnv(t, "HOME", "/home/violet")
	if got := defaultConfigPath(); got != core.PathJoin("/home/violet", ".config", "ofm", "violet.toml") {
		t.Fatalf("defaultConfigPath() with HOME = %q", got)
	}

	// With neither set, the home lookup fails and the path is empty.
	setCoreEnv(t, "HOME", "")
	if got := defaultConfigPath(); got != "" {
		t.Fatalf("defaultConfigPath() with no HOME/XDG = %q, want empty", got)
	}
}

// TestMain_ReadConfigFile_Bad_OpenFailure covers the open-failure path: a
// missing file makes core.Open fail and readConfigFile returns the wrapped
// error from commandResultError.
func TestMain_ReadConfigFile_Bad_OpenFailure(t *testing.T) {
	missing := core.PathJoin(t.TempDir(), "absent.toml")
	err := readConfigFile(missing, &runtimeConfig{Models: make(map[string]string)})
	if err == nil {
		t.Fatal("readConfigFile(missing) = nil, want open error")
	}
	if !core.IsNotExist(err) {
		t.Fatalf("readConfigFile(missing) error = %v, want not-exist", err)
	}
}

// TestMain_ReadConfigFile_Good_SectionsAndAliases covers the [models] section
// branch and the top-level key aliases (socket_path, model_path) that the
// existing happy-path test does not reach.
func TestMain_ReadConfigFile_Good_SectionsAndAliases(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "violet.toml")
	if result := core.WriteFile(path, []byte(`
socket_path = "/tmp/alias.sock"
model_path = "/models/alias-default"

[models]
embed = "/models/embed"

[unknown]
ignored = "/models/ignored"
`), 0o644); !result.OK {
		t.Fatalf("write config: %v", result.Value)
	}

	cfg := runtimeConfig{Models: make(map[string]string)}
	if err := readConfigFile(path, &cfg); err != nil {
		t.Fatalf("readConfigFile() error = %v", err)
	}
	if cfg.SocketPath != "/tmp/alias.sock" {
		t.Fatalf("socket_path alias = %q", cfg.SocketPath)
	}
	if cfg.Models["default"] != "/models/alias-default" {
		t.Fatalf("model_path alias = %q", cfg.Models["default"])
	}
	if cfg.Models["embed"] != "/models/embed" {
		t.Fatalf("[models] embed = %q", cfg.Models["embed"])
	}
	if _, ok := cfg.Models["ignored"]; ok {
		t.Fatalf("unknown section leaked a model: %+v", cfg.Models)
	}
}

// TestMain_ReadConfigFile_Bad_ValueParseError covers the wrapped value-parse
// error branch (line/path context) when a value fails to parse mid-file.
func TestMain_ReadConfigFile_Bad_ValueParseError(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "violet.toml")
	if result := core.WriteFile(path, []byte("model = 'unterminated\n"), 0o644); !result.OK {
		t.Fatalf("write config: %v", result.Value)
	}

	err := readConfigFile(path, &runtimeConfig{Models: make(map[string]string)})
	if err == nil {
		t.Fatal("readConfigFile(bad value) = nil, want parse error")
	}
	if !core.Contains(err.Error(), "unterminated single-quoted value") {
		t.Fatalf("error = %v, want unterminated single-quote context", err)
	}
}

// TestMain_ReadConfigFile_Bad_ScannerError covers the scanner.Err branch. A
// single line longer than bufio's 64KiB token limit (and with no newline) makes
// the scanner stop with ErrTooLong, which readConfigFile must surface.
func TestMain_ReadConfigFile_Bad_ScannerError(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "violet.toml")
	huge := strings.Repeat("a", 70*1024)
	if result := core.WriteFile(path, []byte(huge), 0o644); !result.OK {
		t.Fatalf("write oversized config line: %v", result.Value)
	}

	err := readConfigFile(path, &runtimeConfig{Models: make(map[string]string)})
	if err == nil {
		t.Fatal("readConfigFile(oversized line) = nil, want scanner error")
	}
}

// TestMain_ParseConfigValue_Good_EmptyAndBare covers the empty-input fast path
// and the bare (unquoted) trim path.
func TestMain_ParseConfigValue_Good_EmptyAndBare(t *testing.T) {
	value, err := parseConfigValue("")
	if err != nil || value != "" {
		t.Fatalf("parseConfigValue(empty) = (%q, %v)", value, err)
	}
	value, err = parseConfigValue("  bare-value  ")
	if err != nil {
		t.Fatalf("parseConfigValue(bare) error = %v", err)
	}
	if value != "bare-value" {
		t.Fatalf("parseConfigValue(bare) = %q, want trimmed", value)
	}
}

// TestMain_ParseConfigValue_Bad_DoubleQuoteUnquote covers the strconv.Unquote
// failure branch for a malformed double-quoted value.
func TestMain_ParseConfigValue_Bad_DoubleQuoteUnquote(t *testing.T) {
	if _, err := parseConfigValue(`"\x"`); err == nil {
		t.Fatal("parseConfigValue(bad escape) = nil, want unquote error")
	}
}

// TestMain_StripConfigComment_Good_QuotesAndEscapes covers the remaining
// stripConfigComment branches: an escaped quote inside a double-quoted span, a
// '#' that survives because it is inside quotes, and a single-quoted span.
func TestMain_StripConfigComment_Good_QuotesAndEscapes(t *testing.T) {
	cases := []struct{ in, want string }{
		// Escaped quote keeps the string open so the later # is data, not a comment.
		{`x = "a \" # b"`, `x = "a \" # b"`},
		// Single-quoted span protects its #.
		{`x = 'a # b' # tail`, `x = 'a # b' `},
		// A bare # outside any quote starts a comment.
		{`x = 1 # gone`, `x = 1 `},
		// No comment at all is returned verbatim.
		{`x = plain`, `x = plain`},
	}
	for _, tc := range cases {
		if got := stripConfigComment(tc.in); got != tc.want {
			t.Fatalf("stripConfigComment(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

// TestMain_ApplyModelEnv_Good_NilMapInitialised covers the nil-map guard: when
// the config has no map yet, applyModelEnv allocates one before applying the
// environment fallback.
func TestMain_ApplyModelEnv_Good_NilMapInitialised(t *testing.T) {
	clearVioletEnv(t)
	setCoreEnv(t, "VIOLET_MODEL_PATH", "/models/from-env")

	cfg := &runtimeConfig{} // Models intentionally nil
	applyModelEnv(cfg, "default", "VIOLET_MODEL_PATH")
	if cfg.Models == nil {
		t.Fatal("applyModelEnv left Models nil")
	}
	if cfg.Models["default"] != "/models/from-env" {
		t.Fatalf("default model = %q, want env value", cfg.Models["default"])
	}
}

// TestMain_CommandResultError_Good_RealError covers the branch where the Result
// carries a genuine error value, which is returned as-is.
func TestMain_CommandResultError_Good_RealError(t *testing.T) {
	sentinel := core.NewError("disk gone")
	err := commandResultError(core.Result{OK: false, Value: sentinel})
	if err == nil || !core.Contains(err.Error(), "disk gone") {
		t.Fatalf("commandResultError(real) = %v, want wrapped sentinel", err)
	}
}

// clearVioletEnv blanks every environment variable that loadRuntimeConfig and
// the default-path helpers consult, so each test starts from a known-empty
// environment regardless of the host shell. Restoration is handled by
// setCoreEnv's cleanup.
func clearVioletEnv(t *testing.T) {
	t.Helper()
	for _, key := range []string{
		"VIOLET_SOCKET_PATH",
		"VIOLET_MODEL_PATH",
		"VIOLET_EMBED_MODEL_PATH",
		"VIOLET_SCORE_MODEL_PATH",
		"VIOLET_GENERATE_MODEL_PATH",
		"XDG_CONFIG_HOME",
	} {
		setCoreEnv(t, key, "")
	}
}

// shortVioletSocket returns a socket path under /tmp short enough to stay inside
// the platform's sun_path limit, with cleanup registered.
func shortVioletSocket(t *testing.T) string {
	t.Helper()
	dir := core.MkdirTemp("/tmp", "violet-cv-*")
	if !dir.OK {
		t.Fatalf("create short temp dir: %v", dir.Value)
	}
	path := dir.Value.(string)
	t.Cleanup(func() { core.RemoveAll(path) })
	return core.PathJoin(path, "violet.sock")
}
