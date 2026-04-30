// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"bufio"
	"context"
	"flag"
	"io"
	"os/signal"
	"strconv"
	"syscall"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/daemon"
)

var version = daemon.DefaultVersion

type runtimeConfig struct {
	SocketPath string
	Models     map[string]string
}

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	core.Exit(runCommand(ctx, core.Args()[1:], core.Stdout(), core.Stderr()))
}

func runCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("violet", flag.ContinueOnError)
	fs.SetOutput(stderr)

	configPath := fs.String("config", "", "path to violet.toml")
	socketPath := fs.String("socket", "", "unix socket path")
	showVersion := fs.Bool("version", false, "print version and exit")
	fs.Usage = func() {
		printUsage(stdout, fs)
	}

	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 0 {
		core.Print(stderr, "violet: unexpected argument %q", fs.Arg(0))
		printUsage(stderr, fs)
		return 2
	}
	if *showVersion {
		core.Print(stdout, "violet %s", version)
		return 0
	}

	cfg, err := loadRuntimeConfig(*configPath)
	if err != nil {
		core.Print(stderr, "violet: %v", err)
		return 1
	}

	if envSocket := core.Getenv("VIOLET_SOCKET_PATH"); envSocket != "" && cfg.SocketPath == "" {
		cfg.SocketPath = envSocket
	}
	if *socketPath != "" {
		cfg.SocketPath = *socketPath
	}
	if cfg.SocketPath == "" {
		cfg.SocketPath, err = daemon.DefaultSocketPath()
		if err != nil {
			core.Print(stderr, "violet: %v", err)
			return 1
		}
	}

	// Configuration may name multiple model paths because Violet is a single
	// local sidecar for all actions. Model hot-swap and streaming output are
	// intentionally follow-up work; restart the daemon after config changes.
	srv := daemon.NewServer(daemon.ServerConfig{
		SocketPath: cfg.SocketPath,
		Registry:   daemon.NewRegistry(daemon.DaemonName, version),
		ModelPaths: cfg.Models,
	})

	if err := srv.ListenAndServe(ctx); err != nil {
		core.Print(stderr, "violet: %v", err)
		return 1
	}
	return 0
}

func printUsage(w io.Writer, fs *flag.FlagSet) {
	core.WriteString(w, "Usage: violet [flags]\n")
	core.WriteString(w, "\n")
	core.WriteString(w, "Runs the Violet local-native inference sidecar over a Unix domain socket.\n")
	core.WriteString(w, "\n")
	core.WriteString(w, "Flags:\n")
	fs.VisitAll(func(f *flag.Flag) {
		if f.DefValue == "" {
			core.WriteString(w, core.Sprintf("  -%s\n\t%s\n", f.Name, f.Usage))
			return
		}
		core.WriteString(w, core.Sprintf("  -%s\n\t%s (default %q)\n", f.Name, f.Usage, f.DefValue))
	})
}

func loadRuntimeConfig(explicitPath string) (runtimeConfig, error) {
	cfg := runtimeConfig{Models: make(map[string]string)}

	configPath := explicitPath
	if configPath == "" {
		configPath = defaultConfigPath()
	}
	if configPath != "" {
		if err := readConfigFile(configPath, &cfg); err != nil {
			if explicitPath != "" || !core.IsNotExist(err) {
				return cfg, err
			}
		}
	}

	applyEnvModelFallbacks(&cfg)
	return cfg, nil
}

func defaultConfigPath() string {
	if configHome := core.Getenv("XDG_CONFIG_HOME"); configHome != "" {
		return core.PathJoin(configHome, "ofm", "violet.toml")
	}
	home := core.UserHomeDir()
	if !home.OK {
		return ""
	}
	return core.PathJoin(home.Value.(string), ".config", "ofm", "violet.toml")
}

func readConfigFile(path string, cfg *runtimeConfig) error {
	opened := core.Open(path)
	if !opened.OK {
		return commandResultError(opened)
	}
	file := opened.Value.(*core.OSFile)
	defer file.Close()

	section := ""
	scanner := bufio.NewScanner(file)
	lineNumber := 0
	for scanner.Scan() {
		lineNumber++
		line := core.Trim(stripConfigComment(scanner.Text()))
		if line == "" {
			continue
		}

		if core.HasPrefix(line, "[") && core.HasSuffix(line, "]") {
			section = core.Trim(core.TrimSuffix(core.TrimPrefix(line, "["), "]"))
			continue
		}

		parts := core.SplitN(line, "=", 2)
		if len(parts) != 2 {
			return core.Errorf("%s:%d: expected key = value", path, lineNumber)
		}

		key, rawValue := parts[0], parts[1]
		value, err := parseConfigValue(core.Trim(rawValue))
		if err != nil {
			return core.Errorf("%s:%d: %w", path, lineNumber, err)
		}
		key = core.Trim(key)

		switch section {
		case "models":
			if key != "" && value != "" {
				cfg.Models[key] = value
			}
		case "":
			switch key {
			case "socket", "socket_path":
				cfg.SocketPath = value
			case "model", "model_path", "default_model":
				if value != "" {
					cfg.Models["default"] = value
				}
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return err
	}
	return nil
}

func parseConfigValue(raw string) (string, error) {
	if raw == "" {
		return "", nil
	}
	if core.HasPrefix(raw, "\"") {
		value, err := strconv.Unquote(raw)
		if err != nil {
			return "", err
		}
		return value, nil
	}
	if core.HasPrefix(raw, "'") && core.HasSuffix(raw, "'") {
		return core.TrimSuffix(core.TrimPrefix(raw, "'"), "'"), nil
	}
	return core.Trim(raw), nil
}

func stripConfigComment(line string) string {
	var quote rune
	escaped := false
	for i, r := range line {
		if escaped {
			escaped = false
			continue
		}
		if quote == '"' && r == '\\' {
			escaped = true
			continue
		}
		if quote != 0 {
			if r == quote {
				quote = 0
			}
			continue
		}
		if r == '"' || r == '\'' {
			quote = r
			continue
		}
		if r == '#' {
			return line[:i]
		}
	}
	return line
}

func applyEnvModelFallbacks(cfg *runtimeConfig) {
	applyModelEnv(cfg, "default", "VIOLET_MODEL_PATH")
	applyModelEnv(cfg, "embed", "VIOLET_EMBED_MODEL_PATH")
	applyModelEnv(cfg, "score", "VIOLET_SCORE_MODEL_PATH")
	applyModelEnv(cfg, "generate", "VIOLET_GENERATE_MODEL_PATH")
}

func applyModelEnv(cfg *runtimeConfig, name, envName string) {
	if cfg.Models == nil {
		cfg.Models = make(map[string]string)
	}
	if cfg.Models[name] != "" {
		return
	}
	if value := core.Getenv(envName); value != "" {
		cfg.Models[name] = value
	}
}

func commandResultError(result core.Result) error {
	if err, ok := result.Value.(error); ok {
		return err
	}
	return core.NewError("violet command operation failed")
}
