package main

import (
	"bufio"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"

	"dappco.re/go/mlx/pkg/daemon"
)

var version = daemon.DefaultVersion

type runtimeConfig struct {
	SocketPath string
	Models     map[string]string
}

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	os.Exit(runCommand(ctx, os.Args[1:], os.Stdout, os.Stderr))
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
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "violet: unexpected argument %q\n", fs.Arg(0))
		printUsage(stderr, fs)
		return 2
	}
	if *showVersion {
		fmt.Fprintf(stdout, "violet %s\n", version)
		return 0
	}

	cfg, err := loadRuntimeConfig(*configPath)
	if err != nil {
		fmt.Fprintf(stderr, "violet: %v\n", err)
		return 1
	}

	if envSocket := os.Getenv("VIOLET_SOCKET_PATH"); envSocket != "" && cfg.SocketPath == "" {
		cfg.SocketPath = envSocket
	}
	if *socketPath != "" {
		cfg.SocketPath = *socketPath
	}
	if cfg.SocketPath == "" {
		cfg.SocketPath, err = daemon.DefaultSocketPath()
		if err != nil {
			fmt.Fprintf(stderr, "violet: %v\n", err)
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
		fmt.Fprintf(stderr, "violet: %v\n", err)
		return 1
	}
	return 0
}

func printUsage(w io.Writer, fs *flag.FlagSet) {
	fmt.Fprintln(w, "Usage: violet [flags]")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Runs the Violet local-native inference sidecar over a Unix domain socket.")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Flags:")
	fs.VisitAll(func(f *flag.Flag) {
		if f.DefValue == "" {
			fmt.Fprintf(w, "  -%s\n\t%s\n", f.Name, f.Usage)
			return
		}
		fmt.Fprintf(w, "  -%s\n\t%s (default %q)\n", f.Name, f.Usage, f.DefValue)
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
			if explicitPath != "" || !errors.Is(err, os.ErrNotExist) {
				return cfg, err
			}
		}
	}

	applyEnvModelFallbacks(&cfg)
	return cfg, nil
}

func defaultConfigPath() string {
	if configHome := os.Getenv("XDG_CONFIG_HOME"); configHome != "" {
		return filepath.Join(configHome, "ofm", "violet.toml")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".config", "ofm", "violet.toml")
}

func readConfigFile(path string, cfg *runtimeConfig) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	section := ""
	scanner := bufio.NewScanner(file)
	lineNumber := 0
	for scanner.Scan() {
		lineNumber++
		line := strings.TrimSpace(stripConfigComment(scanner.Text()))
		if line == "" {
			continue
		}

		if strings.HasPrefix(line, "[") && strings.HasSuffix(line, "]") {
			section = strings.TrimSpace(strings.TrimSuffix(strings.TrimPrefix(line, "["), "]"))
			continue
		}

		key, rawValue, ok := strings.Cut(line, "=")
		if !ok {
			return fmt.Errorf("%s:%d: expected key = value", path, lineNumber)
		}

		value, err := parseConfigValue(strings.TrimSpace(rawValue))
		if err != nil {
			return fmt.Errorf("%s:%d: %w", path, lineNumber, err)
		}
		key = strings.TrimSpace(key)

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
	if strings.HasPrefix(raw, "\"") {
		value, err := strconv.Unquote(raw)
		if err != nil {
			return "", err
		}
		return value, nil
	}
	if strings.HasPrefix(raw, "'") && strings.HasSuffix(raw, "'") {
		return strings.TrimSuffix(strings.TrimPrefix(raw, "'"), "'"), nil
	}
	return strings.TrimSpace(raw), nil
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
	if value := os.Getenv(envName); value != "" {
		cfg.Models[name] = value
	}
}
