// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"flag"
	"io"
	"os/signal"
	"syscall"

	core "dappco.re/go"
	mlx "dappco.re/go/mlx"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	core.Exit(runCommand(ctx, core.Args()[1:], core.Stdout(), core.Stderr()))
}

func runCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		printUsage(stdout)
		return 0
	}
	switch args[0] {
	case "bench":
		return runBenchCommand(ctx, args[1:], stdout, stderr)
	case "pack":
		return runPackCommand(ctx, args[1:], stdout, stderr)
	case "-h", "--help", "help":
		printUsage(stdout)
		return 0
	default:
		core.Print(stderr, "go-mlx: unknown command %q", args[0])
		printUsage(stderr)
		return 2
	}
}

var (
	loadBenchModel = mlx.LoadModel
	runBenchReport = mlx.RunFastEvalBench
)

func runBenchCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	cfg := mlx.DefaultFastEvalConfig()
	fs := flag.NewFlagSet("go-mlx bench", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON report")
	prompt := fs.String("prompt", cfg.Prompt, "baseline benchmark prompt")
	cachePrompt := fs.String("cache-prompt", "", "stable prompt used for prompt-cache and KV restore checks")
	maxTokens := fs.Int("max-tokens", cfg.MaxTokens, "generated tokens per pass")
	runs := fs.Int("runs", cfg.Runs, "baseline generation passes")
	contextLen := fs.Int("context", 0, "override context length")
	device := fs.String("device", "", "execution device: gpu or cpu")
	noCache := fs.Bool("no-cache", false, "skip prompt-cache warm/hit check")
	noRestore := fs.Bool("no-restore", false, "skip KV restore latency check")
	noBundle := fs.Bool("no-bundle", false, "skip state-bundle round trip check")
	noProbes := fs.Bool("no-probes", false, "skip probe overhead check")
	fs.Usage = func() {
		core.WriteString(stderr, "Usage: go-mlx bench [flags] <model-path>\n")
		fs.VisitAll(func(f *flag.Flag) {
			if f.DefValue == "" {
				core.WriteString(stderr, core.Sprintf("  -%s\n\t%s\n", f.Name, f.Usage))
				return
			}
			core.WriteString(stderr, core.Sprintf("  -%s\n\t%s (default %q)\n", f.Name, f.Usage, f.DefValue))
		})
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 1 {
		core.WriteString(stderr, "go-mlx bench: expected exactly one model path\n")
		fs.Usage()
		return 2
	}

	modelPath := fs.Arg(0)
	cfg.Model = core.PathBase(modelPath)
	cfg.ModelPath = modelPath
	cfg.Prompt = *prompt
	cfg.CachePrompt = *cachePrompt
	cfg.MaxTokens = *maxTokens
	cfg.Runs = *runs
	cfg.IncludePromptCache = !*noCache
	cfg.IncludeKVRestore = !*noRestore
	cfg.IncludeStateBundleRoundTrip = !*noBundle
	cfg.IncludeProbeOverhead = !*noProbes

	loadOptions := []mlx.LoadOption{}
	if *contextLen > 0 {
		loadOptions = append(loadOptions, mlx.WithContextLength(*contextLen))
	}
	if *device != "" {
		loadOptions = append(loadOptions, mlx.WithDevice(*device))
	}
	model, err := loadBenchModel(modelPath, loadOptions...)
	if err != nil {
		core.Print(stderr, "go-mlx bench: load model: %v", err)
		return 1
	}
	defer model.Close()

	report, err := runBenchReport(ctx, model, cfg)
	if err != nil {
		core.Print(stderr, "go-mlx bench: %v", err)
		return 1
	}
	if *jsonOut {
		data := core.JSONMarshalIndent(report, "", "  ")
		if !data.OK {
			core.Print(stderr, "go-mlx bench: marshal report failed")
			return 1
		}
		core.WriteString(stdout, string(data.Value.([]byte)))
		core.WriteString(stdout, "\n")
		return 0
	}
	printBenchSummary(stdout, report)
	return 0
}

func printBenchSummary(stdout io.Writer, report *mlx.FastEvalReport) {
	if report == nil {
		return
	}
	core.WriteString(stdout, core.Sprintf("fast eval: %s\n", report.ModelPath))
	core.WriteString(stdout, core.Sprintf("  prefill: %.1f tok/s, decode: %.1f tok/s\n", report.Generation.PrefillTokensPerSec, report.Generation.DecodeTokensPerSec))
	core.WriteString(stdout, core.Sprintf("  peak memory: %d MB, active memory: %d MB\n", report.Generation.PeakMemoryBytes/1024/1024, report.Generation.ActiveMemoryBytes/1024/1024))
	if report.PromptCache.Attempted {
		core.WriteString(stdout, core.Sprintf("  prompt cache: %.0f%% hit rate (%d hit, %d miss)\n", report.PromptCache.HitRate*100, report.PromptCache.Hits, report.PromptCache.Misses))
	}
	if report.KVRestore.Attempted {
		core.WriteString(stdout, core.Sprintf("  KV restore: %s\n", report.KVRestore.Duration))
	}
	if report.StateBundle.Attempted {
		core.WriteString(stdout, core.Sprintf("  state bundle: %d bytes, %s round trip\n", report.StateBundle.Bytes, report.StateBundle.Duration))
	}
	if report.Probes.Attempted {
		core.WriteString(stdout, core.Sprintf("  probes: %d events, %.1f%% overhead\n", report.Probes.EventCount, report.Probes.OverheadRatio*100))
	}
}

func runPackCommand(_ context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("go-mlx pack", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON report")
	expectedQuant := fs.Int("quantization", 0, "required quantization bits")
	maxContext := fs.Int("max-context", 0, "maximum allowed context length")
	fs.Usage = func() {
		core.WriteString(stderr, "Usage: go-mlx pack [flags] <model-path>\n")
		fs.VisitAll(func(f *flag.Flag) {
			if f.DefValue == "" {
				core.WriteString(stderr, core.Sprintf("  -%s\n\t%s\n", f.Name, f.Usage))
				return
			}
			core.WriteString(stderr, core.Sprintf("  -%s\n\t%s (default %q)\n", f.Name, f.Usage, f.DefValue))
		})
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 1 {
		core.WriteString(stderr, "go-mlx pack: expected exactly one model path\n")
		fs.Usage()
		return 2
	}

	options := []mlx.ModelPackOption{}
	if *expectedQuant > 0 {
		options = append(options, mlx.WithPackQuantization(*expectedQuant))
	}
	if *maxContext > 0 {
		options = append(options, mlx.WithPackMaxContextLength(*maxContext))
	}
	pack, err := mlx.InspectModelPack(fs.Arg(0), options...)
	if err != nil {
		core.Print(stderr, "go-mlx pack: %v", err)
		return 1
	}
	if *jsonOut {
		data := core.JSONMarshal(pack)
		if !data.OK {
			core.Print(stderr, "go-mlx pack: marshal report failed")
			return 1
		}
		core.WriteString(stdout, string(data.Value.([]byte)))
		core.WriteString(stdout, "\n")
		if !pack.Valid() {
			return 1
		}
		return 0
	}
	if !pack.Valid() {
		printPackIssues(stderr, pack)
		return 1
	}
	core.WriteString(stdout, core.Sprintf(
		"valid model pack: %s (%s, %s, quant=%d, context=%d)\n",
		pack.Root,
		pack.Architecture,
		pack.Format,
		pack.QuantBits,
		pack.ContextLength,
	))
	return 0
}

func printPackIssues(stderr io.Writer, pack mlx.ModelPack) {
	core.WriteString(stderr, "go-mlx pack: invalid model pack\n")
	for _, issue := range pack.Issues {
		if issue.Severity != mlx.ModelPackIssueError {
			continue
		}
		core.WriteString(stderr, core.Sprintf("  %s: %s\n", issue.Code, issue.Message))
	}
}

func printUsage(w io.Writer) {
	core.WriteString(w, "Usage: go-mlx <command> [flags]\n")
	core.WriteString(w, "\n")
	core.WriteString(w, "Commands:\n")
	core.WriteString(w, "  bench   run fast local eval/benchmark harness\n")
	core.WriteString(w, "  pack    validate a local native model pack\n")
}
