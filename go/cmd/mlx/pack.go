// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"flag"
	"io"

	core "dappco.re/go"
	"dappco.re/go/mlx/model"
	pack "dappco.re/go/inference/modelpack"
)

func runPackCommand(_ context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("pack"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON report")
	expectedQuant := fs.Int("quantization", 0, "required quantization bits")
	maxContext := fs.Int("max-context", 0, "maximum allowed context length")
	fs.Usage = func() {
		name := cliName()
		core.WriteString(stderr, core.Sprintf("Usage: %s pack [flags] <model-path>\n", name))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Validate a model pack on disk without loading weights — reads the\n")
		core.WriteString(stderr, "config + tokenizer + safetensors index, reports architecture, layer\n")
		core.WriteString(stderr, "count, embedding size, quantization, context length, and any sentinel\n")
		core.WriteString(stderr, "validation errors. Cheap (no GPU work) — run before serve/bench to\n")
		core.WriteString(stderr, "catch a corrupt download or wrong architecture before allocating VRAM.\n")
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Flags:\n")
		fs.VisitAll(func(f *flag.Flag) {
			if f.DefValue == "" {
				core.WriteString(stderr, core.Sprintf("  -%s\n\t%s\n", f.Name, f.Usage))
				return
			}
			core.WriteString(stderr, core.Sprintf("  -%s\n\t%s (default %q)\n", f.Name, f.Usage, f.DefValue))
		})
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Examples:\n")
		core.WriteString(stderr, core.Sprintf("  %s pack ~/models/lemer-lite\n", name))
		core.WriteString(stderr, core.Sprintf("    # validate + print summary table\n"))
		core.WriteString(stderr, core.Sprintf("  %s pack -json ~/models/lemer-lite\n", name))
		core.WriteString(stderr, core.Sprintf("    # machine-readable output (for CI / scripts)\n"))
		core.WriteString(stderr, core.Sprintf("  %s pack -quantization 4 ~/models/lemer-lite-q4\n", name))
		core.WriteString(stderr, core.Sprintf("    # require q4 (fails non-zero if not)\n"))
		core.WriteString(stderr, core.Sprintf("  %s pack -max-context 8192 ~/models/lemer-lite\n", name))
		core.WriteString(stderr, core.Sprintf("    # require context <= 8192 (fails non-zero if exceeds)\n"))
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 1 {
		core.WriteString(stderr, core.Sprintf("%s pack: expected exactly one model path\n", cliName()))
		fs.Usage()
		return 2
	}

	options := []pack.ModelPackOption{}
	if *expectedQuant > 0 {
		options = append(options, pack.WithPackQuantization(*expectedQuant))
	}
	if *maxContext > 0 {
		options = append(options, pack.WithPackMaxContextLength(*maxContext))
	}
	pack, err := model.Inspect(fs.Arg(0), options...)
	if err != nil {
		core.Print(stderr, "%s pack: %v", cliName(), err)
		return 1
	}
	if *jsonOut {
		data := core.JSONMarshal(pack)
		if !data.OK {
			core.Print(stderr, "%s pack: marshal report failed", cliName())
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

func printPackIssues(stderr io.Writer, p pack.ModelPack) {
	core.WriteString(stderr, core.Sprintf("%s pack: invalid model pack\n", cliName()))
	for _, issue := range p.Issues {
		if issue.Severity != pack.ModelPackIssueError {
			continue
		}
		core.WriteString(stderr, core.Sprintf("  %s: %s\n", issue.Code, issue.Message))
	}
}
