// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"flag"
	"io"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// runFuseCommand folds a trained LoRA adapter into a dense base model and writes
// the fused model dir — the P-phase on-disk fuse. A thin CLI over
// metal.FuseModelDir: the serving artefact afterwards needs no adapter, since
// generation uses the merged dense weights directly.
func runFuseCommand(_ context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("fuse"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	base := fs.String("base", "", "dense base model directory (bf16/fp32)")
	adapter := fs.String("adapter", "", "trained LoRA adapter directory (adapter.safetensors + adapter_config.json)")
	out := fs.String("out", "", "output directory for the fused model")
	fs.Usage = func() {
		name := cliName()
		core.WriteString(stderr, core.Sprintf("Usage: %s fuse -base <dir> -adapter <dir> -out <dir>\n", name))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Fold a trained LoRA adapter into a dense base model and write the fused\n")
		core.WriteString(stderr, "dense model to -out (the P-phase on-disk fuse). Each targeted weight W\n")
		core.WriteString(stderr, "becomes W+(B*A)*scale (scale = adapter alpha/rank); the base's config +\n")
		core.WriteString(stderr, "tokenizer are carried across. The base must be dense — a quantized base\n")
		core.WriteString(stderr, "is refused (dequantize-merge-then-requantize is a separate path).\n")
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Flags:\n")
		fs.VisitAll(func(f *flag.Flag) {
			core.WriteString(stderr, core.Sprintf("  -%s\n\t%s\n", f.Name, f.Usage))
		})
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Example:\n")
		core.WriteString(stderr, core.Sprintf("  %s fuse -base ~/models/lemer-lite -adapter ~/lora/lek2 -out ~/models/lemer-lite-lek2\n", name))
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if *base == "" || *adapter == "" || *out == "" {
		core.WriteString(stderr, core.Sprintf("%s fuse: -base, -adapter and -out are all required\n", cliName()))
		fs.Usage()
		return 2
	}

	fused, err := metal.FuseModelDir(*base, *adapter, *out)
	if err != nil {
		core.Print(stderr, "%s fuse: %v", cliName(), err)
		return 1
	}
	core.WriteString(stdout, core.Sprintf("fused %d layer(s) into %s\n", len(fused), *out))
	return 0
}
