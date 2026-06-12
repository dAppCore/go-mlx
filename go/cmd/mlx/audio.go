// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"flag"
	"io"

	core "dappco.re/go"
	"dappco.re/go/mlx/chat"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/pkg/metal/model/gemma4"
	gemma4chat "dappco.re/go/mlx/pkg/metal/model/gemma4/chat"
)

// runAudioCommand answers a prompt about a WAV clip through the Gemma 4
// audio lane (Mantis #1839): waveform → log-mel front-end → Conformer tower
// → soft tokens spliced over the prompt's audio placeholders → greedy
// decode. Self-contained like the diffuse verb — the serve's OpenAI
// input_audio surface builds on the same seams later.
func runAudioCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("audio", flag.ContinueOnError)
	fs.SetOutput(stderr)
	wavPath := fs.String("audio", "", "16 kHz mono WAV clip (PCM16 or float32)")
	prompt := fs.String("prompt", "What is said in this recording?", "question about the clip")
	maxTokens := fs.Int("max-tokens", 256, "response length bound")
	chatFlag := fs.Bool("chat", true, "format with the model chat template")
	fs.Usage = func() {
		core.WriteString(stderr, "Usage: lthn-mlx audio -audio clip.wav [flags] <model-path>\n\n")
		core.WriteString(stderr, "Answer a prompt about an audio clip (Gemma 4 E2B/E4B audio tower).\n\n")
		core.WriteString(stderr, "Flags:\n")
		fs.PrintDefaults()
		core.WriteString(stderr, "\nExample:\n")
		core.WriteString(stderr, "    lthn-mlx audio -audio speech.wav -prompt 'Transcribe this.' <model>\n")
	}
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 1 || *wavPath == "" {
		fs.Usage()
		return 2
	}

	m, err := gemma4.LoadGemma4(fs.Arg(0))
	if err != nil {
		core.Print(stderr, "%s audio: load: %v", cliName(), err)
		return 1
	}
	defer m.CloseModel()
	if m.AudioEncoder == nil {
		core.Print(stderr, "%s audio: this checkpoint has no audio tower — use a Gemma 4 E2B/E4B snapshot", cliName())
		return 1
	}
	if m.AudioFeatures == nil {
		core.Print(stderr, "%s audio: model ships no processor_config.json audio front-end", cliName())
		return 1
	}
	if m.Cfg == nil || m.Cfg.AudioTokenID == 0 {
		core.Print(stderr, "%s audio: model config declares no audio_token_id", cliName())
		return 1
	}

	samples, err := readWAVMono(*wavPath, m.AudioFeatures.SamplingRate())
	if err != nil {
		core.Print(stderr, "%s audio: %v", cliName(), err)
		return 1
	}
	mel, softTokens, err := m.AudioInputFeatures(samples)
	if err != nil {
		core.Print(stderr, "%s audio: features: %v", cliName(), err)
		return 1
	}
	defer metal.Free(mel)

	// The HF processor convention: BOA + AudioToken×softTokens + EOA ahead
	// of the question text, inside the user turn.
	audioBlock := gemma4.Gemma4BOAToken
	for range softTokens {
		audioBlock += gemma4.Gemma4AudioToken
	}
	audioBlock += gemma4.Gemma4EOAToken
	content := audioBlock + "\n" + *prompt
	formatted := content
	if *chatFlag {
		formatted = gemma4chat.Format([]chat.Message{{Role: "user", Content: content}}, chat.Config{})
	}

	ids := m.Tok.Encode(formatted)
	placeholders := 0
	for _, id := range ids {
		if id == m.Cfg.AudioTokenID {
			placeholders++
		}
	}
	if placeholders != softTokens {
		core.Print(stderr, "%s audio: tokenizer produced %d audio placeholders, want %d — tokenizer/config disagree on %q",
			cliName(), placeholders, softTokens, gemma4.Gemma4AudioToken)
		return 1
	}

	res, err := multimodalGreedyDecode(ctx, m, ids, nil, []*metal.Array{mel}, nil, *maxTokens)
	if err != nil {
		core.Print(stderr, "%s audio: %v", cliName(), err)
		return 1
	}
	generated := res.Generated
	prefillDur, decodeDur := res.PrefillDur, res.DecodeDur

	core.WriteString(stdout, m.Tok.Decode(generated))
	core.WriteString(stdout, "\n\n")
	rate := 0.0
	if decodeDur > 0 {
		rate = float64(len(generated)) / decodeDur.Seconds()
	}
	core.WriteString(stdout, core.Sprintf(
		"audio %.1fs · %d soft tokens · prefill %dms · %d generated · %.1f tok/s\n",
		float64(len(samples))/float64(m.AudioFeatures.SamplingRate()),
		softTokens, prefillDur.Milliseconds(), len(generated), rate))
	return 0
}
