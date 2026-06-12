// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"flag"
	"io"
	"time"

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

	capacity := len(ids) + *maxTokens + 64
	caches := make([]metal.Cache, m.NumLayers())
	for i := range caches {
		caches[i] = metal.NewFixedKVCache(capacity)
	}
	defer metal.FreeCaches(caches)

	// Stop set: tokenizer EOS plus the chat end-of-turn when it encodes to
	// a single id.
	stopIDs := map[int32]struct{}{m.Tok.EOSToken(): {}}
	if eot := m.Tok.Encode("<turn|>"); len(eot) == 1 {
		stopIDs[eot[0]] = struct{}{}
	}

	start := time.Now()
	prefill := metal.FromValues(ids, 1, len(ids))
	logits := m.ForwardUnifiedMultiModal(prefill, nil, []*metal.Array{mel}, caches)
	metal.Free(prefill)
	prefillDur := time.Since(start)

	generated := make([]int32, 0, *maxTokens)
	decodeStart := time.Now()
	for len(generated) < *maxTokens {
		select {
		case <-ctx.Done():
			core.Print(stderr, "%s audio: cancelled", cliName())
			return 1
		default:
		}
		last := metal.SliceAxis(logits, 1, int32(logits.Dim(1)-1), int32(logits.Dim(1)))
		next := metal.Argmax(last, -1, false)
		if err := metal.Eval(next); err != nil {
			metal.Free(logits, last, next)
			core.Print(stderr, "%s audio: decode: %v", cliName(), err)
			return 1
		}
		id := int32(next.Int())
		metal.Free(logits, last, next)
		metal.DetachCaches(caches)
		if _, stop := stopIDs[id]; stop {
			break
		}
		generated = append(generated, id)
		step := metal.FromValues([]int32{id}, 1, 1)
		logits = m.Forward(step, caches)
		metal.Free(step)
	}
	if len(generated) == *maxTokens {
		// The bound fired before a stop token: the dangling logits from the
		// final Forward are still live.
		metal.Free(logits)
	}
	decodeDur := time.Since(decodeStart)

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
