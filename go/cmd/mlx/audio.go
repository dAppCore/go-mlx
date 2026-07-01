// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"flag"
	"io"
	"time"

	core "dappco.re/go"
	"dappco.re/go/mlx/chat"
	"dappco.re/go/mlx/pkg/model"
	_ "dappco.re/go/mlx/pkg/model/gemma4"
	gemma4chat "dappco.re/go/mlx/pkg/model/gemma4/chat"
	"dappco.re/go/mlx/pkg/native"
	"dappco.re/go/mlx/pkg/tokenizer"
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

	return runNativeAudioCommand(ctx, fs.Arg(0), *wavPath, *prompt, *maxTokens, *chatFlag, stdout, stderr)
}

type nativeAudioCommandModel interface {
	model.SessionModel
	AcceptsAudioInput() bool
	AudioPlaceholderTokenID() int32
	AudioPlaceholderBlock(int) string
	AudioSoftTokens(int) int
	ProjectAudioFeatures([]byte, int, int) ([]byte, error)
}

type nativeAudioCommandSession interface {
	PrefillTokenEmbeddings([]int32, [][]byte) error
	GenerateFromCacheEach(int, int, func(int32) bool) ([]int32, error)
}

func runNativeAudioCommand(ctx context.Context, modelPath, wavPath, prompt string, maxTokens int, chatFlag bool, stdout, stderr io.Writer) int {
	if maxTokens <= 0 {
		core.Print(stderr, "%s audio: max-tokens must be > 0", cliName())
		return 1
	}

	tm, err := native.LoadTokenModelDir(modelPath, maxTokens+4096)
	if err != nil {
		core.Print(stderr, "%s audio: load: %v", cliName(), err)
		return 1
	}
	if c, ok := tm.(interface{ Close() error }); ok {
		defer func() { _ = c.Close() }()
	}
	m, ok := tm.(nativeAudioCommandModel)
	if !ok || !m.AcceptsAudioInput() {
		core.Print(stderr, "%s audio: this checkpoint has no audio tower — use a Gemma 4 E2B/E4B snapshot", cliName())
		return 1
	}
	if m.AudioPlaceholderTokenID() == 0 {
		core.Print(stderr, "%s audio: model config declares no audio_token_id", cliName())
		return 1
	}

	featureCfg, err := native.LoadAudioFeatureConfig(modelPath)
	if err != nil {
		core.Print(stderr, "%s audio: %v", cliName(), err)
		return 1
	}
	if featureCfg == nil {
		core.Print(stderr, "%s audio: model ships no processor_config.json audio front-end", cliName())
		return 1
	}
	extractor, err := native.NewAudioFeatureExtractor(featureCfg)
	if err != nil {
		core.Print(stderr, "%s audio: features: %v", cliName(), err)
		return 1
	}

	tok, err := tokenizer.LoadTokenizer(core.PathJoin(modelPath, "tokenizer.json"))
	if err != nil {
		core.Print(stderr, "%s audio: tokenizer: %v", cliName(), err)
		return 1
	}

	samples, err := readWAVMono(wavPath, extractor.SamplingRate())
	if err != nil {
		core.Print(stderr, "%s audio: %v", cliName(), err)
		return 1
	}
	rawFeatures, frames, melBins, err := native.AudioInputFeatures(samples, extractor)
	if err != nil {
		core.Print(stderr, "%s audio: features: %v", cliName(), err)
		return 1
	}
	softTokens := m.AudioSoftTokens(frames)
	if softTokens <= 0 {
		core.Print(stderr, "%s audio: features produced no audio soft tokens", cliName())
		return 1
	}
	audioFeatures, err := m.ProjectAudioFeatures(rawFeatures, frames, melBins)
	if err != nil {
		core.Print(stderr, "%s audio: project: %v", cliName(), err)
		return 1
	}

	// The HF processor convention: BOA + AudioToken×softTokens + EOA ahead
	// of the question text, inside the user turn.
	audioBlock := m.AudioPlaceholderBlock(softTokens)
	if audioBlock == "" {
		core.Print(stderr, "%s audio: model config declares no audio placeholder tokens", cliName())
		return 1
	}
	content := audioBlock + "\n" + prompt
	formatted := content
	if chatFlag {
		formatted = gemma4chat.Format([]chat.Message{{Role: "user", Content: content}}, chat.Config{})
	}

	ids := tok.Encode(formatted)
	placeholders := countTokenID(ids, m.AudioPlaceholderTokenID())
	if placeholders != softTokens {
		core.Print(stderr, "%s audio: tokenizer produced %d audio placeholders, want %d",
			cliName(), placeholders, softTokens)
		return 1
	}

	res, err := nativeAudioGreedyDecode(ctx, m, tok, ids, audioFeatures, maxTokens)
	if err != nil {
		core.Print(stderr, "%s audio: %v", cliName(), err)
		return 1
	}

	core.WriteString(stdout, tok.Decode(res.Generated))
	core.WriteString(stdout, "\n\n")
	rate := 0.0
	if res.DecodeDur > 0 {
		rate = float64(len(res.Generated)) / res.DecodeDur.Seconds()
	}
	core.WriteString(stdout, core.Sprintf(
		"audio %.1fs · %d soft tokens · prefill %dms · %d generated · %.1f tok/s\n",
		float64(len(samples))/float64(extractor.SamplingRate()),
		softTokens, res.PrefillDur.Milliseconds(), len(res.Generated), rate))
	return 0
}

func nativeAudioGreedyDecode(ctx context.Context, m nativeAudioCommandModel, tok *tokenizer.Tokenizer, ids []int32, features []byte, maxTokens int) (multimodalDecodeResult, error) {
	var res multimodalDecodeResult

	start := time.Now()
	embeddings, err := nativeAudioPromptEmbeddings(m, ids, features)
	if err != nil {
		return res, err
	}
	stepper, err := m.OpenSession()
	if err != nil {
		return res, err
	}
	if c, ok := stepper.(interface{ Close() error }); ok {
		defer func() { _ = c.Close() }()
	}
	sess, ok := stepper.(nativeAudioCommandSession)
	if !ok {
		return res, core.NewError("native audio session does not support multimodal prefill")
	}
	if err := sess.PrefillTokenEmbeddings(ids, embeddings); err != nil {
		return res, err
	}
	res.PrefillDur = time.Since(start)

	stopIDs := nativeCommandStopIDs(tok)
	eos := -1
	if tok.HasEOSToken() {
		eos = int(tok.EOSToken())
	}
	emit := func(id int32) bool {
		if ctx != nil {
			if err := ctx.Err(); err != nil {
				return false
			}
		}
		_, stop := stopIDs[id]
		return !stop
	}

	decodeStart := time.Now()
	res.Generated, err = sess.GenerateFromCacheEach(maxTokens, eos, emit)
	res.DecodeDur = time.Since(decodeStart)
	if err != nil {
		return res, err
	}
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return res, err
		}
	}
	if len(res.Generated) > 0 {
		if _, stop := stopIDs[res.Generated[len(res.Generated)-1]]; stop {
			res.Generated = res.Generated[:len(res.Generated)-1]
		}
	}
	return res, nil
}

func nativeAudioPromptEmbeddings(m nativeAudioCommandModel, ids []int32, features []byte) ([][]byte, error) {
	placeholderID := m.AudioPlaceholderTokenID()
	embeddings := make([][]byte, len(ids))
	off, used := 0, 0
	for i, id := range ids {
		emb, err := m.Embed(id)
		if err != nil {
			return nil, err
		}
		if len(emb) == 0 {
			return nil, core.NewError("native audio prompt embedding is empty")
		}
		if id != placeholderID {
			embeddings[i] = append([]byte(nil), emb...)
			continue
		}
		if off+len(emb) > len(features) {
			return nil, core.NewError("native audio feature rows do not match placeholder embeddings")
		}
		embeddings[i] = append([]byte(nil), features[off:off+len(emb)]...)
		off += len(emb)
		used++
	}
	if used == 0 {
		return nil, core.NewError("native audio prompt has no audio placeholders")
	}
	if off != len(features) {
		return nil, core.NewError("native audio prompt has unused feature rows")
	}
	return embeddings, nil
}

func nativeCommandStopIDs(tok *tokenizer.Tokenizer) map[int32]struct{} {
	stopIDs := make(map[int32]struct{}, 2)
	if tok.HasEOSToken() {
		stopIDs[tok.EOSToken()] = struct{}{}
	}
	if eot := tok.Encode("<turn|>"); len(eot) == 1 {
		stopIDs[eot[0]] = struct{}{}
	}
	return stopIDs
}
