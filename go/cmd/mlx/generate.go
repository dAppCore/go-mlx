// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"flag"
	"io"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	state "dappco.re/go/inference/state"
	"dappco.re/go/inference/state/filestore"
	"dappco.re/go/mlx"
	"dappco.re/go/inference/state/agent"
	"dappco.re/go/inference/memory"
	"dappco.re/go/mlx/pkg/metal"
	mlxsession "dappco.re/go/mlx/session"
	"dappco.re/go/mlx/spine"
)

// runGenerateCommand loads a model and generates from a prompt with no HTTP
// serve in the path, reporting decode-only tok/s (prefill excluded) for
// like-for-like comparison against other engines on the same model + quant
// (e.g. llama-cli / llama-bench). It prints the generated text too, so it
// doubles as a quick one-shot run.
//
//	lthn-mlx generate ~/models/gemma-4-e2b-it-4bit
//	lthn-mlx generate -max-tokens 256 ~/models/lemer-lite
func runGenerateCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("generate"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	prompt := fs.String("prompt", "Write a detailed Go function that reverses a singly linked list, with inline comments on every step, then explain the pointer dance.", "user prompt")
	maxTokens := fs.Int("max-tokens", 128, "tokens to generate")
	draftPath := fs.String("draft", "auto", "MTP drafter: 'auto' detects one beside a Gemma 4 target (assistant/ pair layout, MTP/ gguf), a path forces it, '' disables")
	draftBlock := fs.Int("draft-block", 0, "MTP draft block (verify forward = carried lead + block-1 proposals); 0 = engine default 5")
	temp := fs.Float64("temp", 1.0, "sampling temperature (0 = greedy/argmax — fastest, fair vs llama-bench)")
	think := fs.Bool("think", false, "enable the thinking channel (off keeps the decode rate clean)")
	contextLen := fs.Int("context", 0, "context length override (0 = model default)")
	kvCacheMode := fs.String("kv-cache", "", "KV cache mode (paged, fp16, q8, kq8vq4, turboquant; empty = load default) — pass 'paged' with -context to bench the serve regime")
	pipeline := fs.Bool("pipeline", true, "one-ahead pipelined decode (false forces the serial loop, for A/B traces)")
	kvStorage := fs.String("kv-storage", "", "retained KV storage dtype (fp16, bf16; empty = native fp32) — mlx-lm and llama.cpp default to fp16-class caches")
	tracePhases := fs.Bool("trace", false, "print the per-token decode time budget — GPU wait vs host-serial work (runs greedy and sampled lanes; ignores -temp)")
	nativeBackend := fs.Bool("native", false, "generate via the no-cgo native token-loop contract (pkg/model + pkg/native) instead of the cgo metal engine")
	stateName := fs.String("state", "", "conversation state name: wake it from the store if present, generate, sleep it back — the no-prompt-replay turn loop")
	stateStore := fs.String("state-store", "", "state store file (default ~/Lethean/data/state/agent.kv)")
	rawState := fs.Bool("raw", false, "with -state: skip chat-framing and run the original raw completion-loop turn (no template, no assistant opener) — the low-level token-loop instrument; ignored without -state")
	fs.Usage = func() {
		name := cliName()
		core.WriteString(stderr, core.Sprintf("Usage: %s generate [flags] <model-path>\n", name))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Load a model and generate from a prompt with no HTTP serve in the path,\n")
		core.WriteString(stderr, "reporting decode-only tok/s (prefill excluded) for like-for-like benching\n")
		core.WriteString(stderr, "against other engines on the same model + quant (e.g. llama-bench). The\n")
		core.WriteString(stderr, "generated text is printed too, so it also serves as a quick one-shot run.\n")
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
		core.WriteString(stderr, core.Sprintf("  %s generate ~/models/gemma-4-e2b-it-4bit\n", name))
		core.WriteString(stderr, "    # one-shot generate + decode tok/s\n")
		core.WriteString(stderr, core.Sprintf("  %s generate -max-tokens 256 ~/models/lemer-lite\n", name))
		core.WriteString(stderr, "    # 256-token decode rate, for like-for-like comparison\n")
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 1 {
		core.WriteString(stderr, core.Sprintf("%s generate: expected exactly one model path\n", cliName()))
		fs.Usage()
		return 2
	}

	loadOpts := []mlx.LoadOption{}
	if *contextLen > 0 {
		loadOpts = append(loadOpts, mlx.WithContextLength(*contextLen))
	}
	if *kvCacheMode != "" {
		loadOpts = append(loadOpts, mlx.WithKVCacheMode(memory.KVCacheMode(*kvCacheMode)))
	}
	if *kvStorage != "" {
		loadOpts = append(loadOpts, mlx.WithKVCacheStorageDType(*kvStorage))
	}
	if *nativeBackend && *tracePhases {
		return runGenerateNativeTrace(ctx, fs.Arg(0), *prompt, *maxTokens, loadOpts, stdout, stderr)
	}
	if *tracePhases {
		return runGenerateTrace(ctx, fs.Arg(0), *prompt, *maxTokens, *pipeline, loadOpts, stdout, stderr)
	}
	if *nativeBackend && *stateName != "" {
		return runGenerateNativeState(ctx, fs.Arg(0), *prompt, *stateName, *stateStore, *maxTokens, float32(*temp), *contextLen, *rawState, *think, loadOpts, stdout, stderr)
	}
	if *stateName != "" {
		return runGenerateState(ctx, fs.Arg(0), *prompt, *stateName, *stateStore, *maxTokens, float32(*temp), *rawState, *think, loadOpts, stdout, stderr)
	}
	var tm inference.TextModel
	var err error
	detection := resolveServeDraft(fs.Arg(0), *draftPath, true)
	if *nativeBackend {
		core.Print(stderr, "%s generate: no-cgo native token-loop contract (pkg/model + pkg/native)", cliName())
		detection = resolveNativeServeDraft(fs.Arg(0), *draftPath)
		if detection.Active() {
			core.Print(stderr, "%s generate: native MTP speculative decode ACTIVE — drafter %s (%s), block %d",
				cliName(), detection.DraftPath, detection.Note, resolvedDraftBlock(*draftBlock))
			tm, err = mlx.LoadNativeSpeculativePairAsTextModelBlock(fs.Arg(0), detection.DraftPath, *draftBlock, loadOpts...)
		} else {
			tm, err = mlx.LoadNativeTextModel(fs.Arg(0), loadOpts...)
		}
	} else {
		// Reactive MTP pair resolution — same ladder as serve: explicit --draft
		// wins, '' disables, 'auto' detects beside a Gemma 4 target.
		if detection.Active() {
			core.Print(stderr, "%s generate: MTP speculative decode ACTIVE — drafter %s (%s), block %d",
				cliName(), detection.DraftPath, detection.Note, resolvedDraftBlock(*draftBlock))
			tm, err = mlx.LoadSpeculativePairAsTextModelBlock(fs.Arg(0), detection.DraftPath, *draftBlock, loadOpts...)
		} else {
			tm, err = mlx.LoadModelAsTextModel(fs.Arg(0), loadOpts...)
		}
	}
	if err != nil {
		core.Print(stderr, "%s generate: load: %v", cliName(), err)
		return 1
	}

	off := !*think
	msgs := []inference.Message{{Role: "user", Content: *prompt}}

	// run generates up to limit tokens and times prefill (start → first token)
	// separately from decode (first → last token), so the reported rate is the
	// steady-state decode rate, comparable to llama-bench's tg.
	run := func(limit int, collect *[]byte) (n int, prefill, decode time.Duration) {
		start := time.Now()
		var first time.Time
		for tok := range tm.Chat(ctx, msgs, inference.WithMaxTokens(limit), inference.WithEnableThinking(&off), inference.WithTemperature(float32(*temp))) {
			if n == 0 {
				first = time.Now()
				prefill = first.Sub(start)
			}
			if collect != nil {
				*collect = append(*collect, tok.Text...)
			}
			n++
		}
		decode = time.Since(first)
		return n, prefill, decode
	}

	run(8, nil) // warm the kernels — first call pays compilation + allocation
	if r := tm.Err(); !r.OK {
		core.Print(stderr, "%s generate: warm: %v", cliName(), r.Error())
		return 1
	}
	var out []byte
	n, prefill, decode := run(*maxTokens, &out)
	if r := tm.Err(); !r.OK {
		core.Print(stderr, "%s generate: %v", cliName(), r.Error())
		return 1
	}
	if n < 2 {
		core.Print(stderr, "%s generate: produced only %d tokens", cliName(), n)
		return 1
	}

	core.WriteString(stdout, string(out))
	core.WriteString(stdout, "\n\n")
	core.WriteString(stdout, core.Sprintf(
		"decode %.1f tok/s  (%d tok / %.3fs, prefill %dms excluded)  ·  total %.1f tok/s\n",
		float64(n-1)/decode.Seconds(), n, decode.Seconds(), prefill.Milliseconds(),
		float64(n)/(prefill+decode).Seconds(),
	))
	printGenerateMTPMetrics(stdout, tm)
	return 0
}

// resolvedDraftBlock reports the block the MTP lane will run for a flag value
// (0 = engine default).
func resolvedDraftBlock(flag int) int {
	if flag > 0 {
		return flag
	}
	return mlx.MTPDefaultDraftBlock
}

// printGenerateMTPMetrics appends the MTP acceptance line when the generation
// rode the speculative lane — the bench instrument's read on whether the
// drafter is earning its keep.
func printGenerateMTPMetrics(stdout io.Writer, tm inference.TextModel) {
	provider, ok := tm.(interface{ MTPMetrics() *metal.MTPMetrics })
	if !ok {
		return
	}
	mtp := provider.MTPMetrics()
	if mtp == nil {
		return
	}
	core.WriteString(stdout, core.Sprintf(
		"mtp: %.0f%% acceptance (%d/%d drafted) over %d verify forwards\n",
		mtp.AcceptanceRate*100, mtp.AcceptedTokens, mtp.ProposedTokens, mtp.TargetVerifyCalls,
	))
}

// stateChatFormatter chat-frames a -state turn the way serve's conversation
// continuity frames every stateless request (conversation_continuity.go —
// the chat-framed prior art this turn loop now follows): FormatChatPrompt
// opens a fresh conversation (full template + BOS from empty history) and
// FormatChatContinuation appends onto a woken session (closes the
// previously open model turn, renders only the new user turn, reopens the
// assistant header) — no prior-turn replay. mlx.Model (the metal lane) and
// nativeGenerateStateModel (the native lane) both implement it.
type stateChatFormatter interface {
	FormatChatPrompt(messages []inference.Message) string
	FormatChatContinuation(messages []inference.Message) string
}

// stateThinkingFormatter is the optional upgrade a stateChatFormatter may
// offer: the same two renders with an explicit thinking override, so the
// -think flag reaches the template instead of the model default (thinking
// on for gemma4) silently consuming small -state token budgets inside the
// thought channel. Both lanes' models implement it.
type stateThinkingFormatter interface {
	FormatChatPromptThinking(messages []inference.Message, thinking *bool) string
	FormatChatContinuationThinking(messages []inference.Message, thinking *bool) string
}

// stateTurnMessages wraps a -state turn's prompt as the single new user
// message a stateChatFormatter renders. The turn loop carries one message
// per invocation — the prior conversation lives in the woken KV state, not
// in a replayed message list.
func stateTurnMessages(prompt string) []inference.Message {
	return []inference.Message{{Role: "user", Content: prompt}}
}

// runGenerateState runs one conversation turn through the durable state
// system — the no-prompt-replay loop. If the named state exists in the store
// it is woken (KV restored from .kv blocks, no re-prefill of prior turns) and
// only the new turn is appended; otherwise the prompt opens a fresh session.
// After generation the session sleeps back to the store, so the next
// invocation's turn starts where this one ended.
//
// By default each turn is chat-framed (see stateChatFormatter); raw
// preserves the original completion-loop turn — the prompt prefills or
// appends byte-for-byte with no template, the low-level token-loop
// instrument.
//
//	lthn-mlx generate -state chat1 -prompt "Hello, who are you?" <model>
//	lthn-mlx generate -state chat1 -prompt "And what did I just ask you?" <model>
func runGenerateState(ctx context.Context, modelPath, prompt, name, storePath string, maxTokens int, temp float32, raw, think bool, loadOpts []mlx.LoadOption, stdout, stderr io.Writer) int {
	if storePath == "" {
		homeR := core.UserHomeDir()
		if !homeR.OK {
			core.Print(stderr, "%s generate: resolve home for default -state-store", cliName())
			return 1
		}
		home, _ := homeR.Value.(string)
		storePath = core.PathJoin(home, "Lethean", "data", "state", "agent.kv")
	}
	store, err := openOrCreateStateStore(ctx, storePath)
	if err != nil {
		core.Print(stderr, "%s generate: state store %s: %v", cliName(), storePath, err)
		return 1
	}
	defer store.Close()

	m, err := mlx.LoadModel(modelPath, loadOpts...)
	if err != nil {
		core.Print(stderr, "%s generate: load: %v", cliName(), err)
		return 1
	}
	defer m.Close()
	sess, err := m.NewSession()
	if err != nil {
		core.Print(stderr, "%s generate: session: %v", cliName(), err)
		return 1
	}
	defer sess.Close()
	var formatter stateChatFormatter
	if !raw {
		formatter = m
	}
	return runGenerateStateSession(ctx, prompt, name, storePath, maxTokens, temp, store, sess, formatter, &think, stdout, stderr)
}

type nativeGenerateStateModel interface {
	inference.TextModel
	Info() inference.ModelInfo
	NewSession() metal.SessionHandle
	FormatChatPrompt(messages []inference.Message) string
	FormatChatContinuation(messages []inference.Message) string
}

func runGenerateNativeState(ctx context.Context, modelPath, prompt, name, storePath string, maxTokens int, temp float32, contextLen int, raw, think bool, loadOpts []mlx.LoadOption, stdout, stderr io.Writer) int {
	if storePath == "" {
		homeR := core.UserHomeDir()
		if !homeR.OK {
			core.Print(stderr, "%s generate: resolve home for default -state-store", cliName())
			return 1
		}
		home, _ := homeR.Value.(string)
		storePath = core.PathJoin(home, "Lethean", "data", "state", "agent.kv")
	}
	store, err := openOrCreateStateStore(ctx, storePath)
	if err != nil {
		core.Print(stderr, "%s generate: state store %s: %v", cliName(), storePath, err)
		return 1
	}
	defer store.Close()

	core.Print(stderr, "%s generate: no-cgo native state token-loop contract (pkg/model + pkg/native)", cliName())
	tm, err := mlx.LoadNativeTextModel(modelPath, loadOpts...)
	if err != nil {
		core.Print(stderr, "%s generate: load: %v", cliName(), err)
		return 1
	}
	nativeState, ok := tm.(nativeGenerateStateModel)
	if !ok {
		if closer, closeOK := tm.(interface{ Close() core.Result }); closeOK {
			_ = closer.Close()
		}
		core.Print(stderr, "%s generate: native state model does not support sessions", cliName())
		return 1
	}
	defer nativeState.Close()
	handle := nativeState.NewSession()
	if handle == nil {
		core.Print(stderr, "%s generate: native state session: nil session", cliName())
		return 1
	}
	info := nativeGenerateStateModelInfo(nativeState.Info(), contextLen)
	sess := mlxsession.New(handle, info, nil)
	defer sess.Close()
	var formatter stateChatFormatter
	if !raw {
		formatter = nativeState
	}
	return runGenerateStateSession(ctx, prompt, name, storePath, maxTokens, temp, store, sess, formatter, &think, stdout, stderr)
}

func nativeGenerateStateModelInfo(info inference.ModelInfo, contextLen int) spine.ModelInfo {
	if contextLen <= 0 {
		contextLen = 4096
	}
	return spine.ModelInfo{
		Architecture:  info.Architecture,
		VocabSize:     info.VocabSize,
		NumLayers:     info.NumLayers,
		HiddenSize:    info.HiddenSize,
		QuantBits:     info.QuantBits,
		QuantGroup:    info.QuantGroup,
		ContextLength: contextLen,
	}
}

type generateStateStore interface {
	state.Store
	state.Writer
}

// runGenerateStateSession runs one -state turn against an already-open
// session. formatter chat-frames the new turn — FormatChatPrompt for a fresh
// session, FormatChatContinuation for a woken one — matching serve's
// conversation continuity (conversation_continuity.go). A nil formatter is
// the -raw contract: the prompt prefills or appends byte-for-byte with no
// template, exactly the loop's original completion-style behaviour.
func runGenerateStateSession(ctx context.Context, prompt, name, storePath string, maxTokens int, temp float32, store generateStateStore, sess *mlx.ModelSession, formatter stateChatFormatter, thinking *bool, stdout, stderr io.Writer) int {
	entryURI := "mlx://agent/" + name
	indexURI := entryURI + "/index"

	// Wake if the named state exists; a missing index means turn one.
	woke := false
	var wakeDur, prefillDur time.Duration
	var wakeReport *agent.WakeReport
	if _, idxErr := agent.LoadStateIndex(ctx, store, indexURI); idxErr == nil {
		start := time.Now()
		var wakeErr error
		wakeReport, wakeErr = sess.WakeAgentMemory(ctx, store, agent.WakeOptions{IndexURI: indexURI, EntryURI: entryURI})
		if wakeErr != nil {
			core.Print(stderr, "%s generate: wake %s: %v", cliName(), name, wakeErr)
			return 1
		}
		wakeDur = time.Since(start)
		start = time.Now()
		// Continuation form: close the previously open model turn, render
		// only the new user turn, reopen the assistant header — no replay of
		// the retained prefix, matching the woke-prefix-tokens report below.
		turn := "\n" + prompt
		if formatter != nil {
			turn = formatter.FormatChatContinuation(stateTurnMessages(prompt))
			if tf, ok := formatter.(stateThinkingFormatter); ok && thinking != nil {
				turn = tf.FormatChatContinuationThinking(stateTurnMessages(prompt), thinking)
			}
		}
		if err := sess.AppendPrompt(turn); err != nil {
			core.Print(stderr, "%s generate: append turn: %v", cliName(), err)
			return 1
		}
		prefillDur = time.Since(start)
		woke = true
	} else {
		var notFound *state.URIChunkNotFoundError
		if !core.As(idxErr, &notFound) {
			core.Print(stderr, "%s generate: state index %s: %v", cliName(), indexURI, idxErr)
			return 1
		}
		start := time.Now()
		// Fresh form: the full chat template from empty history (BOS,
		// optional system/thinking preamble, the user turn, the assistant
		// opener).
		turn := prompt
		if formatter != nil {
			turn = formatter.FormatChatPrompt(stateTurnMessages(prompt))
			if tf, ok := formatter.(stateThinkingFormatter); ok && thinking != nil {
				turn = tf.FormatChatPromptThinking(stateTurnMessages(prompt), thinking)
			}
		}
		if err := sess.Prefill(turn); err != nil {
			core.Print(stderr, "%s generate: prefill: %v", cliName(), err)
			return 1
		}
		prefillDur = time.Since(start)
	}

	var out []byte
	tokens := 0
	start := time.Now()
	for tok := range sess.GenerateStream(ctx, mlx.WithMaxTokens(maxTokens), mlx.WithTemperature(temp)) {
		out = append(out, tok.Text...)
		tokens++
	}
	decodeDur := time.Since(start)
	if err := sess.Err(); err != nil {
		core.Print(stderr, "%s generate: %v", cliName(), err)
		return 1
	}

	start = time.Now()
	sleepReport, err := sess.SleepAgentMemory(ctx, store, agent.SleepOptions{EntryURI: entryURI, Title: name})
	if err != nil {
		core.Print(stderr, "%s generate: sleep %s: %v", cliName(), name, err)
		return 1
	}
	sleepDur := time.Since(start)

	core.WriteString(stdout, string(out))
	core.WriteString(stdout, "\n\n")
	if woke {
		core.WriteString(stdout, core.Sprintf(
			"turn: woke %d prefix tokens in %dms (no replay) · new-turn prefill %dms\n",
			wakeReport.PrefixTokens, wakeDur.Milliseconds(), prefillDur.Milliseconds()))
	} else {
		core.WriteString(stdout, core.Sprintf(
			"turn: fresh state · prefill %dms\n", prefillDur.Milliseconds()))
	}
	if decodeDur > 0 && tokens > 1 {
		core.WriteString(stdout, core.Sprintf(
			"decode %.1f tok/s (%d tok)\n", float64(tokens)/decodeDur.Seconds(), tokens))
	}
	core.WriteString(stdout, core.Sprintf(
		"slept %d tokens -> %d blocks in %dms\n",
		sleepReport.TokenCount, sleepReport.BlocksWritten, sleepDur.Milliseconds()))
	core.WriteString(stdout, core.Sprintf("state: %s (%s)\n", name, storePath))
	return 0
}

// openOrCreateStateStore opens the append-only state file, creating it (and
// its directory) on first use.
func openOrCreateStateStore(ctx context.Context, path string) (*filestore.Store, error) {
	if core.Stat(path).OK {
		return filestore.Open(ctx, path)
	}
	if dir := core.PathDir(path); dir != "" {
		if r := core.MkdirAll(dir, 0o755); !r.OK {
			return nil, core.E("generate.stateStore", "mkdir store dir", r.Value.(error))
		}
	}
	return filestore.Create(ctx, path)
}

// runGenerateTrace loads the model once via the root API and prints the
// per-token decode time budget from the engine's phase trace: how long the
// host blocks waiting on the GPU result versus how long it spends in serial
// host work (graph build, detokenise, yield) while the GPU sits idle. The
// split locates where decode tok/s goes. Both lanes run on the same load.
func runGenerateTrace(ctx context.Context, modelPath, prompt string, maxTokens int, pipeline bool, loadOpts []mlx.LoadOption, stdout, stderr io.Writer) int {
	m, err := mlx.LoadModel(modelPath, loadOpts...)
	if err != nil {
		core.Print(stderr, "%s generate: load: %v", cliName(), err)
		return 1
	}
	defer m.Close()
	if !pipeline {
		// After load: the model's EngineFeatures.Apply set the gate.
		defer metal.SetRuntimeGate(metal.GatePipelinedDecode, false)()
	}

	// Sessions are the serve's decode path (retained KV, the pipelined loop);
	// tracing through a session measures what the product runs.
	chatPrompt := m.FormatChatPrompt([]inference.Message{{Role: "user", Content: prompt}})
	run := func(temp float32, limit int, trace bool) bool {
		sess, err := m.NewSession()
		if err != nil {
			core.Print(stderr, "%s generate: session: %v", cliName(), err)
			return false
		}
		defer sess.Close()
		if err := sess.Prefill(chatPrompt); err != nil {
			core.Print(stderr, "%s generate: prefill: %v", cliName(), err)
			return false
		}
		opts := []mlx.GenerateOption{mlx.WithMaxTokens(limit), mlx.WithTemperature(temp)}
		if trace {
			opts = append(opts, mlx.WithTokenPhaseTrace())
		}
		for range sess.GenerateStream(ctx, opts...) {
		}
		if err := sess.Err(); err != nil {
			core.Print(stderr, "%s generate: %v", cliName(), err)
			return false
		}
		return true
	}

	if !run(0, 8, false) { // warm: kernel compilation + allocation
		return 1
	}
	lanes := []struct {
		name string
		temp float32
	}{
		{"greedy (temp=0)", 0},
		{"sampled (temp=1)", 1},
	}
	for _, lane := range lanes {
		if !run(lane.temp, maxTokens, true) {
			return 1
		}
		metrics := m.Metrics()
		lane.name += core.Sprintf(" · lane=%s", metrics.DecodeLane)
		if metrics.DecodeLaneReason != "" {
			lane.name += core.Sprintf(" (%s)", metrics.DecodeLaneReason)
		}
		if metrics.GeneratedTokens > 0 {
			lane.name += core.Sprintf(" · compiled-hits/token %.1f", float64(metrics.CompiledLayerHits)/float64(metrics.GeneratedTokens))
		}
		printTokenPhaseBudget(stdout, lane.name, metrics)
	}
	return 0
}

type nativeGenerateTraceModel interface {
	inference.TextModel
	NewSession() metal.SessionHandle
}

type nativeGenerateTraceSession interface {
	LastTokenPhases() []metal.TokenPhaseTrace
}

func runGenerateNativeTrace(ctx context.Context, modelPath, prompt string, maxTokens int, loadOpts []mlx.LoadOption, stdout, stderr io.Writer) int {
	tm, err := mlx.LoadNativeTextModel(modelPath, loadOpts...)
	if err != nil {
		core.Print(stderr, "%s generate: native trace load: %v", cliName(), err)
		return 1
	}
	defer tm.Close()
	nativeModel, ok := tm.(nativeGenerateTraceModel)
	if !ok {
		core.Print(stderr, "%s generate: native trace: loaded model does not expose sessions", cliName())
		return 1
	}
	chatPrompt, err := nativeGenerateTracePrompt(tm, prompt)
	if err != nil {
		core.Print(stderr, "%s generate: native trace prompt: %v", cliName(), err)
		return 1
	}

	run := func(temp float32, limit int, trace bool) ([]metal.TokenPhaseTrace, bool) {
		handle := nativeModel.NewSession()
		if handle == nil {
			core.Print(stderr, "%s generate: native trace session: nil session", cliName())
			return nil, false
		}
		sess := mlxsession.New(handle, nativeGenerateTraceModelInfo(nativeModel.Info()), nil)
		defer sess.Close()
		if err := sess.Prefill(chatPrompt); err != nil {
			core.Print(stderr, "%s generate: native trace prefill: %v", cliName(), err)
			return nil, false
		}
		opts := []mlx.GenerateOption{mlx.WithMaxTokens(limit), mlx.WithTemperature(temp)}
		if trace {
			opts = append(opts, mlx.WithTokenPhaseTrace())
		}
		for range sess.GenerateStream(ctx, opts...) {
		}
		if err := sess.Err(); err != nil {
			core.Print(stderr, "%s generate: native trace: %v", cliName(), err)
			return nil, false
		}
		if !trace {
			return nil, true
		}
		tracer, ok := handle.(nativeGenerateTraceSession)
		if !ok {
			core.Print(stderr, "%s generate: native trace: session did not expose token phases", cliName())
			return nil, false
		}
		return tracer.LastTokenPhases(), true
	}

	if _, ok := run(0, 8, false); !ok {
		return 1
	}
	lanes := []struct {
		name string
		temp float32
	}{
		{"native greedy (temp=0)", 0},
		{"native sampled (temp=1)", 1},
	}
	for _, lane := range lanes {
		phases, ok := run(lane.temp, maxTokens, true)
		if !ok {
			return 1
		}
		metrics := nativeGenerateTraceMetrics(phases)
		printTokenPhaseBudget(stdout, lane.name+" · lane=native", metrics)
	}
	return 0
}

func nativeGenerateTracePrompt(tm inference.TextModel, prompt string) (string, error) {
	templater, ok := tm.(interface {
		ApplyChatTemplate([]inference.Message) (string, error)
	})
	if !ok {
		return prompt, nil
	}
	return templater.ApplyChatTemplate([]inference.Message{{Role: "user", Content: prompt}})
}

func nativeGenerateTraceModelInfo(info inference.ModelInfo) spine.ModelInfo {
	return spine.ModelInfo{
		Architecture: info.Architecture,
		VocabSize:    info.VocabSize,
		NumLayers:    info.NumLayers,
		HiddenSize:   info.HiddenSize,
		QuantBits:    info.QuantBits,
		QuantGroup:   info.QuantGroup,
	}
}

func nativeGenerateTraceMetrics(phases []metal.TokenPhaseTrace) mlx.Metrics {
	return mlx.Metrics{
		GeneratedTokens: len(phases),
		TokenPhases:     nativeGenerateTraceTokenPhases(phases),
		DecodeLane:      "native",
	}
}

func nativeGenerateTraceTokenPhases(phases []metal.TokenPhaseTrace) []mlx.TokenPhaseTrace {
	if len(phases) == 0 {
		return nil
	}
	out := make([]mlx.TokenPhaseTrace, len(phases))
	for i := range phases {
		phase := &phases[i]
		out[i] = mlx.TokenPhaseTrace{
			Step:                   phase.Step,
			TokenID:                phase.TokenID,
			TokenText:              phase.TokenText,
			FinalToken:             phase.FinalToken,
			TotalDuration:          phase.TotalDuration,
			LogitsDuration:         phase.LogitsDuration,
			SampleDuration:         phase.SampleDuration,
			SampleEvalDuration:     phase.SampleEvalDuration,
			TokenReadDuration:      phase.TokenReadDuration,
			DecodeTextDuration:     phase.DecodeTextDuration,
			ProbeTokenDuration:     phase.ProbeTokenDuration,
			YieldDuration:          phase.YieldDuration,
			NextInputDuration:      phase.NextInputDuration,
			ForwardDuration:        phase.ForwardDuration,
			PrefetchDuration:       phase.PrefetchDuration,
			PrefetchLogitsDuration: phase.PrefetchLogitsDuration,
			PrefetchCacheDuration:  phase.PrefetchCacheDuration,
			MaterializeDuration:    phase.MaterializeDuration,
			DetachDuration:         phase.DetachDuration,
			CacheProbeDuration:     phase.CacheProbeDuration,
			OtherDuration:          phase.OtherDuration,
			NativeEvents:           nativeGenerateTraceEvents(phase.NativeEvents),
		}
	}
	return out
}

func nativeGenerateTraceEvents(events []metal.NativePhaseTrace) []mlx.NativePhaseTrace {
	if len(events) == 0 {
		return nil
	}
	out := make([]mlx.NativePhaseTrace, len(events))
	for i := range events {
		event := &events[i]
		out[i] = mlx.NativePhaseTrace{
			Name:     event.Name,
			Duration: event.Duration,
			Error:    event.Error,
			Pages:    event.Pages,
			Tokens:   event.Tokens,
		}
	}
	return out
}

// printTokenPhaseBudget averages the engine's per-token phase trace over the
// warm tokens (step 0 and the final token are skipped) and reports the
// GPU-wait vs host-serial split plus each phase's share.
func printTokenPhaseBudget(stdout io.Writer, lane string, metrics mlx.Metrics) {
	type row struct {
		name string
		get  func(mlx.TokenPhaseTrace) time.Duration
	}
	rows := []row{
		{"token-read wait (GPU busy)", func(p mlx.TokenPhaseTrace) time.Duration { return p.TokenReadDuration }},
		{"sample eval wait (GPU busy)", func(p mlx.TokenPhaseTrace) time.Duration { return p.SampleEvalDuration }},
		{"forward graph build (host)", func(p mlx.TokenPhaseTrace) time.Duration { return p.ForwardDuration }},
		{"logits slice (host)", func(p mlx.TokenPhaseTrace) time.Duration { return p.LogitsDuration }},
		{"sample build (host)", func(p mlx.TokenPhaseTrace) time.Duration { return p.SampleDuration }},
		{"detach (host)", func(p mlx.TokenPhaseTrace) time.Duration { return p.DetachDuration }},
		{"decode text (host)", func(p mlx.TokenPhaseTrace) time.Duration { return p.DecodeTextDuration }},
		{"yield to consumer (host)", func(p mlx.TokenPhaseTrace) time.Duration { return p.YieldDuration }},
		{"next input upload (host)", func(p mlx.TokenPhaseTrace) time.Duration { return p.NextInputDuration }},
		{"prefetch submit (host)", func(p mlx.TokenPhaseTrace) time.Duration { return p.PrefetchDuration }},
		{"  prefetch: logits graph", func(p mlx.TokenPhaseTrace) time.Duration { return p.PrefetchLogitsDuration }},
		{"  prefetch: cache state", func(p mlx.TokenPhaseTrace) time.Duration { return p.PrefetchCacheDuration }},
		{"materialize (host)", func(p mlx.TokenPhaseTrace) time.Duration { return p.MaterializeDuration }},
		{"cache probe (host)", func(p mlx.TokenPhaseTrace) time.Duration { return p.CacheProbeDuration }},
		{"probe token (host)", func(p mlx.TokenPhaseTrace) time.Duration { return p.ProbeTokenDuration }},
		{"other (host)", func(p mlx.TokenPhaseTrace) time.Duration { return p.OtherDuration }},
	}

	var n int
	var total, gpu time.Duration
	sums := make([]time.Duration, len(rows))
	for _, p := range metrics.TokenPhases {
		if p.Step == 0 || p.FinalToken {
			continue
		}
		n++
		total += p.TotalDuration
		gpu += p.TokenReadDuration + p.SampleEvalDuration
		for i, r := range rows {
			sums[i] += r.get(p)
		}
	}
	if n == 0 {
		core.WriteString(stdout, core.Sprintf("%s: no warm token phases captured\n", lane))
		return
	}
	ms := func(d time.Duration) float64 { return float64(d.Microseconds()) / 1000.0 / float64(n) }
	avgTotal := ms(total)
	avgGPU := ms(gpu)
	avgHost := avgTotal - avgGPU
	core.WriteString(stdout, core.Sprintf("\n%s — %d warm tokens · %.3f ms/token · %.1f tok/s\n",
		lane, n, avgTotal, 1000.0/avgTotal))
	core.WriteString(stdout, core.Sprintf("  GPU wait   %8.3f ms  %5.1f%%\n", avgGPU, 100*avgGPU/avgTotal))
	ceiling := "n/a"
	if avgGPU > 0 {
		ceiling = core.Sprintf("%.1f", 1000.0/avgGPU)
	}
	core.WriteString(stdout, core.Sprintf("  host serial%8.3f ms  %5.1f%%   <- GPU idle; tok/s ceiling if zeroed: %s\n",
		avgHost, 100*avgHost/avgTotal, ceiling))
	for i, r := range rows {
		avg := ms(sums[i])
		if avg < 0.001 {
			continue
		}
		core.WriteString(stdout, core.Sprintf("    %-28s %8.3f ms  %5.1f%%\n", r.name, avg, 100*avg/avgTotal))
	}
}
