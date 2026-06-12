// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"flag"
	"io"
	"net/http"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/state/filestore"
	mlx "dappco.re/go/mlx"
	"dappco.re/go/mlx/openai"
)

// runServeCommand mounts the OpenAI / Anthropic / Ollama compatibility HTTP
// surface from dappco.re/go/mlx/openai on a local listen address. lthn-mlx
// becomes a sovereign localhost endpoint that any OpenAI-compatible client
// (go-ai providers/openai, plain curl, llama-index, openai-python, etc.) can
// talk to over the standard wire.
//
// Higher-level consumers (lthn-lem-runtime, lem-desktop, lthn/desktop) should
// reach this through HTTP, never by importing the openai package directly —
// that's the whole point of the binary boundary.
//
//	lthn-mlx serve --model /Volumes/Data/models/lemer-lite --addr :36911
//	curl http://127.0.0.1:36911/v1/health
//	curl http://127.0.0.1:36911/v1/chat/completions -H 'content-type: application/json' \
//	     -d '{"model":"lemer-lite","messages":[{"role":"user","content":"hi"}]}'
func runServeCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("serve"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	addr := fs.String("addr", ":36911", "listen address (Lethean's own port — never collides with an Ollama install)")
	modelPath := fs.String("model", "", "model path to load; empty starts the driver model-less (load a model later via POST /v1/admin/serve/reload)")
	draftPath := fs.String("draft", "auto", "MTP drafter: 'auto' detects one beside a Gemma 4 target (assistant/ pair layout, MTP/ gguf), a path forces it, '' disables")
	draftDetect := fs.Bool("draft-detect", true, "reactive drafter detection for Gemma 4 targets (false = only an explicit --draft engages MTP)")
	draftBlock := fs.Int("draft-block", 0, "MTP draft block (verify forward = carried lead + block-1 proposals); 0 = engine default 5, tuned profile overrides when present")
	contextLen := fs.Int("context", 0, "override context length; 0 uses the model's default")
	kvCacheMode := fs.String("kv-cache", "", "KV cache mode (paged, fp16, q8, kq8vq4, turboquant; empty = load default) — 'paged' with -context activates the fixed-cache compiled decode lane")
	readTimeout := fs.Duration("read-timeout", 30*time.Second, "HTTP read header timeout")
	writeTimeout := fs.Duration("write-timeout", 5*time.Minute, "HTTP write timeout (covers full streaming response)")
	shutdownTimeout := fs.Duration("shutdown-timeout", 10*time.Second, "graceful shutdown deadline after SIGINT/SIGTERM")
	printAdminToken := fs.Bool("print-admin-token", false, "print the admin Bearer token and exit (generates if absent, mode 0600 at ~/Lethean/data/admin.token)")
	rotateAdminToken := fs.Bool("rotate-admin-token", false, "regenerate the admin Bearer token, print it, and exit")
	stateConversations := fs.Bool("state-conversations", true, "conversation continuity: wake each chat from its slept state, append only the new turn, sleep after — no prompt replay (disable with -state-conversations=false)")
	stateStorePath := fs.String("state-store", "", "conversation state store file (default ~/Lethean/data/state/conversations.kv)")
	fs.Usage = func() {
		name := cliName()
		core.WriteString(stderr, core.Sprintf("Usage: %s serve [--model <path>] [flags]\n", name))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Host an OpenAI / Anthropic / Ollama-compatible HTTP API for a model.\n")
		core.WriteString(stderr, "Default port 36911 is Lethean's own — an Ollama install on 11434 never collides.\n")
		core.WriteString(stderr, "Ollama-compatible clients just point at this address instead.\n")
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
		core.WriteString(stderr, core.Sprintf("  %s serve --model ~/models/lemer-lite\n", name))
		core.WriteString(stderr, core.Sprintf("    # default OpenAI HTTP on :36911, model loaded at startup\n"))
		core.WriteString(stderr, core.Sprintf("  %s serve --model ~/models/lemer-lite --addr 127.0.0.1:8080\n", name))
		core.WriteString(stderr, core.Sprintf("    # loopback-only, custom port\n"))
		core.WriteString(stderr, core.Sprintf("  %s serve --model ~/models/lemer-lite --context 8192\n", name))
		core.WriteString(stderr, core.Sprintf("    # cap context length to save KV cache memory\n"))
		core.WriteString(stderr, core.Sprintf("  %s serve --model ~/models/gemma-4-e2b-it-4bit --context 16384 -kv-cache paged\n", name))
		core.WriteString(stderr, core.Sprintf("    # fixed-cache regime: activates the compiled+pipelined decode lane\n"))
		core.WriteString(stderr, core.Sprintf("  %s serve --model ~/models/gemma-4-31b-it-q4\n", name))
		core.WriteString(stderr, core.Sprintf("    # a drafter beside a Gemma 4 target (assistant/ pair layout, MTP/ gguf)\n"))
		core.WriteString(stderr, core.Sprintf("    # is detected and engaged automatically — --draft '' opts out\n"))
		core.WriteString(stderr, core.Sprintf("  %s serve --model ~/models/gemma-4-e2b-it-6bit --draft ~/models/gemma-4-E2B-it-assistant-bf16 --draft-block 5\n", name))
		core.WriteString(stderr, core.Sprintf("    # native Gemma-4 MTP speculative decode, drafter + block chosen explicitly\n"))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Inference routes (all relative to the listen address):\n")
		core.WriteString(stderr, "  POST /v1/chat/completions    OpenAI chat (streaming + non-streaming)\n")
		core.WriteString(stderr, "  POST /v1/completions         OpenAI legacy completion\n")
		core.WriteString(stderr, "  POST /v1/messages            Anthropic Messages\n")
		core.WriteString(stderr, "  POST /api/chat               Ollama chat\n")
		core.WriteString(stderr, "  GET  /v1/models              list loaded models\n")
		core.WriteString(stderr, "  GET  /v1/health              process health probe\n")
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Admin routes (Bearer auth required — see --print-admin-token):\n")
		core.WriteString(stderr, "  GET  /v1/admin/machine        current machine identity (hash + runtime)\n")
		core.WriteString(stderr, "  GET  /v1/admin/serve/status   snapshot of model + applied config\n")
		core.WriteString(stderr, "  POST /v1/admin/models/download    HF download into ~/Lethean/data/models/ (allowlist-gated)\n")
		core.WriteString(stderr, "  GET  /v1/admin/models/download?job=ID  poll a download job\n")
		core.WriteString(stderr, "  POST /v1/admin/serve/reload       hot-swap loaded model (confirmation + sha-manifest gated)\n")
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Admin token (auto-managed):\n")
		core.WriteString(stderr, "  Stored at ~/Lethean/data/admin.token (mode 0600), generated on first\n")
		core.WriteString(stderr, "  serve boot. Reveal with `lthn-mlx serve --print-admin-token` (note this\n")
		core.WriteString(stderr, "  prints to stderr — survives in shell scrollback + launchctl logs; for\n")
		core.WriteString(stderr, "  safer capture use `pbcopy < ~/Lethean/data/admin.token`).\n")
		core.WriteString(stderr, "  Rotate with `--rotate-admin-token`. Rotation does NOT live-reload —\n")
		core.WriteString(stderr, "  restart any running serve for the new token to take effect.\n")
		core.WriteString(stderr, "  Send as:\n")
		core.WriteString(stderr, "    curl -H 'Authorization: Bearer <token>' http://127.0.0.1:36911/v1/admin/machine\n")
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}

	// Token-management subcommands — handled BEFORE the --model check
	// so operators can reveal / rotate without a model loaded.
	tokenPath := standardAdminTokenPath()
	if *rotateAdminToken {
		tok, err := generateAdminToken()
		if err != nil {
			core.Print(stderr, "%s serve: token rotation failed: %v", cliName(), err)
			return 1
		}
		if err := writeAdminToken(tokenPath, tok); err != nil {
			core.Print(stderr, "%s serve: token write failed: %v", cliName(), err)
			return 1
		}
		core.Print(stderr, "%s admin token (rotated):\n  %s\n  saved to %s (mode 0600)\n  any running serve still holds the old token — restart to apply", cliName(), tok, tokenPath)
		return 0
	}
	if *printAdminToken {
		tok, generated, err := ensureAdminToken(tokenPath)
		if err != nil {
			core.Print(stderr, "%s serve: token init failed: %v", cliName(), err)
			return 1
		}
		label := "loaded"
		if generated {
			label = "newly generated"
		}
		core.Print(stderr, "%s admin token (%s):\n  %s\n  at %s (mode 0600)", cliName(), label, tok, tokenPath)
		return 0
	}

	// --model is optional. An empty path starts the driver model-less: it
	// binds the listener + /v1/admin surface immediately and waits for a
	// model via POST /v1/admin/serve/reload. Inference calls return "no
	// model loaded" until one arrives. This is the crew/fleet boot path —
	// the supervisor brings the engine up and the app loads a model on
	// demand. A non-empty --model keeps the eager-bind, lazy-first-load
	// behaviour below.
	modelless := core.Trim(*modelPath) == ""
	if modelless {
		core.Print(stderr, "%s serve: starting model-less — POST /v1/admin/serve/reload to load a model", cliName())
	}

	// Admin token — load existing or generate fresh. Fail-closed:
	// if the token file can't be written, serve refuses to boot
	// rather than binding a listener with an unprotected admin
	// surface (Cerberus DREAD §5.1).
	adminToken, generated, err := ensureAdminToken(tokenPath)
	if err != nil {
		core.Print(stderr, "%s serve: admin token init failed (fail-closed): %v", cliName(), err)
		return 1
	}
	if generated {
		core.Print(stderr, "%s serve: fresh admin token generated at %s — run `%s serve --print-admin-token` to reveal", cliName(), tokenPath, cliName())
	}

	// Serve derives load config from the model's own declarations plus
	// explicit flags — there is no tuned-profile layer. --context is the
	// one load override; everything else comes from the model at load time.
	mlxOpts := []mlx.LoadOption{}
	var statusConfig adminServeStatusConfig
	if *contextLen > 0 {
		mlxOpts = append(mlxOpts, mlx.WithContextLength(*contextLen))
		statusConfig.ContextLength = *contextLen
	}
	if mode, ok := parseRuntimeCacheMode(*kvCacheMode); ok {
		if !isRuntimeCacheMode(mode) {
			core.Print(stderr, "%s serve: unknown -kv-cache mode %q", cliName(), *kvCacheMode)
			return 2
		}
		mlxOpts = append(mlxOpts, mlx.WithKVCacheMode(mode))
		statusConfig.CacheMode = string(mode)
	}

	// Reactive MTP pair resolution (the model declares, the engine reacts):
	// an explicit --draft path wins; --draft="" disables; the default "auto"
	// runs the detection ladder over Gemma 4 targets only.
	detection := resolveServeDraft(*modelPath, *draftPath, *draftDetect)
	hotSwap := newHotSwapResolver(*modelPath, detection.DraftPath, *draftBlock, mlxOpts)
	// Conversation continuity is on by default — the serve IS the state
	// product. Any failure here degrades to stateless serving with an honest
	// notice; it never blocks the serve from coming up.
	if *stateConversations {
		storePath := core.Trim(*stateStorePath)
		if storePath == "" {
			if homeR := core.UserHomeDir(); homeR.OK {
				home, _ := homeR.Value.(string)
				storePath = core.PathJoin(home, "Lethean", "data", "state", "conversations.kv")
			}
		}
		var store *filestore.Store
		if storePath != "" {
			if opened, storeErr := openOrCreateStateStore(ctx, storePath); storeErr == nil {
				store = opened
			} else {
				core.Print(stderr, "%s serve: conversation state store %s: %v", cliName(), storePath, storeErr)
			}
		}
		if store == nil {
			core.Print(stderr, "%s serve: conversation continuity unavailable — serving stateless", cliName())
		} else {
			hotSwap.setOnLoad(func(tm inference.TextModel) {
				if _, err := mlx.EnableConversationContinuity(tm, mlx.ConversationContinuityOptions{Store: store}); err != nil {
					core.Print(stderr, "%s serve: conversation continuity unavailable (stateless serving continues): %v", cliName(), err)
					return
				}
				core.Print(stderr, "%s serve: conversation continuity ON — chats wake from %s, no prompt replay (disable with -state-conversations=false)", cliName(), storePath)
			})
		}
	}
	admin := openai.AdminConfig{
		Health: func(_ context.Context) (openai.Health, error) {
			// Report the currently-loaded model (post-reload), or no
			// models when the driver started model-less and none has
			// been loaded yet.
			models := []string{}
			if p := hotSwap.CurrentPath(); p != "" {
				models = append(models, p)
			}
			return openai.Health{
				Status:  "ok",
				Runtime: "go-mlx",
				Models:  models,
				Time:    time.Now().Unix(),
			}, nil
		},
	}
	openaiMux := openai.NewMuxWithAdmin(hotSwap.openaiResolver(), admin)

	// Compose the OpenAI/Anthropic/Ollama compatibility surface with
	// the /v1/admin/* admin API. http.ServeMux uses longest-prefix
	// match, so /v1/admin/ routes hit the admin handlers and everything
	// else falls through to the openai mux. See admin.go for the
	// admin endpoint surface (machine / profiles / auto-tune / etc).
	// Snapshot the effective config at boot for /v1/admin/serve/status.
	// Captured once so the response reflects what actually got applied
	// after profile resolution + --context override, not recomputed per
	// request (and resilient if profile files mutate post-boot).
	serveStatus := adminServeStatus{
		ModelPath:    *modelPath,
		Runtime:      adminRuntimeMetal,
		LoadedAtUnix: time.Now().Unix(),
		Config:       statusConfig,
	}

	rootMux := http.NewServeMux()
	rootMux.Handle("/v1/admin/", newAdminMux(ctx, adminMuxConfig{
		Stderr:      stderr,
		ServeStatus: serveStatus,
		Resolver:    hotSwap,
	}))
	rootMux.Handle("/", openaiMux)

	// Bearer auth on /v1/admin/* only — inference paths pass through.
	// Middleware mounted at rootMux per Cerberus DREAD §5.3 (mounting
	// it inside openaiMux instead would leave admin handlers
	// unauthenticated by composition order).
	srv := &http.Server{
		Addr:              *addr,
		Handler:           requireBearerOnAdmin(rootMux, adminToken, stderr),
		ReadHeaderTimeout: *readTimeout,
		WriteTimeout:      *writeTimeout,
	}

	if notice := speculativeServeNotice(detection, *draftBlock); notice != "" {
		core.Print(stderr, "%s serve: %s", cliName(), notice)
	}
	core.Print(stderr, "%s serve: listening on %s (model=%s)", cliName(), *addr, *modelPath)

	errCh := make(chan error, 1)
	go func() {
		err := srv.ListenAndServe()
		if err != nil && err != http.ErrServerClosed {
			errCh <- err
			return
		}
		errCh <- nil
	}()

	select {
	case err := <-errCh:
		if err != nil {
			core.Print(stderr, "%s serve: listen failed: %v", cliName(), err)
			return 1
		}
		return 0
	case <-ctx.Done():
		shutdownCtx, cancel := context.WithTimeout(context.Background(), *shutdownTimeout)
		defer cancel()
		if err := srv.Shutdown(shutdownCtx); err != nil {
			core.Print(stderr, "%s serve: shutdown error: %v", cliName(), err)
			return 1
		}
		return 0
	}
}

// resolveServeDraft turns the serve flags into a drafter decision: "auto"
// (the default) runs the reactive detection ladder, an explicit path forces
// the drafter, and "" (or -draft-detect=false) stands the ladder down.
func resolveServeDraft(modelPath, draftFlag string, detect bool) mlx.DraftDetection {
	explicit := ""
	opts := mlx.DraftDetectOptions{Disabled: !detect}
	switch trimmed := core.Trim(draftFlag); trimmed {
	case "auto":
		// ladder decides
	case "":
		opts.Disabled = true
	default:
		explicit = trimmed
	}
	return mlx.DetectGemma4DraftPath(modelPath, explicit, opts)
}

// speculativeServeNotice reports the ACTIVE MTP pair at boot: which drafter
// engaged, which ladder rung chose it, and the draft block the verify
// forwards will run. Drafterless serves print nothing extra. The pair loads
// lazily with the model — a drafter that fails to load surfaces on the first
// request, not here.
//
//	if notice := speculativeServeNotice(detection, *draftBlock); notice != "" {
//	    core.Print(stderr, "%s serve: %s", cliName(), notice)
//	}
func speculativeServeNotice(detection mlx.DraftDetection, draftBlock int) string {
	if !detection.Active() {
		return ""
	}
	if draftBlock <= 0 {
		draftBlock = mlx.MTPDefaultDraftBlock
	}
	return core.Sprintf("MTP speculative decode ACTIVE — drafter %s (%s), block %d; greedy and sampled requests ride the verified lane, repetition-penalty/probe requests fall back to plain decode",
		detection.DraftPath, detection.Note, draftBlock)
}
