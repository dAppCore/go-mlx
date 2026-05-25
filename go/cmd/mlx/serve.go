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
//	lthn-mlx serve --model /Volumes/Data/models/lemer-lite --addr :11434
//	curl http://127.0.0.1:11434/v1/health
//	curl http://127.0.0.1:11434/v1/chat/completions -H 'content-type: application/json' \
//	     -d '{"model":"lemer-lite","messages":[{"role":"user","content":"hi"}]}'
func runServeCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("serve"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	addr := fs.String("addr", ":11434", "listen address (default mirrors Ollama's port)")
	modelPath := fs.String("model", "", "model path to load (required)")
	contextLen := fs.Int("context", 0, "override context length; 0 uses the model's default")
	readTimeout := fs.Duration("read-timeout", 30*time.Second, "HTTP read header timeout")
	writeTimeout := fs.Duration("write-timeout", 5*time.Minute, "HTTP write timeout (covers full streaming response)")
	shutdownTimeout := fs.Duration("shutdown-timeout", 10*time.Second, "graceful shutdown deadline after SIGINT/SIGTERM")
	fs.Usage = func() {
		core.WriteString(stderr, core.Sprintf("Usage: %s serve [flags]\n", cliName()))
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

	if core.Trim(*modelPath) == "" {
		core.Print(stderr, "%s serve: --model is required", cliName())
		fs.Usage()
		return 2
	}

	loadOpts := []inference.LoadOption{}
	if *contextLen > 0 {
		loadOpts = append(loadOpts, inference.WithContextLen(*contextLen))
	}

	resolver := openai.NewResolver(*modelPath, loadOpts...)
	admin := openai.AdminConfig{
		Health: func(_ context.Context) (openai.Health, error) {
			return openai.Health{
				Status:  "ok",
				Runtime: "go-mlx",
				Models:  []string{*modelPath},
				Time:    time.Now().Unix(),
			}, nil
		},
	}
	mux := openai.NewMuxWithAdmin(resolver, admin)

	srv := &http.Server{
		Addr:              *addr,
		Handler:           mux,
		ReadHeaderTimeout: *readTimeout,
		WriteTimeout:      *writeTimeout,
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
