// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package main

/*
#cgo darwin CFLAGS: -x objective-c
#cgo darwin LDFLAGS: -framework Foundation
#import <Foundation/Foundation.h>
#include <stdbool.h>

// Returns true when the running binary is inside a .app bundle —
// detected via NSBundle's bundleIdentifier (set in Info.plist).
// Used to default to the menubar subcommand when launched from
// Finder vs the CLI.
static bool mlx_go_is_inside_app_bundle(void) {
    @autoreleasepool {
        NSBundle *bundle = [NSBundle mainBundle];
        if (bundle == nil) { return false; }
        NSString *identifier = [bundle bundleIdentifier];
        return identifier != nil && [identifier length] > 0;
    }
}
*/
import "C"

import (
	"context"
	"embed"
	"io"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/mlx/openai"
	"github.com/wailsapp/wails/v3/pkg/application"
)

//go:embed assets/tray.png assets/app-icon.png
var menubarAssets embed.FS

// isInsideAppBundle returns true when this binary is running inside a
// macOS .app bundle (as set by the Info.plist bundle identifier). The
// CLI dispatch uses this to choose the default subcommand: menubar when
// launched from Finder, help when invoked from a terminal flat.
func isInsideAppBundle() bool {
	return bool(C.mlx_go_is_inside_app_bundle())
}

// menubarState tracks the serve lifecycle for the menubar's start/stop
// menu items. Atomic Bool covers concurrent access from the UI thread
// (tray clicks) and the server goroutine.
type menubarState struct {
	mu      sync.Mutex
	serving atomic.Bool
	server  *http.Server
	model   string
	addr    string
}

// runMenubarCommand drives the lthn-mlx tray-only macOS app. Wails
// creates the application with accessory activation policy (no Dock
// icon, just the tray). The tray IS the app's lifetime anchor — closing
// would-be windows in a future iteration won't quit the process; only
// the explicit Quit menu item or SIGTERM does.
//
// The serve subcommand's HTTP mux runs in a background goroutine when
// the user clicks Start; menu state reflects the serve lifecycle.
//
//	lthn-mlx menubar                       # explicit invocation
//	# (also the default when Finder launches lthn-mlx.app)
func runMenubarCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	state := &menubarState{
		model: core.Env("LTHN_MLX_MODEL"),
		addr:  ":11434",
	}
	if core.Trim(state.model) == "" {
		// Default to the lemer-lite snapshot if installed locally.
		state.model = core.PathJoin(core.Env("HOME"), ".cache", "huggingface", "hub", "models--lthn--lemer-lite")
	}

	appIcon, _ := menubarAssets.ReadFile("assets/app-icon.png")
	trayIcon, _ := menubarAssets.ReadFile("assets/tray.png")

	app := application.New(application.Options{
		Name:        "lthn-mlx",
		Description: "Lethean Lemma — local AI engine",
		Icon:        appIcon,
		Mac: application.MacOptions{
			ActivationPolicy: application.ActivationPolicyAccessory,
		},
	})

	tray := app.SystemTray.New()
	tray.SetTemplateIcon(trayIcon)
	tray.SetLabel("")

	menu := app.NewMenu()
	statusItem := menu.Add("Lemma — idle")
	statusItem.SetEnabled(false)

	menu.AddSeparator()
	modelItem := menu.Add(core.Sprintf("Model: %s", shortPath(state.model)))
	modelItem.SetEnabled(false)
	addrItem := menu.Add(core.Sprintf("Address: http://localhost%s", state.addr))
	addrItem.SetEnabled(false)

	menu.AddSeparator()
	startItem := menu.Add("Start serve")
	stopItem := menu.Add("Stop serve")
	stopItem.SetEnabled(false)

	menu.AddSeparator()
	openItem := menu.Add("Open endpoint in browser")
	copyItem := menu.Add("Copy endpoint URL")

	menu.AddSeparator()
	quitItem := menu.Add("Quit lthn-mlx")

	refresh := func() {
		if state.serving.Load() {
			statusItem.SetLabel(core.Sprintf("Lemma — serving %s", state.addr))
			startItem.SetEnabled(false)
			stopItem.SetEnabled(true)
		} else {
			statusItem.SetLabel("Lemma — idle")
			startItem.SetEnabled(true)
			stopItem.SetEnabled(false)
		}
	}

	startItem.OnClick(func(_ *application.Context) {
		state.mu.Lock()
		defer state.mu.Unlock()
		if state.serving.Load() {
			return
		}
		startMenubarServe(state, refresh)
		refresh()
	})

	stopItem.OnClick(func(_ *application.Context) {
		state.mu.Lock()
		defer state.mu.Unlock()
		if !state.serving.Load() {
			return
		}
		stopMenubarServe(state)
		refresh()
	})

	endpoint := "http://localhost" + state.addr
	openItem.OnClick(func(_ *application.Context) {
		_ = app.Browser.OpenURL(endpoint + "/v1/health")
	})
	copyItem.OnClick(func(_ *application.Context) {
		_ = app.Clipboard.SetText(endpoint)
	})
	quitItem.OnClick(func(_ *application.Context) {
		state.mu.Lock()
		if state.serving.Load() {
			stopMenubarServe(state)
		}
		state.mu.Unlock()
		app.Quit()
	})

	tray.SetMenu(menu)
	refresh()

	if err := app.Run(); err != nil {
		core.Print(stderr, "lthn-mlx menubar: %v", err)
		return 1
	}
	return 0
}

func startMenubarServe(state *menubarState, refresh func()) {
	loadOpts := []inference.LoadOption{}
	resolver := openai.NewResolver(state.model, loadOpts...)
	admin := openai.AdminConfig{
		Health: func(_ context.Context) (openai.Health, error) {
			return openai.Health{
				Status:  "ok",
				Runtime: "go-mlx-menubar",
				Models:  []string{state.model},
				Time:    time.Now().Unix(),
			}, nil
		},
	}
	mux := openai.NewMuxWithAdmin(resolver, admin)
	srv := &http.Server{
		Addr:              state.addr,
		Handler:           mux,
		ReadHeaderTimeout: 30 * time.Second,
		WriteTimeout:      5 * time.Minute,
	}
	state.server = srv
	state.serving.Store(true)

	go func() {
		_ = srv.ListenAndServe()
		state.serving.Store(false)
		refresh()
	}()
}

func stopMenubarServe(state *menubarState) {
	if state.server != nil {
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = state.server.Shutdown(shutdownCtx)
		state.server = nil
	}
	state.serving.Store(false)
}

func shortPath(p string) string {
	if home := core.Env("HOME"); home != "" && len(p) > len(home) && p[:len(home)] == home {
		return "~" + p[len(home):]
	}
	return p
}
