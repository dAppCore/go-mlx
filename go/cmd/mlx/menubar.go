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
	"io/fs"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/mlx/openai"
	"github.com/wailsapp/wails/v3/pkg/application"
)

// menubarPrefs persists user choices across launches so the tray
// picks up where it left off. JSON at the macOS-conventional
// Application Support path; missing file = empty zero-value prefs.
type menubarPrefs struct {
	Model string `json:"model,omitempty"`
}

func prefsPath() string {
	return core.PathJoin(core.Env("HOME"), "Library", "Application Support", "lthn-mlx", "preferences.json")
}

func loadPrefs() menubarPrefs {
	var p menubarPrefs
	data := core.ReadFile(prefsPath())
	if !data.OK {
		return p
	}
	raw, _ := data.Value.([]byte)
	if r := core.JSONUnmarshal(raw, &p); !r.OK {
		return menubarPrefs{}
	}
	return p
}

func savePrefs(p menubarPrefs) {
	dir := core.PathJoin(core.Env("HOME"), "Library", "Application Support", "lthn-mlx")
	_ = core.MkdirAll(dir, 0o755)
	encoded := core.JSONMarshal(p)
	if !encoded.OK {
		return
	}
	raw, _ := encoded.Value.([]byte)
	_ = core.WriteFile(prefsPath(), raw, 0o644)
}

//go:embed assets/tray.png assets/app-icon.png
var menubarAssets embed.FS

// frontendDist embeds the lthn/desktop Vite-built frontend. Copied
// into go/cmd/mlx/frontend/dist/ at build time by
// scripts/make-app-bundle.sh — the lthn/desktop frontend repo is the
// single source of truth. Surfaces that depend on lthn-desktop-only
// services won't function from inside lthn-mlx; the lemma surface
// (added in lthn/desktop/frontend/src/lit/ext/lemma-window.ts) is
// purpose-built to use only the OpenAI HTTP endpoints lthn-mlx exposes.
//
//go:embed all:frontend/dist
var frontendDist embed.FS

// isInsideAppBundle returns true when this binary is running inside a
// macOS .app bundle (as set by the Info.plist bundle identifier). The
// CLI dispatch uses this to choose the default subcommand: menubar when
// launched from Finder, help when invoked from a terminal flat.
func isInsideAppBundle() bool {
	return bool(C.mlx_go_is_inside_app_bundle())
}

// menubarState tracks the serve lifecycle for the menubar's start/stop
// menu items. Atomic Bool covers concurrent access from the UI thread
// (tray clicks) and the server goroutine. lastErr surfaces ListenAndServe
// failures (port in use, etc) back into the status line.
type menubarState struct {
	mu      sync.Mutex
	serving atomic.Bool
	server  *http.Server
	model   string
	addr    string
	lastErr string
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
	prefs := loadPrefs()
	state := &menubarState{
		model: pickModelPath(prefs),
		addr:  ":11434",
	}

	appIcon, _ := menubarAssets.ReadFile("assets/app-icon.png")
	trayIcon, _ := menubarAssets.ReadFile("assets/tray.png")

	frontendFS, _ := fs.Sub(frontendDist, "frontend/dist")

	app := application.New(application.Options{
		Name:        "lthn-mlx",
		Description: "Lethean Lemma — local AI engine",
		Icon:        appIcon,
		Mac: application.MacOptions{
			ActivationPolicy: application.ActivationPolicyAccessory,
			// Without this Wails quits when the lemma window closes —
			// but the tray IS the app's lifetime anchor, not any window.
			ApplicationShouldTerminateAfterLastWindowClosed: false,
		},
		Assets: application.AssetOptions{
			Handler: application.BundledAssetFileServer(frontendFS),
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
	chooseItem := menu.Add("Choose model…")

	menu.AddSeparator()
	startItem := menu.Add("Start serve")
	stopItem := menu.Add("Stop serve")
	stopItem.SetEnabled(false)

	menu.AddSeparator()
	lemmaWindowItem := menu.Add("Open Lemma window")
	openItem := menu.Add("Open endpoint in browser")
	copyItem := menu.Add("Copy endpoint URL")

	menu.AddSeparator()
	quitItem := menu.Add("Quit lthn-mlx")

	refresh := func() {
		modelItem.SetLabel(core.Sprintf("Model: %s", shortPath(state.model)))
		switch {
		case state.serving.Load():
			statusItem.SetLabel(core.Sprintf("Lemma — serving %s", state.addr))
			startItem.SetEnabled(false)
			stopItem.SetEnabled(true)
		case state.lastErr != "":
			statusItem.SetLabel(core.Sprintf("Lemma — failed: %s", state.lastErr))
			startItem.SetEnabled(true)
			stopItem.SetEnabled(false)
		default:
			statusItem.SetLabel("Lemma — idle")
			startItem.SetEnabled(true)
			stopItem.SetEnabled(false)
		}
	}

	chooseItem.OnClick(func(_ *application.Context) {
		dialog := app.Dialog.OpenFile().
			CanChooseDirectories(true).
			CanChooseFiles(false).
			SetTitle("Choose a model directory")
		path, err := dialog.PromptForSingleSelection()
		if err != nil || core.Trim(path) == "" {
			return
		}
		state.mu.Lock()
		state.model = path
		savePrefs(menubarPrefs{Model: path})
		state.mu.Unlock()
		refresh()
	})

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

	// Window opener — mirrors lthn/desktop's openWindowSpec pattern:
	// a frameless lighter-shell window pointing at ?surface=lemma in
	// the embedded frontend. Tray is the lifetime anchor (closing the
	// window doesn't quit the app, only the Quit menu item does).
	var lemmaWindow application.Window
	lemmaWindowItem.OnClick(func(_ *application.Context) {
		if lemmaWindow != nil {
			lemmaWindow.Show()
			lemmaWindow.Focus()
			return
		}
		lemmaWindow = app.Window.NewWithOptions(application.WebviewWindowOptions{
			Name:             "lemma",
			Title:            "Lemma",
			Width:            720,
			Height:           480,
			MinWidth:         480,
			MinHeight:        360,
			Frameless:        true,
			URL:              "/?surface=lemma",
			BackgroundColour: application.NewRGBA(0, 0, 0, 0),
			Mac: application.MacWindow{
				InvisibleTitleBarHeight: 40,
			},
		})
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
	state.lastErr = ""
	state.serving.Store(true)

	go func() {
		err := srv.ListenAndServe()
		state.mu.Lock()
		state.serving.Store(false)
		if err != nil && err != http.ErrServerClosed {
			state.lastErr = err.Error()
		}
		state.mu.Unlock()
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

// pickModelPath resolves the initial model: saved prefs win, then env
// var, then the default lemer-lite snapshot. Used at boot only.
func pickModelPath(prefs menubarPrefs) string {
	if core.Trim(prefs.Model) != "" {
		return prefs.Model
	}
	if env := core.Trim(core.Env("LTHN_MLX_MODEL")); env != "" {
		return env
	}
	return core.PathJoin(core.Env("HOME"), ".cache", "huggingface", "hub", "models--lthn--lemer-lite")
}

func shortPath(p string) string {
	if home := core.Env("HOME"); home != "" && len(p) > len(home) && p[:len(home)] == home {
		return "~" + p[len(home):]
	}
	return p
}
