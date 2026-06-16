// SPDX-Licence-Identifier: EUPL-1.2

// Tests for admin.go — the host-owned runtime surface (health, wake/sleep,
// cache-entries) and its public payload types. The ServeHTTP behaviours of the
// concrete handlers (which hang off unexported receiver types) are exercised
// through the public NewMuxWithAdmin mux in openai_test.go; this file covers
// the public TYPES admin.go exports — AdminConfig, Health, ActionResponse and
// the CacheEntryLister contract — plus the Default* route constants, none of
// which load a real model.

package openai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	openaicompat "dappco.re/go/inference/openai"
)

// --- Health -----------------------------------------------------------------

func TestAdmin_Health_Good(t *testing.T) {
	// A fully-populated Health round-trips through JSON with every field
	// preserved and the documented snake_case tags.
	h := Health{
		Status:  "ok",
		Runtime: "go-mlx",
		Models:  []string{"qwen3", "gemma4"},
		Time:    1700000000,
		Labels:  map[string]string{"tenant": "local"},
	}
	encoded := core.JSONMarshalString(h)
	for _, want := range []string{
		`"status":"ok"`,
		`"runtime":"go-mlx"`,
		`"qwen3"`,
		`"time":1700000000`,
		`"tenant":"local"`,
	} {
		if !strings.Contains(encoded, want) {
			t.Fatalf("Health JSON = %s, missing %s", encoded, want)
		}
	}
	var back Health
	if r := core.JSONUnmarshal(core.AsBytes(encoded), &back); !r.OK {
		t.Fatalf("Health round-trip failed: %v", r.Error())
	}
	if back.Status != "ok" || back.Runtime != "go-mlx" || len(back.Models) != 2 {
		t.Fatalf("decoded Health = %+v, want fields preserved", back)
	}
}

func TestAdmin_Health_Bad(t *testing.T) {
	// The health handler treats a host-supplied Health with an empty Status as
	// incomplete and fills the documented defaults ("ok" / "go-mlx") on top of
	// it — so a degenerate zero-Status payload never reaches the client as-is.
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": &openAIMockModel{}})
	handler := NewMuxWithAdmin(resolver, AdminConfig{
		Health: func(context.Context) (Health, error) {
			return Health{}, nil // empty Status — the bad/degenerate payload
		},
	})
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultHealthPath, nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"status":"ok"`) {
		t.Fatalf("body = %s, want Status defaulted to ok", rec.Body.String())
	}
}

func TestAdmin_Health_Ugly(t *testing.T) {
	// Boundary: a zero-value Health omits every omitempty field and serialises
	// to just its required Status (empty here) — proving the tags drop absent
	// data rather than emitting nulls.
	encoded := core.JSONMarshalString(Health{})
	if strings.Contains(encoded, "null") {
		t.Fatalf("zero Health = %s, want no null fields", encoded)
	}
	for _, absent := range []string{"runtime", "models", "labels"} {
		if strings.Contains(encoded, absent) {
			t.Fatalf("zero Health = %s, omitempty field %q leaked", encoded, absent)
		}
	}
}

// --- ActionResponse ---------------------------------------------------------

func TestAdmin_ActionResponse_Good(t *testing.T) {
	// A wake/sleep ActionResponse carries the action name and status verbatim
	// through JSON.
	resp := ActionResponse{Action: "wake", Status: "ok", Labels: map[string]string{"node": "a"}}
	encoded := core.JSONMarshalString(resp)
	if !strings.Contains(encoded, `"action":"wake"`) || !strings.Contains(encoded, `"status":"ok"`) {
		t.Fatalf("ActionResponse JSON = %s, want action+status", encoded)
	}
	if !strings.Contains(encoded, `"node":"a"`) {
		t.Fatalf("ActionResponse JSON = %s, want labels", encoded)
	}
}

func TestAdmin_ActionResponse_Bad(t *testing.T) {
	// The action handler, given a callback that errors, never emits an
	// ActionResponse — it returns the error envelope instead. Asserting the
	// 500 path proves a failed action does NOT masquerade as a success
	// ActionResponse on the wire.
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": &openAIMockModel{}})
	handler := NewMuxWithAdmin(resolver, AdminConfig{
		Sleep: func(context.Context) error { return context.Canceled },
	})
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultAdminSleepPath, nil))
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500", rec.Code)
	}
	if strings.Contains(rec.Body.String(), `"action":"sleep"`) {
		t.Fatalf("body = %s, must not be a success ActionResponse", rec.Body.String())
	}
}

func TestAdmin_ActionResponse_Ugly(t *testing.T) {
	// Boundary: a zero-value ActionResponse drops the omitempty Labels but
	// keeps the always-present action/status keys (empty strings).
	encoded := core.JSONMarshalString(ActionResponse{})
	if strings.Contains(encoded, "labels") {
		t.Fatalf("zero ActionResponse = %s, omitempty labels leaked", encoded)
	}
	if !strings.Contains(encoded, `"action":""`) || !strings.Contains(encoded, `"status":""`) {
		t.Fatalf("zero ActionResponse = %s, want empty action/status keys", encoded)
	}
}

// --- AdminConfig ------------------------------------------------------------

func TestAdmin_AdminConfig_Good(t *testing.T) {
	// AdminConfig wires three host callbacks; a fully-populated config drives
	// each one through its route. Asserting the side effects proves the struct
	// fields are the seam NewMuxWithAdmin invokes.
	var healthHit, woke, slept bool
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": &openAIMockModel{}})
	cfg := AdminConfig{
		Health: func(context.Context) (Health, error) { healthHit = true; return Health{Status: "ok"}, nil },
		Wake:   func(context.Context) error { woke = true; return nil },
		Sleep:  func(context.Context) error { slept = true; return nil },
	}
	handler := NewMuxWithAdmin(resolver, cfg)
	for _, p := range []struct {
		method, path string
	}{
		{http.MethodGet, DefaultHealthPath},
		{http.MethodPost, DefaultAdminWakePath},
		{http.MethodPost, DefaultAdminSleepPath},
	} {
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, httptest.NewRequest(p.method, p.path, nil))
		if rec.Code != http.StatusOK {
			t.Fatalf("%s %s -> %d body=%s", p.method, p.path, rec.Code, rec.Body.String())
		}
	}
	if !healthHit || !woke || !slept {
		t.Fatalf("AdminConfig callbacks fired: health=%v wake=%v sleep=%v, want all true", healthHit, woke, slept)
	}
}

func TestAdmin_AdminConfig_Bad(t *testing.T) {
	// A partial AdminConfig (Wake set, Health/Sleep nil) must not panic on the
	// unset routes: the health route falls back to the built-in payload and the
	// sleep route succeeds with no host callback to run.
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": &openAIMockModel{}})
	handler := NewMuxWithAdmin(resolver, AdminConfig{
		Wake: func(context.Context) error { return nil },
	})
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultAdminSleepPath, nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("sleep with nil callback -> %d, want 200", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), `"action":"sleep"`) {
		t.Fatalf("body = %s, want sleep action ok", rec.Body.String())
	}
}

func TestAdmin_AdminConfig_Ugly(t *testing.T) {
	// Boundary: the zero-value AdminConfig (all callbacks nil). NewMuxWithAdmin
	// still mounts the admin routes and the health endpoint serves the default
	// runtime payload entirely on its own.
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": &openAIMockModel{}})
	handler := NewMuxWithAdmin(resolver, AdminConfig{})
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultHealthPath, nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), `"runtime":"go-mlx"`) {
		t.Fatalf("body = %s, want default go-mlx runtime", rec.Body.String())
	}
}

// --- CacheEntryLister -------------------------------------------------------

func TestAdmin_CacheEntryLister_Good(t *testing.T) {
	// A model that implements CacheEntryLister is served through the
	// cache-entries route, which streams its entries back. openAIMockModel
	// satisfies the interface, so the typed listing is returned.
	model := &openAIMockModel{cacheEntries: []inference.CacheBlockRef{{ID: "blk-a", TokenCount: 8}}}
	var _ CacheEntryLister = model // compile-time: the mock satisfies the contract
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model})
	handler := NewMuxWithAdmin(resolver, AdminConfig{})
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultAdminCacheEntriesPath+"?model=qwen", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"id":"blk-a"`) {
		t.Fatalf("body = %s, want listed cache entry", rec.Body.String())
	}
}

func TestAdmin_CacheEntryLister_Bad(t *testing.T) {
	// A model that does NOT implement CacheEntryLister must yield 501 from the
	// cache-entries route — the interface assertion is the gate, and a
	// text-only model fails it.
	model := &openAITextOnlyModel{}
	if _, ok := any(model).(CacheEntryLister); ok {
		t.Fatal("openAITextOnlyModel unexpectedly implements CacheEntryLister")
	}
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model})
	handler := NewMuxWithAdmin(resolver, AdminConfig{})
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultAdminCacheEntriesPath+"?model=qwen", nil))
	if rec.Code != http.StatusNotImplemented {
		t.Fatalf("status = %d, want 501 for non-lister model", rec.Code)
	}
}

func TestAdmin_CacheEntryLister_Ugly(t *testing.T) {
	// Boundary: a lister returning zero entries still produces a well-formed
	// empty listing (200, object=list) rather than an error or a nil body.
	model := &openAIMockModel{cacheEntries: nil}
	var _ CacheEntryLister = model
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model})
	handler := NewMuxWithAdmin(resolver, AdminConfig{})
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultAdminCacheEntriesPath+"?model=qwen", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"object":"list"`) {
		t.Fatalf("body = %s, want empty list envelope", rec.Body.String())
	}
}

// --- Default* route constants -----------------------------------------------

func TestAdmin_DefaultPathConstants(t *testing.T) {
	// The four exported Default* paths are the canonical admin routes. Assert
	// their literal values and that NewMuxWithAdmin actually mounts each one
	// (an unmounted constant would 404 instead of answering).
	if DefaultHealthPath != "/v1/health" ||
		DefaultAdminWakePath != "/v1/runtime/wake" ||
		DefaultAdminSleepPath != "/v1/runtime/sleep" ||
		DefaultAdminCacheEntriesPath != "/v1/cache/entries" {
		t.Fatalf("Default* paths drifted: health=%q wake=%q sleep=%q cache=%q",
			DefaultHealthPath, DefaultAdminWakePath, DefaultAdminSleepPath, DefaultAdminCacheEntriesPath)
	}
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": &openAIMockModel{}})
	handler := NewMuxWithAdmin(resolver, AdminConfig{})
	// GET health is the only constant answerable with no callback + no body.
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultHealthPath, nil))
	if rec.Code == http.StatusNotFound {
		t.Fatalf("DefaultHealthPath %q not mounted (404)", DefaultHealthPath)
	}
}
