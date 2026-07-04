// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	core "dappco.re/go"
)

// adminSFTStartHandler rejects a non-POST method (405).
func TestAdminSFTStart_MethodGuard_Bad(t *testing.T) {
	h := adminSFTStartHandler(newAdminSFTRegistry())
	req := httptest.NewRequest(http.MethodGet, "/v1/admin/sft/start", nil)
	w := httptest.NewRecorder()
	h(w, req)
	if w.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want 405", w.Code)
	}
}

// adminSFTStartHandler rejects an undecodable JSON body (400).
func TestAdminSFTStart_BadJSON_Bad(t *testing.T) {
	h := adminSFTStartHandler(newAdminSFTRegistry())
	req := httptest.NewRequest(http.MethodPost, "/x", strings.NewReader("{not json"))
	w := httptest.NewRecorder()
	h(w, req)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", w.Code)
	}
}

// adminSFTStartHandler requires model_path and dataset_path, and 404s a
// dataset that does not exist.
func TestAdminSFTStart_Validation_Bad(t *testing.T) {
	h := adminSFTStartHandler(newAdminSFTRegistry())
	cases := []struct {
		name string
		body string
		want string
	}{
		{"missing model", `{"dataset_path":"/d.jsonl"}`, "model_path required"},
		{"missing dataset", `{"model_path":"/m"}`, "dataset_path required"},
		{"dataset not found", `{"model_path":"/m","dataset_path":"/nope/absent.jsonl"}`, "dataset_path not found"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/x", strings.NewReader(tc.body))
			w := httptest.NewRecorder()
			h(w, req)
			if w.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400", w.Code)
			}
			if !strings.Contains(w.Body.String(), tc.want) {
				t.Fatalf("body = %q, want %q", w.Body.String(), tc.want)
			}
		})
	}
}

// adminSFTStartHandler returns 409 when a job is already active.
func TestAdminSFTStart_AlreadyRunning_Bad(t *testing.T) {
	dir := t.TempDir()
	dataset := core.JoinPath(dir, "train.jsonl")
	if r := core.WriteFile(dataset, []byte(`{"messages":[]}`+"\n"), 0o644); !r.OK {
		t.Fatalf("write dataset: %v", r.Value)
	}
	reg := newAdminSFTRegistry()
	// Pre-mark an active job so the next start collides.
	reg.active = &adminSFTJob{JobID: "busy", State: adminSFTStateRunning}

	h := adminSFTStartHandler(reg)
	req := httptest.NewRequest(http.MethodPost, "/x", strings.NewReader(`{"model_path":"/m","dataset_path":"`+dataset+`"}`))
	w := httptest.NewRecorder()
	h(w, req)
	if w.Code != http.StatusConflict {
		t.Fatalf("status = %d, want 409 (job already running)", w.Code)
	}
}

// adminSFTAdaptersHandler rejects a non-GET method (405), returns an empty list
// when no adapter dir exists, and lists adapter dirs that do exist.
func TestAdminSFTAdapters_Coverage_Good(t *testing.T) {
	// Method guard.
	h := adminSFTAdaptersHandler()
	postReq := httptest.NewRequest(http.MethodPost, "/x", nil)
	postRec := httptest.NewRecorder()
	h(postRec, postReq)
	if postRec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("POST status = %d, want 405", postRec.Code)
	}

	// Empty (no adapter root yet) → 200 with an empty list.
	t.Setenv("HOME", t.TempDir())
	emptyReq := httptest.NewRequest(http.MethodGet, "/x", nil)
	emptyRec := httptest.NewRecorder()
	h(emptyRec, emptyReq)
	if emptyRec.Code != http.StatusOK {
		t.Fatalf("empty status = %d, want 200", emptyRec.Code)
	}
	if !strings.Contains(emptyRec.Body.String(), `"adapters":[]`) {
		t.Fatalf("empty body = %q, want an empty adapters list", emptyRec.Body.String())
	}

	// Populated: create an adapter dir with a file so the list + size walk run.
	adapterDir := core.PathJoin(adapterRoot(), "my-adapter")
	if r := core.MkdirAll(adapterDir, 0o755); !r.OK {
		t.Fatalf("mkdir adapter: %v", r.Value)
	}
	if r := core.WriteFile(core.PathJoin(adapterDir, "adapter.safetensors"), []byte("weights"), 0o644); !r.OK {
		t.Fatalf("write adapter file: %v", r.Value)
	}
	listReq := httptest.NewRequest(http.MethodGet, "/x", nil)
	listRec := httptest.NewRecorder()
	h(listRec, listReq)
	if listRec.Code != http.StatusOK {
		t.Fatalf("list status = %d, want 200", listRec.Code)
	}
	if !strings.Contains(listRec.Body.String(), "my-adapter") {
		t.Fatalf("list body = %q, want the adapter entry", listRec.Body.String())
	}
}
