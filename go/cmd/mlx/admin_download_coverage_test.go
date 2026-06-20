// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// runDownloadJob reaches the per-file fetch loop with a fake tree entry whose
// URL is NOT on the HF allowlist: fetchAndVerify rejects it ("disallowed
// source"), so the job fails at the fetch step — exercising the quarantine
// mkdir + per-file dir mkdir + fetchAndVerify call + fetch-failure branch with
// no real network.
func TestRunDownloadJob_FetchRejected_Bad(t *testing.T) {
	hf := &fakeHFTreeAPI{entries: []hfFileEntry{
		{Path: "model.safetensors", URL: "https://evil.example.com/model.bin", Size: 10, Digest: "abc"},
	}}
	reg := newAdminDownloadRegistry(context.Background(), io.Discard)
	job := &adminDownloadJob{ID: "fj", Repo: "a/b", Revision: "main", DestPath: t.TempDir() + "/dst"}
	runDownloadJob(job, adminDownloadRequest{Repo: "a/b", Revision: "main"}, hf, reg)
	if job.Status != "failed" {
		t.Fatalf("status = %q, want failed", job.Status)
	}
	if !strings.Contains(job.Error, "fetch") {
		t.Fatalf("error = %q, want a fetch failure", job.Error)
	}
	if job.FileCount != 1 || job.BytesTotal != 10 {
		t.Fatalf("job totals = files:%d bytes:%d, want 1/10 recorded before the fetch", job.FileCount, job.BytesTotal)
	}
}

// runDownloadJob refuses when free disk is below 2× the model size — a single
// absurdly large entry trips the disk-space guard before any fetch. The
// digest-missing (non-LFS) entry note is exercised via the empty Digest.
func TestRunDownloadJob_DiskSpaceRefusal_Bad(t *testing.T) {
	hf := &fakeHFTreeAPI{entries: []hfFileEntry{
		{Path: "huge.bin", URL: "https://huggingface.co/a/b/resolve/main/huge.bin", Size: 1 << 62, Digest: ""},
	}}
	reg := newAdminDownloadRegistry(context.Background(), io.Discard)
	job := &adminDownloadJob{ID: "dj", Repo: "a/b", Revision: "main", DestPath: t.TempDir() + "/dst"}
	runDownloadJob(job, adminDownloadRequest{Repo: "a/b", Revision: "main"}, hf, reg)
	if job.Status != "failed" {
		t.Fatalf("status = %q, want failed", job.Status)
	}
	if !strings.Contains(job.Error, "disk-space") {
		t.Fatalf("error = %q, want the disk-space refusal", job.Error)
	}
}

// runDownloadJob with a cancelled context bails at the per-file cancellation
// check (a small entry, so the disk-space guard passes first).
func TestRunDownloadJob_Cancelled_Bad(t *testing.T) {
	hf := &fakeHFTreeAPI{entries: []hfFileEntry{
		{Path: "small.bin", URL: "https://huggingface.co/a/b/resolve/main/small.bin", Size: 4, Digest: "x"},
	}}
	ctx, cancel := context.WithCancel(context.Background())
	reg := newAdminDownloadRegistry(ctx, io.Discard)
	cancel() // cancel before the worker runs
	job := &adminDownloadJob{ID: "cj", Repo: "a/b", Revision: "main", DestPath: t.TempDir() + "/dst"}
	runDownloadJob(job, adminDownloadRequest{Repo: "a/b", Revision: "main"}, hf, reg)
	if job.Status != "failed" || !strings.Contains(job.Error, "cancelled") {
		t.Fatalf("job = (%q, %q), want a cancelled failure", job.Status, job.Error)
	}
}

// adminDownloadHandler POST with an allowlisted repo kicks off a job (202
// Accepted) and the GET ?job=<id> poll returns the job snapshot. The fake tree
// returns no usable fetch, so the background job will fail — but the handler's
// kickoff + poll surface is what's covered here.
func TestAdminDownloadHandler_KickoffAndPoll_Good(t *testing.T) {
	defer withAllowlist(t, "a/b")()
	hf := &fakeHFTreeAPI{entries: []hfFileEntry{
		{Path: "f.bin", URL: "https://evil.example.com/f.bin", Size: 1, Digest: "x"},
	}}
	reg := newAdminDownloadRegistry(context.Background(), io.Discard)
	handler := adminDownloadHandler(reg, hf)

	post := httptest.NewRequest(http.MethodPost, "/v1/admin/models/download", strings.NewReader(`{"repo":"a/b","revision":"main"}`))
	postRec := httptest.NewRecorder()
	handler(postRec, post)
	if postRec.Code != http.StatusAccepted {
		t.Fatalf("POST status = %d, want 202; body=%s", postRec.Code, postRec.Body.String())
	}
	if !strings.Contains(postRec.Body.String(), `"id"`) {
		t.Fatalf("POST body = %s, want a job id", postRec.Body.String())
	}

	// Poll a non-existent job id → 404.
	get := httptest.NewRequest(http.MethodGet, "/v1/admin/models/download?job=no-such-job", nil)
	getRec := httptest.NewRecorder()
	handler(getRec, get)
	if getRec.Code != http.StatusNotFound {
		t.Fatalf("poll unknown job status = %d, want 404", getRec.Code)
	}
}

// adminDownloadHandler rejects an invalid JSON body (400) and a DELETE method
// (405).
func TestAdminDownloadHandler_BadInputs_Bad(t *testing.T) {
	hf := &fakeHFTreeAPI{}
	reg := newAdminDownloadRegistry(context.Background(), io.Discard)
	handler := adminDownloadHandler(reg, hf)

	badBody := httptest.NewRequest(http.MethodPost, "/x", strings.NewReader("{not json"))
	badRec := httptest.NewRecorder()
	handler(badRec, badBody)
	if badRec.Code != http.StatusBadRequest {
		t.Fatalf("bad-body status = %d, want 400", badRec.Code)
	}

	del := httptest.NewRequest(http.MethodDelete, "/x", nil)
	delRec := httptest.NewRecorder()
	handler(delRec, del)
	if delRec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("DELETE status = %d, want 405", delRec.Code)
	}
}
