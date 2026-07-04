// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"io"
	"net/http"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/score"
)

// POST /v1/score — the lem-scorer over a (prompt, response) chat pair.
//
// Pure text compute: no model, no files, no engine state — so it mounts
// beside the inference paths without bearer auth and works even while the
// model is still lazy-loading. The GUI scores the pair it is displaying;
// the LoRA cascade oracle (#50) reuses the same scorer in-process.
//
//	curl -s http://127.0.0.1:36911/v1/score \
//	  -H 'content-type: application/json' \
//	  -d '{"prompt":"explain your reasoning","response":"you are absolutely right"}'
//
// Response: the score.DiffResult wire shape — prompt + response ScoreResults
// (sycophancy tier, LEK 0..100 composite, hostility, grammar+phonetic
// imprint) plus the cross-text differential (echo/shift) and authority read.
type scoreRequest struct {
	Prompt   string `json:"prompt"`
	Response string `json:"response"`
}

// scoreBodyLimit caps the request body. An 8K-token gemma answer serialises
// well under 100KB; the cap exists to bound adversarial POSTs, not real chat.
const scoreBodyLimit = 256 * 1024

func handleScorePair(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("allow", http.MethodPost)
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{
			"error": "use POST with {\"prompt\":..., \"response\":...}",
		})
		return
	}
	defer r.Body.Close()
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, scoreBodyLimit))
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{
			"error": "body unreadable or over 256KB",
		})
		return
	}
	var req scoreRequest
	if res := core.JSONUnmarshal(body, &req); !res.OK {
		writeJSON(w, http.StatusBadRequest, map[string]string{
			"error": "invalid JSON: expected {\"prompt\":string, \"response\":string}",
		})
		return
	}
	writeJSON(w, http.StatusOK, score.ScorePair(req.Prompt, req.Response))
}
