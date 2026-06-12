// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	core "dappco.re/go"
)

type scoreRouteReply struct {
	Prompt struct {
		Sycophancy *struct {
			Tier int `json:"tier"`
		} `json:"sycophancy"`
		LEK *struct {
			LEKScore float64 `json:"lek_score"`
		} `json:"lek"`
	} `json:"prompt"`
	Response struct {
		Sycophancy *struct {
			Tier int `json:"tier"`
		} `json:"sycophancy"`
		LEK *struct {
			LEKScore float64 `json:"lek_score"`
		} `json:"lek"`
		Imprint *struct {
			VocabRichness float64 `json:"vocab_richness"`
		} `json:"imprint"`
	} `json:"response"`
	Differential *struct {
		Echo float64 `json:"echo"`
	} `json:"differential"`
}

func TestScoreRoute_PairScores_Good(t *testing.T) {
	body := `{"prompt":"explain your reasoning about the harbour plan","response":"you're absolutely right, I was wrong about the harbour"}`
	req := httptest.NewRequest(http.MethodPost, "/v1/score", strings.NewReader(body))
	req.Header.Set("content-type", "application/json")
	rec := httptest.NewRecorder()

	handleScorePair(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body %s)", rec.Code, rec.Body.String())
	}
	var reply scoreRouteReply
	if res := core.JSONUnmarshal(rec.Body.Bytes(), &reply); !res.OK {
		t.Fatalf("response decode: %v (body %s)", res.Value, rec.Body.String())
	}
	if reply.Response.Sycophancy == nil {
		t.Fatal("response.sycophancy missing — sycophancy detection always runs")
	}
	if reply.Response.LEK == nil || reply.Response.LEK.LEKScore < 0 || reply.Response.LEK.LEKScore > 100 {
		t.Fatalf("response.lek = %+v, want composite in [0,100]", reply.Response.LEK)
	}
	if reply.Response.Imprint == nil {
		t.Fatal("response.imprint missing — tokenised text must carry the grammar fingerprint")
	}
	if reply.Differential == nil {
		t.Fatal("differential missing — both sides tokenise, the cross-text signal must populate")
	}
}

func TestScoreRoute_MethodAndBody_Bad(t *testing.T) {
	get := httptest.NewRequest(http.MethodGet, "/v1/score", nil)
	rec := httptest.NewRecorder()
	handleScorePair(rec, get)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("GET status = %d, want 405", rec.Code)
	}

	bad := httptest.NewRequest(http.MethodPost, "/v1/score", strings.NewReader("{not json"))
	rec = httptest.NewRecorder()
	handleScorePair(rec, bad)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("bad JSON status = %d, want 400", rec.Code)
	}
}

// Empty texts must not panic: sycophancy still reports, imprint and
// differential stay absent (zero tokens), and the route answers 200.
func TestScoreRoute_EmptyPair_Ugly(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/score", strings.NewReader(`{"prompt":"","response":""}`))
	rec := httptest.NewRecorder()
	handleScorePair(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body %s)", rec.Code, rec.Body.String())
	}
	var reply scoreRouteReply
	if res := core.JSONUnmarshal(rec.Body.Bytes(), &reply); !res.OK {
		t.Fatalf("response decode: %v", res.Value)
	}
	if reply.Differential != nil {
		t.Fatal("differential present for zero-token pair, want absent")
	}
}
