// SPDX-Licence-Identifier: EUPL-1.2

package m2

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/probe"
)

func TestMiniMaxM2_RouteTokens_Good(t *testing.T) {
	cfg := Config{NumLocalExperts: 4, NumExpertsPerToken: 2, ScoringFunc: "sigmoid", UseRoutingBias: true}

	decisions, err := RouteTokens(cfg, [][]float32{{0, 2, 1, -1}}, []float32{0, 0, 0, 4})
	if err != nil {
		t.Fatalf("RouteTokens() error = %v", err)
	}

	if len(decisions) != 1 || len(decisions[0].ExpertIDs) != 2 {
		t.Fatalf("decisions = %+v, want one top-2 decision", decisions)
	}
	if decisions[0].ExpertIDs[0] != 3 || decisions[0].ExpertIDs[1] != 1 {
		t.Fatalf("expert order = %+v, want bias-boosted expert 3 then expert 1", decisions[0].ExpertIDs)
	}
	if !roughlyEqual32(decisions[0].Weights[0]+decisions[0].Weights[1], 1, 0.0001) {
		t.Fatalf("weights = %+v, want renormalized top-k weights", decisions[0].Weights)
	}
}

func TestMiniMaxM2_DispatchExpertsAndProbes_Good(t *testing.T) {
	hidden := [][]float32{{1, 2}}
	decisions := []RouterDecision{{
		TokenIndex: 0,
		ExpertIDs:  []int{1, 0},
		Weights:    []float32{0.25, 0.75},
	}}
	experts := map[int]ExpertFunc{
		0: func(values []float32) []float32 { return []float32{values[0] * 10, values[1] * 10} },
		1: func(values []float32) []float32 { return []float32{values[0] * 2, values[1] * 2} },
	}

	out, err := DispatchExperts(hidden, decisions, experts)
	if err != nil {
		t.Fatalf("DispatchExperts() error = %v", err)
	}
	if len(out) != 1 || !roughlyEqual32(out[0][0], 8, 0.0001) || !roughlyEqual32(out[0][1], 16, 0.0001) {
		t.Fatalf("out = %+v, want weighted expert sum [8 16]", out)
	}

	events := RouterProbeEvents(3, []int32{42}, decisions)
	if len(events) != 1 || events[0].Kind != probe.KindRouterDecision || events[0].RouterDecision.Layer != 3 {
		t.Fatalf("events = %+v, want router decision probe", events)
	}
	if events[0].RouterDecision.TokenID != 42 || events[0].Meta["architecture"] != "minimax_m2" {
		t.Fatalf("event = %+v, want token id and architecture metadata", events[0])
	}
}

func TestMiniMaxM2_RouteTokens_Bad(t *testing.T) {
	// top-k exceeding the expert count is rejected.
	if _, err := RouteTokens(Config{NumLocalExperts: 2, NumExpertsPerToken: 4}, [][]float32{{0, 1}}, nil); err == nil || !core.Contains(err.Error(), "top-k") {
		t.Fatalf("error = %v, want top-k exceeds expert count", err)
	}
	// Bias length must match the expert count.
	if _, err := RouteTokens(Config{NumLocalExperts: 3, NumExpertsPerToken: 1}, [][]float32{{0, 1, 2}}, []float32{0, 0}); err == nil || !core.Contains(err.Error(), "bias length") {
		t.Fatalf("error = %v, want bias length mismatch", err)
	}
	// A score row whose width != expert count is rejected per-token.
	if _, err := RouteTokens(Config{NumLocalExperts: 3, NumExpertsPerToken: 1}, [][]float32{{0, 1}}, nil); err == nil || !core.Contains(err.Error(), "scores") {
		t.Fatalf("error = %v, want per-row score width mismatch", err)
	}
	// Missing expert count fails before any routing.
	if _, err := RouteTokens(Config{}, [][]float32{{0}}, nil); err == nil || !core.Contains(err.Error(), "local expert count") {
		t.Fatalf("error = %v, want missing local expert count", err)
	}
}

func TestMiniMaxM2_RouteTokens_Ugly(t *testing.T) {
	// All-equal scores with the non-default scoring func selects the identity
	// scorer (exercises scoringFunc default + identityScore) and the
	// total==0 branch leaves weights un-normalised at zero.
	cfg := Config{NumLocalExperts: 3, NumExpertsPerToken: 2, ScoringFunc: "softmax"}
	decisions, err := RouteTokens(cfg, [][]float32{{0, 0, 0}}, nil)
	if err != nil {
		t.Fatalf("RouteTokens() error = %v", err)
	}
	if len(decisions) != 1 || len(decisions[0].ExpertIDs) != 2 {
		t.Fatalf("decisions = %+v, want one top-2 decision", decisions)
	}
	// identityScore(0) == 0 for every expert, so the renormalisation guard
	// (total == 0) leaves both weights at zero rather than dividing by zero.
	if decisions[0].Weights[0] != 0 || decisions[0].Weights[1] != 0 {
		t.Fatalf("weights = %+v, want zero (no NaN) when all scores are zero", decisions[0].Weights)
	}
	// topK defaults to 1 when NumExpertsPerToken is unset.
	dflt := Config{NumLocalExperts: 4, ScoringFunc: "sigmoid"}
	one, err := RouteTokens(dflt, [][]float32{{1, 2, 3, 0}}, nil)
	if err != nil {
		t.Fatalf("RouteTokens(default topK) error = %v", err)
	}
	if len(one[0].ExpertIDs) != 1 || one[0].ExpertIDs[0] != 2 {
		t.Fatalf("default-topK decision = %+v, want single highest-scoring expert 2", one[0])
	}
}

// --- DispatchExperts (Bad/Ugly; Good already covered above) ---

func TestMiniMaxM2_DispatchExperts_Bad(t *testing.T) {
	hidden := [][]float32{{1, 2}}
	experts := map[int]ExpertFunc{0: func(v []float32) []float32 { return v }}

	// Token index out of range.
	if _, err := DispatchExperts(hidden, []RouterDecision{{TokenIndex: 5, ExpertIDs: []int{0}, Weights: []float32{1}}}, experts); err == nil || !core.Contains(err.Error(), "token index") {
		t.Fatalf("error = %v, want token index out of range", err)
	}
	// Expert/weight length mismatch.
	if _, err := DispatchExperts(hidden, []RouterDecision{{TokenIndex: 0, ExpertIDs: []int{0, 1}, Weights: []float32{1}}}, experts); err == nil || !core.Contains(err.Error(), "length mismatch") {
		t.Fatalf("error = %v, want expert/weight length mismatch", err)
	}
	// Missing expert function.
	if _, err := DispatchExperts(hidden, []RouterDecision{{TokenIndex: 0, ExpertIDs: []int{7}, Weights: []float32{1}}}, experts); err == nil || !core.Contains(err.Error(), "missing expert") {
		t.Fatalf("error = %v, want missing expert", err)
	}
	// Expert output shape mismatch against a previously sized output row.
	twoExperts := map[int]ExpertFunc{
		0: func(v []float32) []float32 { return []float32{v[0]} },
		1: func(v []float32) []float32 { return []float32{v[0], v[1]} },
	}
	if _, err := DispatchExperts(hidden, []RouterDecision{{TokenIndex: 0, ExpertIDs: []int{0, 1}, Weights: []float32{0.5, 0.5}}}, twoExperts); err == nil || !core.Contains(err.Error(), "output shape mismatch") {
		t.Fatalf("error = %v, want expert output shape mismatch", err)
	}
}

func TestMiniMaxM2_DispatchExperts_Ugly(t *testing.T) {
	// A decision with no experts is skipped (no arena window consumed) and its
	// output row stays nil. The expert it would have used must never be called.
	called := false
	experts := map[int]ExpertFunc{0: func(v []float32) []float32 { called = true; return v }}
	out, err := DispatchExperts([][]float32{{1, 2}}, []RouterDecision{{TokenIndex: 0, ExpertIDs: nil, Weights: nil}}, experts)
	if err != nil {
		t.Fatalf("DispatchExperts() error = %v", err)
	}
	if len(out) != 1 || out[0] != nil {
		t.Fatalf("out = %+v, want a nil output row for the empty decision", out)
	}
	if called {
		t.Fatal("expert function was called for an empty decision")
	}
}

// --- LoadRouter / ProjectRouterScores error legs ---

func TestMiniMaxM2_ProjectRouterScores_Bad(t *testing.T) {
	// Missing expert/hidden sizes.
	if _, err := ProjectRouterScores([][]float32{{1}}, RouterWeights{}); err == nil || !core.Contains(err.Error(), "expert and hidden sizes") {
		t.Fatalf("error = %v, want expert/hidden size diagnostic", err)
	}
	// Weight length inconsistent with declared shape.
	if _, err := ProjectRouterScores([][]float32{{1, 2}}, RouterWeights{NumExperts: 2, HiddenSize: 2, Weight: []float32{1, 2, 3}}); err == nil || !core.Contains(err.Error(), "weight length") {
		t.Fatalf("error = %v, want router weight length diagnostic", err)
	}
	// Hidden row width does not match the router hidden size.
	if _, err := ProjectRouterScores([][]float32{{1, 2, 3}}, RouterWeights{NumExperts: 1, HiddenSize: 2, Weight: []float32{1, 2}}); err == nil || !core.Contains(err.Error(), "hidden row") {
		t.Fatalf("error = %v, want hidden row width diagnostic", err)
	}
}

// --- LoadPackedExperts / BuildLayerForwardSkeleton error legs ---
