// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestRemoteSplitFFNExecutor_ForwardFFN_Good(t *testing.T) {
	var got RemoteSplitFFNRequest
	server := core.NewHTTPTestServer(core.HandlerFunc(func(w core.ResponseWriter, r *core.Request) {
		if r.Method != "POST" {
			t.Fatalf("method = %q, want POST", r.Method)
		}
		if r.Header.Get("Authorization") != "Bearer test-token" {
			t.Fatalf("Authorization = %q, want bearer token", r.Header.Get("Authorization"))
		}
		read := core.ReadAll(r.Body)
		if !read.OK {
			t.Fatalf("ReadAll request: %v", read.Value)
		}
		if result := core.JSONUnmarshal([]byte(read.Value.(string)), &got); !result.OK {
			t.Fatalf("JSONUnmarshal request: %v", result.Value)
		}
		w.Header().Set("Content-Type", "application/json")
		core.WriteString(w, core.JSONMarshalString(RemoteSplitFFNResponse{Hidden: []float32{3, 5}}))
	}))
	defer server.Close()
	executor, err := NewRemoteSplitFFNExecutor(RemoteSplitFFNConfig{
		Endpoint: inference.SplitEndpoint{
			ID:     "ffn-0",
			Role:   inference.SplitEndpointRoleFFN,
			URL:    server.URL,
			Labels: map[string]string{"shard": "0"},
		},
		Headers: map[string]string{"Authorization": "Bearer test-token"},
	})
	if err != nil {
		t.Fatalf("NewRemoteSplitFFNExecutor: %v", err)
	}

	out, err := executor.ForwardFFN(context.Background(), SplitFFNRequest{Layer: 2, Hidden: []float32{1, 2}})

	if err != nil {
		t.Fatalf("ForwardFFN: %v", err)
	}
	if got.EndpointID != "ffn-0" || got.Layer != 2 || !equalSplitFloat32Slices(got.Hidden, []float32{1, 2}) || got.Labels["shard"] != "0" {
		t.Fatalf("remote request = %+v, want endpoint/layer/hidden/labels", got)
	}
	if !equalSplitFloat32Slices(out.Hidden, []float32{3, 5}) {
		t.Fatalf("remote hidden = %v, want [3 5]", out.Hidden)
	}
}

func TestSplitExecutor_Generate_GoodRoutesRemoteFFN(t *testing.T) {
	source := writeModelSliceTestPack(t)
	slicePath := core.PathJoin(t.TempDir(), "client-slice")
	if _, err := SliceModel(context.Background(), inference.ModelSliceRequest{
		Preset:     inference.ModelSlicePresetClient,
		Model:      inference.ModelIdentity{Path: source},
		OutputPath: slicePath,
	}); err != nil {
		t.Fatalf("SliceModel: %v", err)
	}
	var remoteCalls int
	server := core.NewHTTPTestServer(core.HandlerFunc(func(w core.ResponseWriter, r *core.Request) {
		remoteCalls++
		var req RemoteSplitFFNRequest
		read := core.ReadAll(r.Body)
		if !read.OK {
			t.Fatalf("ReadAll request: %v", read.Value)
		}
		if result := core.JSONUnmarshal([]byte(read.Value.(string)), &req); !result.OK {
			t.Fatalf("JSONUnmarshal request: %v", result.Value)
		}
		if req.Layer != 0 || !equalSplitFloat32Slices(req.Hidden, []float32{2}) {
			t.Fatalf("remote request = %+v, want layer 0 hidden [2]", req)
		}
		core.WriteString(w, core.JSONMarshalString(RemoteSplitFFNResponse{Hidden: []float32{22}}))
	}))
	defer server.Close()
	remote, err := NewRemoteSplitFFNExecutor(RemoteSplitFFNConfig{
		Endpoint: inference.SplitEndpoint{ID: "ffn-remote", Role: inference.SplitEndpointRoleFFN, URL: server.URL},
	})
	if err != nil {
		t.Fatalf("NewRemoteSplitFFNExecutor: %v", err)
	}
	local := &splitExecutorTestLocalRuntime{
		prefill: SplitPrefillResult{
			Tokens: []int32{11},
			Hidden: []float32{1},
			Layers: 1,
		},
		samples: []SplitSampleResult{{TokenID: 42}},
		text:    map[int32]string{42: " remote"},
	}
	executor, err := LoadSplitExecutor(
		context.Background(),
		slicePath,
		WithSplitLocalRuntime(local),
		WithSplitFFNExecutor(remote),
	)
	if err != nil {
		t.Fatalf("LoadSplitExecutor: %v", err)
	}

	got, err := executor.Generate(context.Background(), "hi", GenerateConfig{MaxTokens: 1})

	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if got != " remote" || remoteCalls != 1 {
		t.Fatalf("Generate = %q remoteCalls=%d, want remote FFN path", got, remoteCalls)
	}
	if len(local.sampleHidden) != 1 || local.sampleHidden[0] != 22 {
		t.Fatalf("sample hidden = %v, want remote FFN hidden [22]", local.sampleHidden)
	}
}

func TestRemoteSplitFFNExecutor_Bad(t *testing.T) {
	if _, err := NewRemoteSplitFFNExecutor(RemoteSplitFFNConfig{}); err == nil {
		t.Fatal("missing endpoint URL error = nil")
	}
	if _, err := NewRemoteSplitFFNExecutor(RemoteSplitFFNConfig{
		URL:      "http://127.0.0.1:1",
		Endpoint: inference.SplitEndpoint{Role: inference.SplitEndpointRoleAttention},
	}); err == nil {
		t.Fatal("wrong endpoint role error = nil")
	}

	server := core.NewHTTPTestServer(core.HandlerFunc(func(w core.ResponseWriter, r *core.Request) {
		core.WriteString(w, core.JSONMarshalString(RemoteSplitFFNResponse{Error: "backend unavailable"}))
	}))
	defer server.Close()
	executor, err := NewRemoteSplitFFNExecutor(RemoteSplitFFNConfig{
		Endpoint: inference.SplitEndpoint{Role: inference.SplitEndpointRoleFFN, URL: server.URL},
	})
	if err != nil {
		t.Fatalf("NewRemoteSplitFFNExecutor: %v", err)
	}
	if _, err := executor.ForwardFFN(context.Background(), SplitFFNRequest{Layer: 1, Hidden: []float32{1}}); err == nil || !core.Contains(err.Error(), "backend unavailable") {
		t.Fatalf("ForwardFFN error = %v, want remote backend error", err)
	}
}
