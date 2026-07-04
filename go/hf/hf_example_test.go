// SPDX-Licence-Identifier: EUPL-1.2

package hf

import (
	"context"
	"fmt"

	core "dappco.re/go"
)

// ExampleNewRemoteSource constructs a Hugging Face Hub metadata source. The
// constructor trims a trailing slash from the base URL and defaults the
// user-agent when none is supplied — no network is touched here.
func ExampleNewRemoteSource() {
	source := NewRemoteSource(RemoteConfig{
		BaseURL: "https://huggingface.co/",
	})
	fmt.Println(source.baseURL, source.userAgent)
	// Output: https://huggingface.co go-mlx
}

// ExampleRemoteSource_SearchModels queries the Hub model-search endpoint. The
// example points the source at a loopback test server (no real network) that
// returns one model, so the result is deterministic.
func ExampleRemoteSource_SearchModels() {
	server := core.NewHTTPTestServer(core.HandlerFunc(func(w core.ResponseWriter, _ *core.Request) {
		w.Header().Set("Content-Type", "application/json")
		core.WriteString(w, `[{"id": "Qwen/Qwen3-0.6B", "config": {"model_type": "qwen3"}}]`)
	}))
	defer server.Close()

	source := NewRemoteSource(RemoteConfig{BaseURL: server.URL})
	models, err := source.SearchModels(context.Background(), "qwen", 5)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(len(models), models[0].ID, models[0].Config.ModelType)
	// Output: 1 Qwen/Qwen3-0.6B qwen3
}

// ExampleRemoteSource_ModelMetadata fetches metadata for a single model id. The
// example points the source at a loopback test server (no real network); when
// the body carries no `id`/`modelId`, the requested id is filled in.
func ExampleRemoteSource_ModelMetadata() {
	server := core.NewHTTPTestServer(core.HandlerFunc(func(w core.ResponseWriter, _ *core.Request) {
		w.Header().Set("Content-Type", "application/json")
		core.WriteString(w, `{"modelId": "Qwen/Qwen3-0.6B", "config": {"model_type": "qwen3", "num_hidden_layers": 28}}`)
	}))
	defer server.Close()

	source := NewRemoteSource(RemoteConfig{BaseURL: server.URL})
	meta, err := source.ModelMetadata(context.Background(), "Qwen/Qwen3-0.6B")
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(meta.ModelID, meta.Config.ModelType, meta.Config.NumHiddenLayers)
	// Output: Qwen/Qwen3-0.6B qwen3 28
}
