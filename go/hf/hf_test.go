// SPDX-Licence-Identifier: EUPL-1.2

package hf

import (
	"context"
	"testing"

	core "dappco.re/go"
	mp "dappco.re/go/mlx/pack"
	"dappco.re/go/mlx/profile"
)

type fakeHFModelSource struct {
	searchCalled bool
	search       []ModelMetadata
	byID         map[string]ModelMetadata
}

func (s *fakeHFModelSource) SearchModels(_ context.Context, query string, limit int) ([]ModelMetadata, error) {
	if query != "qwen 0.6b" {
		return nil, core.NewError("unexpected query: " + query)
	}
	s.searchCalled = true
	if limit > 0 && limit < len(s.search) {
		return append([]ModelMetadata(nil), s.search[:limit]...), nil
	}
	return append([]ModelMetadata(nil), s.search...), nil
}

func (s *fakeHFModelSource) ModelMetadata(_ context.Context, id string) (ModelMetadata, error) {
	if meta, ok := s.byID[id]; ok {
		return meta, nil
	}
	return ModelMetadata{}, core.NewError("not found: " + id)
}

// TestHf_RemoteSource_SearchModels_Good drives the happy path end-to-end: a
// loopback httptest server serves the HF /api/models search list and the
// per-id metadata body, and NewRemoteSource + SearchModels + ModelMetadata
// round-trip them (verifying the search query/limit, the Bearer auth header on
// the metadata call, and the size/sizeBytes fallbacks). No real network.
func TestHf_RemoteSource_SearchModels_Good(t *testing.T) {
	server := core.NewHTTPTestServer(core.HandlerFunc(func(w core.ResponseWriter, r *core.Request) {
		switch r.URL.Path {
		case "/api/models":
			if r.URL.Query().Get("search") != "qwen" || r.URL.Query().Get("limit") != "2" {
				t.Fatalf("query = %q, want search/limit", r.URL.RawQuery)
			}
			w.Header().Set("Content-Type", "application/json")
			core.WriteString(w, `[{
				"id": "Qwen/Qwen3-0.6B",
				"pipeline_tag": "text-generation",
				"config": {"model_type": "qwen3", "hidden_size": 1024},
				"siblings": [{"rfilename": "model.safetensors", "sizeBytes": 440401920}]
			}]`)
		case "/api/models/Qwen/Qwen3-0.6B":
			if r.Header.Get("Authorization") != "Bearer test-token" {
				t.Fatalf("Authorization = %q", r.Header.Get("Authorization"))
			}
			w.Header().Set("Content-Type", "application/json")
			core.WriteString(w, `{
				"modelId": "Qwen/Qwen3-0.6B",
				"config": {"model_type": "qwen3", "num_hidden_layers": 28},
				"siblings": [{"rfilename": "model.safetensors", "size": 440401920}]
			}`)
		default:
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
	}))
	defer server.Close()

	source := NewRemoteSource(RemoteConfig{
		BaseURL: server.URL,
		Token:   "test-token",
	})
	found, err := source.SearchModels(context.Background(), "qwen", 2)
	if err != nil {
		t.Fatalf("SearchModels() error = %v", err)
	}
	if len(found) != 1 || found[0].ID != "Qwen/Qwen3-0.6B" {
		t.Fatalf("SearchModels() = %+v", found)
	}
	if found[0].Files[0].byteSize() != 440401920 {
		t.Fatalf("file size = %+v", found[0].Files[0])
	}

	meta, err := source.ModelMetadata(context.Background(), "Qwen/Qwen3-0.6B")
	if err != nil {
		t.Fatalf("ModelMetadata() error = %v", err)
	}
	if meta.ModelID != "Qwen/Qwen3-0.6B" || meta.Config.NumHiddenLayers != 28 {
		t.Fatalf("ModelMetadata() = %+v", meta)
	}
}

// TestHf_NewRemoteSource_Bad covers the constructor's defaulting alongside the
// error surface of a source built from a trailing-slash BaseURL: the slash is
// trimmed and the user-agent override is retained, while a nil receiver and a
// malformed/404 response from the resulting source surface errors rather than
// panicking. Loopback httptest only — no real network.
func TestHf_NewRemoteSource_Bad(t *testing.T) {
	var source *RemoteSource
	if _, err := source.SearchModels(context.Background(), "qwen", 1); err == nil {
		t.Fatal("expected nil SearchModels error")
	}
	if _, err := source.ModelMetadata(context.Background(), "qwen/model"); err == nil {
		t.Fatal("expected nil ModelMetadata error")
	}

	server := core.NewHTTPTestServer(core.HandlerFunc(func(w core.ResponseWriter, r *core.Request) {
		switch r.URL.Path {
		case "/api/models":
			core.WriteString(w, "{")
		case "/api/models/missing":
			w.WriteHeader(404)
			core.WriteString(w, "not found")
		default:
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
	}))
	defer server.Close()

	source = NewRemoteSource(RemoteConfig{BaseURL: server.URL + "/", UserAgent: "tests"})
	if source.baseURL != server.URL || source.userAgent != "tests" || source.client == nil {
		t.Fatalf("source defaults = %+v", source)
	}
	if _, err := source.SearchModels(context.Background(), "qwen", 0); err == nil {
		t.Fatal("expected parse error from malformed search response")
	}
	if _, err := source.ModelMetadata(context.Background(), "missing"); err == nil || !core.Contains(err.Error(), "404") {
		t.Fatalf("expected HTTP status error, got %v", err)
	}
}

// TestHf_NewRemoteSource_Good covers the constructor's happy path: a fully
// specified RemoteConfig (BaseURL without a trailing slash, explicit
// UserAgent, a token, and an injected client) is stored verbatim and the
// Authorization header value is pre-built once as "Bearer <token>". No
// network — the constructor only assembles fields.
func TestHf_NewRemoteSource_Good(t *testing.T) {
	client := &core.HTTPClient{}
	source := NewRemoteSource(RemoteConfig{
		BaseURL:   "https://hub.example.com",
		Token:     "secret-token",
		UserAgent: "go-mlx-tests",
		Client:    client,
	})
	if source.baseURL != "https://hub.example.com" {
		t.Fatalf("baseURL = %q, want the supplied URL verbatim", source.baseURL)
	}
	if source.userAgent != "go-mlx-tests" {
		t.Fatalf("userAgent = %q, want the supplied override", source.userAgent)
	}
	if source.authValue != "Bearer secret-token" {
		t.Fatalf("authValue = %q, want pre-built \"Bearer secret-token\"", source.authValue)
	}
	if source.client != client {
		t.Fatal("client = injected client not retained, want the supplied *core.HTTPClient")
	}
}

// TestHf_NewRemoteSource_Ugly covers the constructor's degenerate inputs: a
// zero-value RemoteConfig must default the BaseURL to the public hub and the
// user-agent to "go-mlx", leave the auth header empty (no token), and
// synthesise a non-nil client. A trailing slash on a supplied BaseURL is
// trimmed exactly once. No network.
func TestHf_NewRemoteSource_Ugly(t *testing.T) {
	empty := NewRemoteSource(RemoteConfig{})
	if empty.baseURL != "https://huggingface.co" {
		t.Fatalf("zero-config baseURL = %q, want the default hub", empty.baseURL)
	}
	if empty.userAgent != "go-mlx" {
		t.Fatalf("zero-config userAgent = %q, want default go-mlx", empty.userAgent)
	}
	if empty.authValue != "" {
		t.Fatalf("zero-config authValue = %q, want empty (no token)", empty.authValue)
	}
	if empty.client == nil {
		t.Fatal("zero-config client = nil, want a synthesised *core.HTTPClient")
	}

	trimmed := NewRemoteSource(RemoteConfig{BaseURL: "https://hub.example.com/"})
	if trimmed.baseURL != "https://hub.example.com" {
		t.Fatalf("trailing-slash baseURL = %q, want the slash trimmed", trimmed.baseURL)
	}
}

// TestHf_RemoteSource_ModelMetadata_Good drives ModelMetadata against a
// loopback httptest server that returns a metadata body for one model id; the
// returned struct carries the modelId and the nested config. No real network.
func TestHf_RemoteSource_ModelMetadata_Good(t *testing.T) {
	server := core.NewHTTPTestServer(core.HandlerFunc(func(w core.ResponseWriter, r *core.Request) {
		if r.URL.Path != "/api/models/Qwen/Qwen3-0.6B" {
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		core.WriteString(w, `{
			"modelId": "Qwen/Qwen3-0.6B",
			"config": {"model_type": "qwen3", "num_hidden_layers": 28},
			"siblings": [{"rfilename": "model.safetensors", "size": 440401920}]
		}`)
	}))
	defer server.Close()

	source := NewRemoteSource(RemoteConfig{BaseURL: server.URL})
	meta, err := source.ModelMetadata(context.Background(), "Qwen/Qwen3-0.6B")
	if err != nil {
		t.Fatalf("ModelMetadata() error = %v", err)
	}
	if meta.ModelID != "Qwen/Qwen3-0.6B" || meta.Config.NumHiddenLayers != 28 {
		t.Fatalf("ModelMetadata() = %+v, want the served modelId + config", meta)
	}
	if len(meta.Files) != 1 || meta.Files[0].byteSize() != 440401920 {
		t.Fatalf("ModelMetadata() files = %+v, want one sibling with the size fallback", meta.Files)
	}
}

// TestHf_RemoteSource_ModelMetadata_Bad covers ModelMetadata's error surface: a
// nil receiver returns a guard error, and an HTTP 404 from the server surfaces
// a status error carrying the code. Loopback httptest only — no real network.
func TestHf_RemoteSource_ModelMetadata_Bad(t *testing.T) {
	var nilSource *RemoteSource
	if _, err := nilSource.ModelMetadata(context.Background(), "org/model"); err == nil {
		t.Fatal("ModelMetadata(nil receiver) error = nil, want a guard error")
	}

	server := core.NewHTTPTestServer(core.HandlerFunc(func(w core.ResponseWriter, r *core.Request) {
		if r.URL.Path != "/api/models/missing" {
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
		w.WriteHeader(404)
		core.WriteString(w, "not found")
	}))
	defer server.Close()

	source := NewRemoteSource(RemoteConfig{BaseURL: server.URL})
	if _, err := source.ModelMetadata(context.Background(), "missing"); err == nil || !core.Contains(err.Error(), "404") {
		t.Fatalf("ModelMetadata(404) error = %v, want an HTTP status error mentioning 404", err)
	}
}

// TestHf_RemoteSource_ModelMetadata_Ugly covers ModelMetadata's two awkward
// edges: when the Hub returns a body carrying neither `id` nor `modelId` the
// requested id is filled in, and pointing the source at a closed loopback
// server surfaces a transport error from the client.Do call. The transport
// server is started then immediately closed so the dial fails locally — no
// real network egress.
func TestHf_RemoteSource_ModelMetadata_Ugly(t *testing.T) {
	idless := core.NewHTTPTestServer(core.HandlerFunc(func(w core.ResponseWriter, _ *core.Request) {
		w.Header().Set("Content-Type", "application/json")
		core.WriteString(w, `{"config": {"model_type": "qwen3"}}`)
	}))
	defer idless.Close()

	source := NewRemoteSource(RemoteConfig{BaseURL: idless.URL})
	meta, err := source.ModelMetadata(context.Background(), "org/no-id-model")
	if err != nil {
		t.Fatalf("ModelMetadata() error = %v", err)
	}
	if meta.ID != "org/no-id-model" {
		t.Fatalf("ModelMetadata().ID = %q, want the requested id filled in", meta.ID)
	}

	closed := core.NewHTTPTestServer(core.HandlerFunc(func(w core.ResponseWriter, _ *core.Request) {
		core.WriteString(w, "{}")
	}))
	closedURL := closed.URL
	closed.Close() // nothing listens at closedURL now → dial fails
	dead := NewRemoteSource(RemoteConfig{BaseURL: closedURL})
	if _, err := dead.ModelMetadata(context.Background(), "org/model"); err == nil {
		t.Fatal("ModelMetadata(closed server) error = nil, want a transport error")
	}
}

// TestHf_RemoteSource_SearchModels_Bad covers SearchModels' error surface: a
// nil receiver returns a guard error, and a malformed JSON body from the
// server surfaces a parse error. Loopback httptest only — no real network.
func TestHf_RemoteSource_SearchModels_Bad(t *testing.T) {
	var nilSource *RemoteSource
	if _, err := nilSource.SearchModels(context.Background(), "qwen", 1); err == nil {
		t.Fatal("SearchModels(nil receiver) error = nil, want a guard error")
	}

	server := core.NewHTTPTestServer(core.HandlerFunc(func(w core.ResponseWriter, r *core.Request) {
		if r.URL.Path != "/api/models" {
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
		core.WriteString(w, "{") // malformed JSON array
	}))
	defer server.Close()

	source := NewRemoteSource(RemoteConfig{BaseURL: server.URL})
	if _, err := source.SearchModels(context.Background(), "qwen", 5); err == nil {
		t.Fatal("SearchModels(malformed response) error = nil, want a parse error")
	}
}

// TestHf_RemoteSource_SearchModels_Ugly covers SearchModels' awkward edges: a
// non-positive limit is normalised to the default 10 (asserted on the wire via
// the query the server observes), and pointing the source at a closed loopback
// server surfaces a transport error. The transport server is started then
// immediately closed so the dial fails locally — no real network egress.
func TestHf_RemoteSource_SearchModels_Ugly(t *testing.T) {
	var gotLimit string
	server := core.NewHTTPTestServer(core.HandlerFunc(func(w core.ResponseWriter, r *core.Request) {
		if r.URL.Path != "/api/models" {
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
		gotLimit = r.URL.Query().Get("limit")
		w.Header().Set("Content-Type", "application/json")
		core.WriteString(w, `[]`)
	}))

	source := NewRemoteSource(RemoteConfig{BaseURL: server.URL})
	if _, err := source.SearchModels(context.Background(), "qwen", 0); err != nil {
		t.Fatalf("SearchModels(limit 0) error = %v", err)
	}
	if gotLimit != "10" {
		t.Fatalf("limit on the wire = %q, want 10 (non-positive limit defaults)", gotLimit)
	}
	closedURL := server.URL
	server.Close() // nothing listens at closedURL now → dial fails
	dead := NewRemoteSource(RemoteConfig{BaseURL: closedURL})
	if _, err := dead.SearchModels(context.Background(), "qwen", 1); err == nil {
		t.Fatal("SearchModels(closed server) error = nil, want a transport error")
	}
}

func TestHFLocalMetadataHelpers_Good(t *testing.T) {
	cacheRoot := core.PathJoin(t.TempDir(), "models--org--name")
	snapshot := core.PathJoin(cacheRoot, "snapshots", "b")
	if result := core.MkdirAll(snapshot, 0o755); !result.OK {
		t.Fatalf("mkdir snapshot: %v", result.Value)
	}
	writeModelPackFile(t, core.PathJoin(snapshot, "config.json"), `{"architectures":["Qwen3ForCausalLM"],"context_length":32768}`)
	writeModelPackFile(t, core.PathJoin(snapshot, "model-q4.gguf"), "gguf")
	writeModelPackFile(t, core.PathJoin(snapshot, "model.safetensors"), "safe")
	writeModelPackFile(t, core.PathJoin(snapshot, "pytorch_model.bin"), "bin")
	writeModelPackFile(t, core.PathJoin(snapshot, "tokenizer.json"), "{}")

	meta, root, err := inspectLocalMetadata(cacheRoot)
	if err != nil {
		t.Fatalf("inspectLocalMetadata: %v", err)
	}
	if root != snapshot {
		t.Fatalf("root = %q, want %q", root, snapshot)
	}
	if meta.ID != "org/name" {
		t.Fatalf("ID = %q, want org/name", meta.ID)
	}
	if len(meta.Files) != 4 {
		t.Fatalf("files = %+v", meta.Files)
	}
	if got := resolveLocalMetadataRoot(core.PathJoin(snapshot, "config.json")); got != snapshot {
		t.Fatalf("resolve config root = %q, want %q", got, snapshot)
	}
}

func hfFitPlanHasNote(plan FitPlan, fragment string) bool {
	for _, note := range plan.Notes {
		if core.Contains(note, fragment) {
			return true
		}
	}
	return false
}

// TestModelConfigAccessors_Good exercises the value-receiver ModelConfig
// accessors (architecture / contextLength / quantization) directly. planFit
// inlines these for the hot path, so only the benches drive them today; this
// asserts the normalize-then-read logic with real config shapes — including
// the nested text_config promotion that normalized() performs.
func TestModelConfigAccessors_Good(t *testing.T) {
	flat := ModelConfig{
		ModelType:             "qwen3",
		Architectures:         []string{"Qwen3ForCausalLM"},
		ContextLength:         0,
		MaxPositionEmbeddings: 40960,
		QuantizationConfig:    &QuantizationConfig{Bits: 4, GroupSize: 64},
	}
	if got := flat.architecture(); got != "qwen3" {
		t.Fatalf("architecture() = %q, want qwen3", got)
	}
	if got := flat.contextLength(); got != 40960 {
		t.Fatalf("contextLength() = %d, want 40960 (falls back to max_position_embeddings)", got)
	}
	if bits, group := flat.quantization(); bits != 4 || group != 64 {
		t.Fatalf("quantization() = %d/%d, want 4/64", bits, group)
	}

	// Nested text_config: normalized() lifts the inner config so the
	// accessors read the real architecture/context, not the outer wrapper.
	nested := ModelConfig{
		ModelType: "qwen3_5",
		TextConfig: &ModelConfig{
			ModelType:     "qwen3_next",
			Architectures: []string{"Qwen3NextForCausalLM"},
			ContextLength: 98304,
			Quantization:  &QuantizationConfig{Bits: 8, GroupSize: 32},
		},
	}
	if got := nested.architecture(); got != "qwen3_next" {
		t.Fatalf("nested architecture() = %q, want qwen3_next", got)
	}
	if got := nested.contextLength(); got != 98304 {
		t.Fatalf("nested contextLength() = %d, want 98304", got)
	}
	if bits, group := nested.quantization(); bits != 8 || group != 32 {
		t.Fatalf("nested quantization() = %d/%d, want 8/32 (read from text_config)", bits, group)
	}
}

// TestModelConfigQuantization_Bad covers the no-quant-block path of the
// quantization accessor — an unquantised (dense) config returns 0/0.
func TestModelConfigQuantization_Bad(t *testing.T) {
	dense := ModelConfig{ModelType: "qwen3", HiddenSize: 1024}
	if bits, group := dense.quantization(); bits != 0 || group != 0 {
		t.Fatalf("quantization() on dense config = %d/%d, want 0/0", bits, group)
	}
	if got := dense.architecture(); got != "qwen3" {
		t.Fatalf("architecture() = %q, want qwen3", got)
	}
}

// TestModelConfigQuantizationType_Good covers quantizationType — the string
// label of the active quant block. QuantizationConfig wins over Quantization
// when both are present (normalized() promotes the nested text_config first,
// then the accessor prefers quantization_config over the legacy quantization
// key), and an unquantised config returns "".
func TestModelConfigQuantizationType_Good(t *testing.T) {
	cfg := ModelConfig{
		ModelType:          "qwen3",
		QuantizationConfig: &QuantizationConfig{Bits: 4, GroupSize: 64, Type: "mxfp4"},
	}
	if got := cfg.quantizationType(); got != "mxfp4" {
		t.Fatalf("quantizationType() = %q, want mxfp4 (from quantization_config)", got)
	}

	// Legacy `quantization` key only — quantization_config absent.
	legacy := ModelConfig{
		ModelType:    "qwen3",
		Quantization: &QuantizationConfig{Bits: 8, GroupSize: 32, Type: "affine"},
	}
	if got := legacy.quantizationType(); got != "affine" {
		t.Fatalf("quantizationType() = %q, want affine (from legacy quantization)", got)
	}

	// Dense (no quant block) → empty type.
	if got := (ModelConfig{ModelType: "qwen3"}).quantizationType(); got != "" {
		t.Fatalf("quantizationType() on dense config = %q, want empty", got)
	}
}

// TestConfigArchitecture_Good exercises configArchitecture, the
// already-normalised pointer-receiver variant, across its three resolution
// tiers: the BertForSequenceClassification → bert_rerank special case wins
// even when a model_type is present; otherwise model_type is normalised; and
// when model_type is empty the transformers architecture name resolves.
func TestConfigArchitecture_Good(t *testing.T) {
	// Rerank special case: the bert_rerank short-circuit fires before
	// model_type, so a sequence-classification head reports bert_rerank.
	rerank := ModelConfig{
		ModelType:     "bert",
		Architectures: []string{"BertForSequenceClassification"},
	}
	if got := configArchitecture(&rerank); got != "bert_rerank" {
		t.Fatalf("configArchitecture() = %q, want bert_rerank (special case wins over model_type)", got)
	}

	// model_type present, no rerank head → normalise the model_type.
	byType := ModelConfig{ModelType: "qwen3", Architectures: []string{"Qwen3ForCausalLM"}}
	if got := configArchitecture(&byType); got != "qwen3" {
		t.Fatalf("configArchitecture() = %q, want qwen3 (from model_type)", got)
	}

	// model_type empty → fall through to the transformers architecture name.
	byArch := ModelConfig{Architectures: []string{"LlamaForCausalLM"}}
	if got := configArchitecture(&byArch); got != "llama" {
		t.Fatalf("configArchitecture() = %q, want llama (from architectures)", got)
	}
}

// TestConfigArchitecture_Bad covers the empty-result path: neither a known
// model_type nor a recognised transformers architecture name resolves, so
// configArchitecture returns "".
func TestConfigArchitecture_Bad(t *testing.T) {
	if got := configArchitecture(&ModelConfig{}); got != "" {
		t.Fatalf("configArchitecture(empty) = %q, want empty", got)
	}
	unknown := ModelConfig{Architectures: []string{"TotallyMadeUpForCausalLM"}}
	if got := configArchitecture(&unknown); got != "" {
		t.Fatalf("configArchitecture(unknown arch) = %q, want empty", got)
	}
}

// TestGemma4ConfigDetectors_Good covers isGemma4UnifiedConfig and
// isGemma4AssistantConfig — the two detectors PlanFits uses to pick the
// unified-multimodal and attached-drafter memory plans. Each matches on both
// a normalised model_type and a transformers architecture name.
func TestGemma4ConfigDetectors_Good(t *testing.T) {
	// Unified via model_type.
	if !isGemma4UnifiedConfig(ModelConfig{ModelType: "gemma4_unified"}) {
		t.Fatal("isGemma4UnifiedConfig(model_type) = false, want true")
	}
	// Unified via architecture name.
	if !isGemma4UnifiedConfig(ModelConfig{Architectures: []string{"Gemma4UnifiedForConditionalGeneration"}}) {
		t.Fatal("isGemma4UnifiedConfig(architectures) = false, want true")
	}
	// Assistant via model_type.
	if !isGemma4AssistantConfig(ModelConfig{ModelType: "gemma4_assistant"}) {
		t.Fatal("isGemma4AssistantConfig(model_type) = false, want true")
	}
	// Assistant via architecture name.
	if !isGemma4AssistantConfig(ModelConfig{Architectures: []string{"Gemma4AssistantForCausalLM"}}) {
		t.Fatal("isGemma4AssistantConfig(architectures) = false, want true")
	}
}

// TestGemma4ConfigDetectors_Bad covers the negative path: a non-Gemma-4
// config matches neither detector, and the architecture-name loop must run to
// exhaustion without a hit.
func TestGemma4ConfigDetectors_Bad(t *testing.T) {
	plain := ModelConfig{ModelType: "qwen3", Architectures: []string{"Qwen3ForCausalLM"}}
	if isGemma4UnifiedConfig(plain) {
		t.Fatal("isGemma4UnifiedConfig(qwen3) = true, want false")
	}
	if isGemma4AssistantConfig(plain) {
		t.Fatal("isGemma4AssistantConfig(qwen3) = true, want false")
	}
}

// TestArchProfileHelpers_Good covers archSupported, archNativeRuntime and
// usesGenerationKVCache against the live architecture registry. qwen3 is a
// supported native generation arch; bert is a supported native *embedding*
// arch (so no generation KV cache); an unknown arch is unsupported on every
// axis.
//
// NOTE: archNativeRuntime's middle branch (recognised && !NativeRuntime) is
// not reachable through the public registry — every registered arch is a
// native* profile with NativeRuntime=true (verified: 0 metadata-only
// registrations). The lookup-miss path (!ok → false) and the native-true
// path are both covered here.
func TestArchProfileHelpers_Good(t *testing.T) {
	if !archSupported("qwen3") {
		t.Fatal("archSupported(qwen3) = false, want true")
	}
	if !archNativeRuntime("qwen3") {
		t.Fatal("archNativeRuntime(qwen3) = false, want true")
	}
	if !usesGenerationKVCache(nil, "qwen3") {
		t.Fatal("usesGenerationKVCache(nil, qwen3) = false, want true (generation arch)")
	}

	// bert is a recognised embedding encoder → no generation KV cache.
	if !archSupported("bert") {
		t.Fatal("archSupported(bert) = false, want true")
	}
	if usesGenerationKVCache(nil, "bert") {
		t.Fatal("usesGenerationKVCache(nil, bert) = true, want false (embedding arch)")
	}
}

// TestArchProfileHelpers_Bad covers the unsupported-architecture paths: an
// unknown name is neither supported nor native, and usesGenerationKVCache
// still defaults to true for an unknown arch (no profile says otherwise).
func TestArchProfileHelpers_Bad(t *testing.T) {
	const unknown = "totally_made_up_arch"
	if archSupported(unknown) {
		t.Fatalf("archSupported(%q) = true, want false", unknown)
	}
	if archNativeRuntime(unknown) {
		t.Fatalf("archNativeRuntime(%q) = true, want false", unknown)
	}
	if !usesGenerationKVCache(nil, unknown) {
		t.Fatalf("usesGenerationKVCache(nil, %q) = false, want true (unknown defaults to generation)", unknown)
	}
}

// TestUsesGenerationKVCache_PackOverrides_Ugly covers the ModelPack-driven
// branches of usesGenerationKVCache: an Embedding/Rerank pack short-circuits
// to false; a pack Architecture overrides the passed name; and an
// ArchitectureProfile flagged Embeddings/Rerank forces false even when the
// arch name would otherwise be a generation target.
func TestUsesGenerationKVCache_PackOverrides_Ugly(t *testing.T) {
	// Embedding pack → false regardless of arch name.
	embedPack := &mp.ModelPack{Embedding: &mp.ModelEmbeddingProfile{}}
	if usesGenerationKVCache(embedPack, "qwen3") {
		t.Fatal("usesGenerationKVCache(embedding pack) = true, want false")
	}

	// Pack architecture overrides the passed name: pass a generation arch
	// but let the pack declare an embedding arch → false.
	overridePack := &mp.ModelPack{Architecture: "bert"}
	if usesGenerationKVCache(overridePack, "qwen3") {
		t.Fatal("usesGenerationKVCache(pack arch=bert) = true, want false (pack arch overrides)")
	}

	// ArchitectureProfile flagged Rerank → false.
	rerankPack := &mp.ModelPack{
		Architecture:        "qwen3",
		ArchitectureProfile: &profile.ModelArchitectureProfile{Rerank: true},
	}
	if usesGenerationKVCache(rerankPack, "qwen3") {
		t.Fatal("usesGenerationKVCache(profile.Rerank) = true, want false")
	}

	// A generation pack (no embedding/rerank signal) keeps the KV cache.
	genPack := &mp.ModelPack{Architecture: "qwen3"}
	if !usesGenerationKVCache(genPack, "qwen3") {
		t.Fatal("usesGenerationKVCache(generation pack) = false, want true")
	}
}

// TestResolveArchitectureProfile_Good covers resolveArchitectureProfile: it
// fills a pack's ArchitectureProfile from the registry when the pack names a
// recognised arch and has no profile yet, and is a no-op when the pack is
// nil, has no architecture, or already carries a profile.
func TestResolveArchitectureProfile_Good(t *testing.T) {
	// nil pack: must not panic.
	resolveArchitectureProfile(nil)

	// No architecture: profile stays nil.
	empty := &mp.ModelPack{}
	resolveArchitectureProfile(empty)
	if empty.ArchitectureProfile != nil {
		t.Fatal("resolveArchitectureProfile(no arch) populated a profile, want nil")
	}

	// Recognised arch, no profile yet → populated from the registry.
	pack := &mp.ModelPack{Architecture: "qwen3"}
	resolveArchitectureProfile(pack)
	if pack.ArchitectureProfile == nil || pack.ArchitectureProfile.ID != "qwen3" {
		t.Fatalf("resolveArchitectureProfile(qwen3) profile = %+v, want qwen3 profile", pack.ArchitectureProfile)
	}

	// Already has a profile → left untouched (no overwrite).
	sentinel := &profile.ModelArchitectureProfile{ID: "preset"}
	preset := &mp.ModelPack{Architecture: "qwen3", ArchitectureProfile: sentinel}
	resolveArchitectureProfile(preset)
	if preset.ArchitectureProfile != sentinel {
		t.Fatal("resolveArchitectureProfile overwrote an existing profile, want no-op")
	}

	// Unrecognised arch → profile stays nil.
	unknown := &mp.ModelPack{Architecture: "totally_made_up_arch"}
	resolveArchitectureProfile(unknown)
	if unknown.ArchitectureProfile != nil {
		t.Fatal("resolveArchitectureProfile(unknown arch) populated a profile, want nil")
	}
}

// TestRemoteSource_ModelMetadataIDFallback_Good covers the ModelMetadata
// fallback branch: when the Hub returns a metadata body carrying neither `id`
// nor `modelId`, the requested model id is filled in. Loopback httptest
// server only — no real network.
func TestRemoteSource_ModelMetadataIDFallback_Good(t *testing.T) {
	server := core.NewHTTPTestServer(core.HandlerFunc(func(w core.ResponseWriter, _ *core.Request) {
		w.Header().Set("Content-Type", "application/json")
		core.WriteString(w, `{"config": {"model_type": "qwen3"}}`)
	}))
	defer server.Close()

	source := NewRemoteSource(RemoteConfig{BaseURL: server.URL})
	meta, err := source.ModelMetadata(context.Background(), "org/no-id-model")
	if err != nil {
		t.Fatalf("ModelMetadata() error = %v", err)
	}
	if meta.ID != "org/no-id-model" {
		t.Fatalf("ModelMetadata().ID = %q, want the requested id filled in", meta.ID)
	}
}

// TestRemoteSource_TransportError_Bad covers getJSON's client.Do failure
// branch: pointing the source at a closed loopback server surfaces a
// transport error from both SearchModels and ModelMetadata. The server is
// started then immediately closed so the dial fails locally — no real
// network egress.
func TestRemoteSource_TransportError_Bad(t *testing.T) {
	server := core.NewHTTPTestServer(core.HandlerFunc(func(w core.ResponseWriter, _ *core.Request) {
		core.WriteString(w, "{}")
	}))
	closedURL := server.URL
	server.Close() // nothing listens at closedURL now → dial fails

	source := NewRemoteSource(RemoteConfig{BaseURL: closedURL})
	if _, err := source.ModelMetadata(context.Background(), "org/model"); err == nil {
		t.Fatal("ModelMetadata(closed server) error = nil, want transport error")
	}
	if _, err := source.SearchModels(context.Background(), "qwen", 1); err == nil {
		t.Fatal("SearchModels(closed server) error = nil, want transport error")
	}
}

// TestModelConfigProbe_SyntheticConfig_Good covers the modelConfigProbe family
// — readModelConfig plus the architecture/numLayers/vocabSize/hiddenSize/
// contextLength/quantBits/quantGroup accessors — against a synthetic
// config.json. Top-level fields win; the nested text_config supplies the
// fallbacks. NOTE: this probe family is package-internal and has no live
// caller inside hf (the live, used copy is in gguf/info.go); these tests
// exercise the duplicated logic directly. Fixtures via t.TempDir(), no
// network.
func TestModelConfigProbe_SyntheticConfig_Good(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{
		"model_type": "qwen3",
		"architectures": ["Qwen3ForCausalLM"],
		"vocab_size": 151936,
		"hidden_size": 1024,
		"num_hidden_layers": 28,
		"max_position_embeddings": 40960,
		"quantization_config": {"bits": 4, "group_size": 64}
	}`)

	probe, err := readModelConfig(dir)
	if err != nil {
		t.Fatalf("readModelConfig() error = %v", err)
	}
	if got := probe.architecture(); got != "qwen3" {
		t.Fatalf("architecture() = %q, want qwen3", got)
	}
	if got := probe.numLayers(); got != 28 {
		t.Fatalf("numLayers() = %d, want 28", got)
	}
	if got := probe.vocabSize(); got != 151936 {
		t.Fatalf("vocabSize() = %d, want 151936", got)
	}
	if got := probe.hiddenSize(); got != 1024 {
		t.Fatalf("hiddenSize() = %d, want 1024", got)
	}
	if got := probe.contextLength(); got != 40960 {
		t.Fatalf("contextLength() = %d, want 40960", got)
	}
	if got := probe.quantBits(); got != 4 {
		t.Fatalf("quantBits() = %d, want 4", got)
	}
	if got := probe.quantGroup(); got != 64 {
		t.Fatalf("quantGroup() = %d, want 64", got)
	}
}

// TestModelConfigProbe_NestedAndRerank_Ugly covers the probe fallbacks: a
// nested text_config supplies layers/vocab/hidden/context when the top level
// is empty, the legacy `quantization` key supplies bits/group when
// `quantization_config` is absent, the BertForSequenceClassification →
// bert_rerank special case fires, and a nil probe is safe on every accessor.
func TestModelConfigProbe_NestedAndRerank_Ugly(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{
		"architectures": ["BertForSequenceClassification"],
		"text_config": {
			"model_type": "bert",
			"vocab_size": 30522,
			"hidden_size": 768,
			"num_hidden_layers": 12,
			"max_position_embeddings": 512
		},
		"quantization": {"bits": 8, "group_size": 32}
	}`)

	probe, err := readModelConfig(dir)
	if err != nil {
		t.Fatalf("readModelConfig() error = %v", err)
	}
	if got := probe.architecture(); got != "bert_rerank" {
		t.Fatalf("architecture() = %q, want bert_rerank (sequence-classification special case)", got)
	}
	if got := probe.numLayers(); got != 12 {
		t.Fatalf("numLayers() = %d, want 12 (from text_config)", got)
	}
	if got := probe.vocabSize(); got != 30522 {
		t.Fatalf("vocabSize() = %d, want 30522 (from text_config)", got)
	}
	if got := probe.hiddenSize(); got != 768 {
		t.Fatalf("hiddenSize() = %d, want 768 (from text_config)", got)
	}
	if got := probe.contextLength(); got != 512 {
		t.Fatalf("contextLength() = %d, want 512 (from text_config)", got)
	}
	if got := probe.quantBits(); got != 8 {
		t.Fatalf("quantBits() = %d, want 8 (from legacy quantization key)", got)
	}
	if got := probe.quantGroup(); got != 32 {
		t.Fatalf("quantGroup() = %d, want 32 (from legacy quantization key)", got)
	}

	// nil probe is safe on every accessor.
	var nilProbe *modelConfigProbe
	if nilProbe.architecture() != "" || nilProbe.numLayers() != 0 || nilProbe.vocabSize() != 0 ||
		nilProbe.hiddenSize() != 0 || nilProbe.contextLength() != 0 || nilProbe.quantBits() != 0 ||
		nilProbe.quantGroup() != 0 {
		t.Fatal("nil probe accessors returned non-zero, want all zero/empty")
	}
}

// TestModelConfigProbe_ReadErrors_Bad covers readModelConfig's failure
// branches: a missing config.json surfaces a read error, and a malformed JSON
// body surfaces an unmarshal error.
func TestModelConfigProbe_ReadErrors_Bad(t *testing.T) {
	if _, err := readModelConfig(t.TempDir()); err == nil {
		t.Fatal("readModelConfig(no config.json) error = nil, want read error")
	}

	bad := t.TempDir()
	writeModelPackFile(t, core.PathJoin(bad, "config.json"), "{not valid json")
	if _, err := readModelConfig(bad); err == nil {
		t.Fatal("readModelConfig(malformed) error = nil, want unmarshal error")
	}
}

// TestIndexString_Good covers indexString, the allocation-free substring
// search helper: empty needle returns 0, a longer needle than the haystack
// returns -1, a hit returns the first index, and a miss returns -1.
func TestIndexString_Good(t *testing.T) {
	cases := []struct {
		s, sub string
		want   int
	}{
		{"hello world", "", 0},
		{"hi", "longer-than-haystack", -1},
		{"hello world", "world", 6},
		{"hello world", "lo", 3},
		{"hello world", "xyz", -1},
		{"aaa", "aaa", 0},
	}
	for _, tc := range cases {
		if got := indexString(tc.s, tc.sub); got != tc.want {
			t.Fatalf("indexString(%q, %q) = %d, want %d", tc.s, tc.sub, got, tc.want)
		}
	}
}
