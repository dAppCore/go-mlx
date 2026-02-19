# TODO.md — go-mlx Task Queue

Dispatched from core/go orchestration. Pick up tasks in order.

---

## Phase 1: Standalone Package Hardening

- [ ] **Verify go generate → test round-trip** — Run `go generate ./...` to build mlx-c, then `go test ./...`. Confirm all 3 test files (grad, lora, optim) pass on Apple Silicon. Document any CMake version requirements.
- [ ] **Add missing tests for core operations** — `ops.go` (353 LOC), `array.go` (261 LOC), `nn.go`, `compile.go`, `fast.go` have zero tests. Priority: ops (MatMul, Softmax, Add) and array (create, reshape, data access).
- [ ] **Add missing tests for model/tokenizer/sample/cache** — `model/`, `tokenizer/`, `sample/`, `cache/` have zero test files. Priority: tokenizer (BPE round-trip), sample (temperature/top-k), cache (KV append/trim).
- [ ] **Benchmark suite** — No benchmarks exist. Add: MatMul (various sizes), Softmax, model.Forward (single token), tokenizer.Encode/Decode, full Generate (tokens/sec). Baseline on M3 Ultra.

## Phase 2: Model Support

- [ ] **Gemma3-1B inference validation** — The go-i18n Phase 2a needs 1B model inference for domain classification at ~5K sentences/sec. Validate Gemma3-1B loads and generates correctly via `model.LoadModel()` + `model.Generate()`. Report tokens/sec.
- [ ] **Model loading robustness** — Test with missing files, corrupted safetensors, wrong dtype. Currently no error handling tests for `io.go`.
- [ ] **Add Llama model support** — Only Gemma3 and Qwen3 exist. Llama architecture would cover Meta's model family (Llama 3, CodeLlama).

## Phase 3: Training Pipeline

- [ ] **LoRA fine-tuning end-to-end** — `lora.go` has the adapter but no integration test showing: load base model → apply LoRA → train on small dataset → save adapter → reload. Critical for LEM Lab.
- [ ] **Gradient checkpointing** — `grad.go` has VJP but large models will OOM without checkpointing. Add selective recomputation.
- [ ] **Mixed precision training** — MLX supports BFloat16/Float16. Add dtype selection for training (currently inference uses model's native dtype).

## Phase 4: API Polish

- [ ] **Error handling audit** — `checkError()` only logs. Should return errors to callers instead of silent logging. The C error handler stores last error but Go code doesn't propagate it.
- [ ] **Memory management audit** — Array finalizers use `runtime.SetFinalizer`. Verify no leaks under sustained inference (1000+ tokens). Check C-side deallocation.
- [ ] **Documentation** — Public API has minimal godoc. Add examples for common workflows: load model, generate text, fine-tune with LoRA.

---

## Upstream Dependencies

- **go-i18n Phase 2a** is blocked on this package providing working Gemma3-1B inference
- **go-ai/ml/backend_mlx.go** imports this package — after split, go-ai needs `replace` directive or published module

## Workflow

1. Virgil in core/go writes tasks here after research
2. This repo's session picks up tasks in phase order
3. Mark `[x]` when done, note commit hash
4. New discoveries → add tasks, flag in FINDINGS.md
