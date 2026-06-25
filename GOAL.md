# Goal: Native Engine Replacement

Updated 2026-06-25.

Contract: make `go/pkg/native` replace `go/pkg/metal` by copying proven engine behaviour; feature parity first; no gates/settings.

Env: use repo `AGENTS.md` native test env.

Target/proof: `go/pkg/native` >=95%; latest 91.5%; coverage 40.687s.

Done: resident heads/scratch; BF16/qmv greedy; MTP direct greedy; softcap; whole-tensor/fused MoE; MoE quant per-weight geometry; mmap MoE quant triple+norm/scale views; prompt cache.

Done 2026-06-25: dispatchSink op-unification — the live re-encode path and the ICB record path now share ONE emit per op (RMSNorm, binary, fused-gelu, RoPE, QK-norm-rope, SDPA, 4-bit qmv, bf16 gemv) across arch + non-arch, byte-identical, zero-alloc (b2dc1a9e..90996e26, 1/N–10/N); ICB fast path opened to non-uniform kvHeads so 12B/31B MQA-global ride replay (3a3f94a6) — validated end-to-end on the REAL models: e2b AND 12B generate correct text on the ICB path (smoke: "2+2=4", farmer→17); 2-pass SDPA on live decode past the long-context knee (cf8bef36).

Done 2026-06-25 (streaming): fixed `generate -native` reporting decode 0.000s (2507bb56). Root cause was NOT greedy — greedy always streamed via the session's `GenerateEach`. It was the **sampled (temp>0)** path: `model.GenerateSampledWithStopTokensTransform` returns `[]int32` in one batch, so the iterator yielded every token at the end (whole run mislabelled as a 16s "prefill"). `generate -temp` defaults to 1.0, so default benching hit it. Fix: threaded an observational per-token `yield` through the shared contract loop + added `GenerateSampledWithStopTokensTransformEach` (yield==nil = byte-identical batch); native `stream()` routes both sampled branches through it. pkg/metal parity restored.

Perf reality (e2b 4-bit, decode-only, warmup-excluded): pkg/native **157.7** tok/s vs pkg/metal **178.4** — native ~12% BEHIND. This session shifted correctness + structure, NOT the decode rate (157.2→157.7 = noise). The no-cgo path should be AHEAD (one hop); the deficit is recoverable host overhead, found by profiling — not by more op-sharing.

Next:
- SAMPLED path doesn't use ICB (exposed by the streaming fix): `generate -native` temp 1.0 now reads honestly at **16.9** tok/s vs greedy's **157** — the sampled session loop runs the generic `generateStepwiseWithSession`/`StepWithID`, not the ICB fast replay. pkg/metal's sampled decode IS fast (178 at temp 1.0). **Why it's not a drop-in (read before attempting):** greedy's 157 comes from `generateChainedGPUTail`/`generatePipelinedGPUTail` — the argmax runs ON-GPU and the next token feeds back ON-GPU (one command buffer/token, ZERO host logit readback). Sampling needs the full vocab logits on the HOST (temp/top-k/top-p), which breaks that exact optimisation. So the real options are (a) **on-GPU sampling kernels** (temp/top-k/top-p in Metal, feeding the drawn token back on-GPU — keeps the chained path, ~150), or (b) **ICB-replayed layers + host full-logits head + host sample** (loses the chained on-GPU feedback but the layers still replay — likely well above 16 but below 157; MEASURE). Either is a real piece with a fixed-seed parity gate (same seed: generic-stepwise vs new path → identical tokens), not a ~50-line add. Cladius assessed + deliberately did NOT rush a half-version into a closing window.
- Collapse the TWO decode orchestrations into ONE (the contract's "one path, no gates"): live `stepToken` (encAttnHalfKV/encMLPHalfBF16, decode_forward_arch.go) and ICB `recordArchICB` (decode_forward_arch_icb.go) are still two hand-written layer expressions gated by `icbEligible`. dispatchSink unified the OPS, not the orchestration. Borrow pkg/metal's model (one compiled forward, replayed by decode_replay.go): make the ICB RECORD the single live expression via a recording sink (icbSink that calls emit() per op); delete recordArchICB's inline loop as dead code.
- Profile the ~20 tok/s native↔metal decode gap (per-token host overhead: purego marshaling vs cgo per-call; dispatch count vs pkg/metal's compiled-graph fusion).
- (carried) ICB MoE dispatch; fused qmv argmax; benchmark-backed resource wins.

Known reds (pre-existing — verified failing at 6868d283, BEFORE the 2026-06-25 session commits, so NOT caused by the dispatchSink/streaming work; "green" otherwise = no regression):
- `TestGemma4PostNorms` (gemma4_norms_test.go) — integrated `DecodeForwardArch` ≠ the op-composed `archDenseNormRef` on the gemma4 sandwich post-norms (PostAttn/PostFF norm applied to the projection output before the residual): tok0 byte0 0x48 vs 0xa0. One of {integrated forward, composed reference} is wrong on post-norm placement. Diagnostic: if real gemma4 (e2b/12B) carries live post-norms and serving is correct, the *reference* is stale; if the synthetic post-norm path is untested by serving, the *forward* may mis-place the norm. Resolve before trusting the forward gate.
- `TestSpikeFineGrainedReplayMatchesCoarse` (spike_e2b_bench_test.go) — fine-grained ICB replay ≠ coarse because Metal memory barriers don't enforce the true data deps (the test name says so); nondeterministic cosine (0.0004…-0.037 run to run). Deep ICB-barrier dependency-ordering work — part of the replay/orchestration lane.
