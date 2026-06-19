#!/bin/bash
# SPDX-Licence-Identifier: EUPL-1.2
#
# native-smoke.sh — the real-model correctness gate for the no-cgo native engine.
#
# Loads each gemma4 variant on NATIVE (lthn-mlx generate --native), generates from
# a real instruction at temp 0, and JUDGES the output:
#
#   PASS  coherent text
#   FAIL  load error | no output (silent crash) | empty | repetition collapse
#   MISS  model not cached
#
# This is the bar for "native works". `go test` green only ever covered a synthetic
# uniform-dense arch, which hid every real-model failure (MatFormer, KV-share, the
# unified hybrid decode, MoE). Running a real checkpoint is the only thing that
# proves the user path — perf benching stays AX-11 synthetic, correctness does not.
#
#   native-smoke.sh                # all cached gemma4 4bit variants
#   native-smoke.sh 12b e2b        # specific keys
#   native-smoke.sh /path/to/snap  # an explicit snapshot dir
#
# Env: LTHN_MLX_BIN, MLX_METALLIB_PATH, PROMPT, MAXTOK.
set -uo pipefail # NOT -e: a model failing the gate is the RESULT, not a script error.

BIN="${LTHN_MLX_BIN:-/private/tmp/go-mlx-self/bin/lthn-mlx}"
: "${MLX_METALLIB_PATH:=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib}"
export MLX_METALLIB_PATH
PROMPT="${PROMPT:-Explain what a hash map is and why its lookups are fast.}"
MAXTOK="${MAXTOK:-48}"
HUB="$HOME/.cache/huggingface/hub"

declare -A REPO=(
	[e2b]=models--mlx-community--gemma-4-E2B-it-4bit
	[e2b6]=models--mlx-community--gemma-4-e2b-it-6bit
	[e4b]=models--mlx-community--gemma-4-E4B-it-qat-4bit
	[12b]=models--mlx-community--gemma-4-12B-it-4bit
	[26b]=models--mlx-community--gemma-4-26B-A4B-it-qat-4bit
	[31b]=models--mlx-community--gemma-4-31B-it-4bit
)
# e2b spans the bit-width matrix (4-bit + 6-bit) and e4b carries the mixed 8-bit MLP, so the
# E-family alone exercises 4/6/8-bit through the shared quant-agnostic loader (R9), cache-cheap.
ORDER=(e2b e2b6 e4b 12b 26b 31b)

resolve() { # key-or-path -> snapshot dir (empty if unresolved)
	local k="$1"
	[ -d "$k" ] && { echo "$k"; return; }
	local repo="${REPO[$k]:-}"
	[ -z "$repo" ] && return
	ls -d "$HUB/$repo"/snapshots/*/ 2>/dev/null | head -1
}

# The coherence judge lives in a temp file so the generate output reaches its
# stdin — `python3 - <<EOF` would read the heredoc AS the program and leave
# sys.stdin empty. judge() reads a generate run on stdin → "VERDICT|reason|snippet".
JUDGE="$(mktemp -t native-smoke-judge.XXXXXX)"
trap 'rm -f "$JUDGE"' EXIT
cat >"$JUDGE" <<'PY'
import sys, re, collections
raw = sys.stdin.read()
m = re.search(r'generate: (load|warm|decode):.*|Assemble\w*:.*|missing \.weight.*|packed byte span.*|native\.\w+:.*(supported|mismatch|unsupported).*|panic:.*', raw)
if m:
    print("FAIL|error|" + m.group(0)[:110]); raise SystemExit
body = re.sub(r'^.*no-cgo native token-loop contract.*$', '', raw, flags=re.M)
dec = re.search(r'^decode .*tok/s.*$', body, flags=re.M)
text = (body[:dec.start()] if dec else body).strip()
if not dec and not text:
    print("FAIL|no output (silent crash)|"); raise SystemExit
if not text:
    print("FAIL|empty output|"); raise SystemExit
words = re.findall(r'\S+', text)
mx = cur = 1
for a, b in zip(words, words[1:]):
    cur = cur + 1 if a == b else 1
    mx = max(mx, cur)
top, n = collections.Counter(words).most_common(1)[0]
snip = text[:90].replace("\n", " ")
if mx >= 8 or n > 0.4 * len(words):
    print(f"FAIL|repetition collapse (x{mx} {top!r})|{snip}")
else:
    print(f"PASS|coherent|{snip}")
PY
judge() { python3 "$JUDGE"; }

printf '%-5s  %-5s  %-26s  %s\n' MODEL VERDICT REASON SNIPPET
printf '%-5s  %-5s  %-26s  %s\n' ----- ------- -------------------------- -------
keys=("$@"); [ ${#keys[@]} -eq 0 ] && keys=("${ORDER[@]}")
pass=0; total=0
for k in "${keys[@]}"; do
	snap="$(resolve "$k")"
	total=$((total + 1))
	if [ -z "$snap" ]; then
		printf '%-5s  %-5s  %s\n' "$k" "MISS" "not cached"; continue
	fi
	# Cap the context to what a short smoke needs: the native KV cache allocs maxLen·kvDim PER LAYER
	# (sliding layers don't shrink to their window), so the model-default context (gemma4 = 128K-256K)
	# allocs tens of GB on the big models (26b OOM'd silently). The smoke generates ~MAXTOK tokens, so
	# a small cap is correct here; full-context serving sizing is a separate concern.
	out="$("$BIN" generate --native --temp 0 --context "$((MAXTOK + 1024))" --max-tokens "$MAXTOK" --prompt "$PROMPT" "$snap" 2>&1)"
	res="$(printf '%s' "$out" | judge)"
	v="${res%%|*}"; rest="${res#*|}"; reason="${rest%%|*}"; snip="${rest#*|}"
	[ "$v" = PASS ] && pass=$((pass + 1))
	printf '%-5s  %-5s  %-26s  %s\n' "$k" "$v" "$reason" "$snip"
done
echo "---"
echo "native real-model gate: $pass/$total coherent"
