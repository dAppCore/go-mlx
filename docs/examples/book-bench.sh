#!/bin/bash
# SPDX-Licence-Identifier: EUPL-1.2
#
# book-bench.sh — the multi-turn book demo: one OpenAI-compatible endpoint
# writes a ten-chapter book, one chapter per turn, with the full conversation
# resent every turn — the honest agent-workflow shape. Engines that reuse
# prompt state prefill only the new tokens each turn; engines that don't
# re-read the whole book so far, every turn.
#
# The same script drives every engine (lthn-mlx serve, llama-server,
# mlx_lm.server), so the comparison is the engine, not the harness:
#
#   lthn-mlx serve --model <snapshot> --addr 127.0.0.1:11434
#   book-bench.sh -a 127.0.0.1:11434 -l lthn-mlx -i C037
#
#   llama-server -m <model.gguf> --port 8082 --jinja
#   book-bench.sh -a 127.0.0.1:8082 -l llama.cpp -i C037
#
# Ideas come from creative-demo.json beside this script. Output is one book
# per run under -o (default /tmp/book-bench), plus a per-chapter timing line —
# the line IS the vhs .tape footage.

set -euo pipefail

ADDR="127.0.0.1:11434"
LABEL="engine"
IDEA="random"
CHAPTERS=10
MAXTOK=800
TEMP=0.8
OUTDIR="/tmp/book-bench"
QUIET=0
NOTHINK=0
IDEAS="$(cd "$(dirname "$0")" && pwd)/creative-demo.json"

usage() {
  cat >&2 <<EOF
Usage: book-bench.sh [-a addr] [-l label] [-i idea-id|random] [-c chapters]
                     [-t max-tokens] [-T temperature] [-o outdir] [-q] [-n]
  -a  endpoint host:port            (default 127.0.0.1:11434)
  -l  engine label for output/file  (default engine)
  -i  idea id from creative-demo.json, or "random"
  -c  chapters                      (default 10)
  -t  max_tokens per chapter        (default 800)
  -T  temperature                   (default 0.8)
  -o  output directory              (default /tmp/book-bench)
  -q  quiet: timing lines only, no chapter text
  -n  no-think: chat_template_kwargs.enable_thinking=false
EOF
  exit 2
}

while getopts "a:l:i:c:t:T:o:qnh" opt; do
  case "$opt" in
    a) ADDR="$OPTARG" ;;
    l) LABEL="$OPTARG" ;;
    i) IDEA="$OPTARG" ;;
    c) CHAPTERS="$OPTARG" ;;
    t) MAXTOK="$OPTARG" ;;
    T) TEMP="$OPTARG" ;;
    o) OUTDIR="$OPTARG" ;;
    q) QUIET=1 ;;
    n) NOTHINK=1 ;;
    *) usage ;;
  esac
done

[ -f "$IDEAS" ] || { echo "ideas file missing: $IDEAS" >&2; exit 1; }

if [ "$IDEA" = "random" ]; then
  IDEA=$(jq -r ".[$((RANDOM % $(jq 'length' "$IDEAS")))].id" "$IDEAS")
fi
PROMPT=$(jq -r --arg id "$IDEA" '.[] | select(.id == $id) | .prompt' "$IDEAS")
[ -n "$PROMPT" ] || { echo "unknown idea id: $IDEA" >&2; exit 1; }

mkdir -p "$OUTDIR"
BOOK="$OUTDIR/book-$LABEL-$IDEA.md"
HIST="$OUTDIR/.messages-$LABEL-$IDEA.json"
echo "[]" > "$HIST"
: > "$BOOK"

# Snider's two-prompt shape: chapter one sets the arc, every later turn
# continues it, the final turn lands the ending chapter one set up.
turn_prompt() {
  local n=$1
  if [ "$n" -eq 1 ]; then
    printf 'We are writing a %s chapter book from this idea: "%s". Write chapter one, setting the overall arc of the book.' "$CHAPTERS" "$PROMPT"
  elif [ "$n" -eq "$CHAPTERS" ]; then
    printf 'Please write the final chapter, taking inspiration from the book idea: "%s". Incorporate elements of previous chapters, and end the book as the ending your first chapter set up.' "$PROMPT"
  else
    printf 'Please write the next chapter, taking inspiration from the book idea: "%s". As the story progresses, incorporate elements of previous chapters while maintaining the overall arc set by chapter one.' "$PROMPT"
  fi
}

echo "── book-bench · $LABEL @ $ADDR · idea $IDEA · $CHAPTERS chapters · $MAXTOK tok/ch ──"
[ "$QUIET" -eq 1 ] || { echo "idea: $PROMPT"; echo; }

TOTAL_WALL=0
TOTAL_PROMPT=0
TOTAL_GEN=0

for n in $(seq 1 "$CHAPTERS"); do
  USER_MSG=$(turn_prompt "$n")
  jq --arg c "$USER_MSG" '. + [{role:"user", content:$c}]' "$HIST" > "$HIST.tmp" && mv "$HIST.tmp" "$HIST"

  PAYLOAD=$(jq -n --arg label "$LABEL" --argjson msgs "$(cat "$HIST")" \
    --argjson maxtok "$MAXTOK" --argjson temp "$TEMP" --argjson nothink "$NOTHINK" '
    {model: $label, messages: $msgs, max_tokens: $maxtok, temperature: $temp, stream: false}
    + (if $nothink == 1 then {chat_template_kwargs: {enable_thinking: false}} else {} end)')

  RESP_FILE="$OUTDIR/.resp-$LABEL.json"
  WALL=$(curl -sS -m 900 -o "$RESP_FILE" -w '%{time_total}' \
    -H 'Content-Type: application/json' \
    -d "$PAYLOAD" "http://$ADDR/v1/chat/completions")

  CONTENT=$(jq -r '.choices[0].message.content // empty' "$RESP_FILE")
  [ -n "$CONTENT" ] || { echo "ch $n: empty response — $(head -c 300 "$RESP_FILE")" >&2; exit 1; }
  PTOK=$(jq -r '.usage.prompt_tokens // 0' "$RESP_FILE")
  GTOK=$(jq -r '.usage.completion_tokens // 0' "$RESP_FILE")

  jq --arg c "$CONTENT" '. + [{role:"assistant", content:$c}]' "$HIST" > "$HIST.tmp" && mv "$HIST.tmp" "$HIST"
  printf '\n## Chapter %d\n\n%s\n' "$n" "$CONTENT" >> "$BOOK"

  TOTAL_WALL=$(echo "$TOTAL_WALL + $WALL" | bc)
  TOTAL_PROMPT=$((TOTAL_PROMPT + PTOK))
  TOTAL_GEN=$((TOTAL_GEN + GTOK))
  RATE=$(echo "scale=1; $GTOK / $WALL" | bc)

  [ "$QUIET" -eq 1 ] || { echo "$CONTENT"; echo; }
  printf 'ch %2d │ prompt %5d tok │ gen %4d tok │ %6.1fs │ total %7.1fs │ %s tok/s\n' \
    "$n" "$PTOK" "$GTOK" "$WALL" "$TOTAL_WALL" "$RATE"
done

AVG=$(echo "scale=1; $TOTAL_GEN / $TOTAL_WALL" | bc)
echo "──"
printf '%s · %s · %d chapters · prompt %d tok (resent history) · gen %d tok · wall %.1fs · %s gen tok/s\n' \
  "$LABEL" "$IDEA" "$CHAPTERS" "$TOTAL_PROMPT" "$TOTAL_GEN" "$TOTAL_WALL" "$AVG"
echo "book: $BOOK"
