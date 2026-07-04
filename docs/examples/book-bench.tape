# book-bench.tape — one engine lane of the three-way book demo.
#
# Record each engine SEPARATELY (one engine on the GPU at a time keeps the
# numbers honest), then compose the lanes side-by-side in remotion with the
# wall clocks in frame. lthn-mlx goes on the left.
#
# Before recording, start the lane's engine:
#   lthn-mlx:  lthn-mlx serve --model <snapshot> --addr 127.0.0.1:11434 \
#                --context 16384 -kv-cache paged
#   llama.cpp: llama-server -m <model.gguf> --port 8082 -c 16384 -ngl 99 --jinja
#   mlx-lm:    uv tool run --from mlx-lm mlx_lm.server --model <repo> --port 8083
#
# Render:  vhs docs/examples/book-bench.tape
Output book-bench-lthn-mlx.gif

Set FontSize 14
Set Width 1200
Set Height 800
Set Theme "Catppuccin Mocha"
Set TypingSpeed 40ms

Type "docs/examples/book-bench.sh -a 127.0.0.1:11434 -l lthn-mlx -i C037_STORY_GLASS -n"
Enter

# Ten chapters of an e2b book ≈ 50s on the fixed-cache lane; bigger models and
# the other engines need proportionally longer — size per lane after a dry run.
Sleep 60s
