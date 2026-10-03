#!/usr/bin/env bash
# One-shot RTX PRO 6000 recheck for branch bench/recheck-04-07, driven from the Mac.
#
# The branch changes two graded surfaces: cuda 04 check.py/problem.yaml (graded
# shapes + 1e-5 near-tie margin rule) and hard 07 PROMPT.txt/reference.py
# (integer zero-point contract). Every published RTX cell on those two problems
# is stale under publish gate E until it is replayed check-only on an RTX PRO
# 6000. The same GPU slot also runs the same-buffer overwrite probes the
# GPT-6.1 Sol audits are waiting on (cuda 03, cuda 04, mega 02).
#
#   scripts/recheck_04_07.sh prep            # no GPU: ship the branch decks, verify digests and archives
#   scripts/recheck_04_07.sh launch <gpu>    # refuses a GPU with any compute PID; detached on tetra
#   scripts/recheck_04_07.sh status          # tail the tetra log
#   scripts/recheck_04_07.sh pull            # result.json + check logs + probe log back to the Mac
#
# Nothing here merges, publishes or pushes. After `pull`: in a worktree of the
# branch, `python3 scripts/check_publish_gates.py --list-stale hard cuda` must be
# empty, then merge and publish as usual (root AGENTS.md).
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
BRANCH=${RECHECK_BRANCH:-bench/recheck-04-07}
HOST=${RECHECK_HOST:-tetra}
TR=${RECHECK_TETRA_ROOT:-kernelbench.com}          # tetra checkout, relative to ~
DECK_REL=outputs/recheck-deck/problems-rtxpro6000   # overlay deck root inside each bench (gitignored outputs/)
# Cells that are not on the board yet but must carry the branch's 04 stamp.
EXTRA_CUDA=${RECHECK_EXTRA_CUDA:-20260930_115512_codex_gpt-6.1-sol_04_grid_mingru_sps}
PROBES=(cuda:20260930_133214_codex_gpt-6.1-sol_03_megaqwen_decode:cuda03
        cuda:20260930_115512_codex_gpt-6.1-sol_04_grid_mingru_sps:cuda04
        mega:20260930_115311_codex_gpt-6.1-sol_02_kimi_linear_decode:mega02)
STATE=$R/benchmarks/cuda/outputs/recheck-04-07
mkdir -p "$STATE"

cells() {  # bench problem -> published run ids on the RTX board
    python3 - "$R/benchmarks/$1/results/leaderboard.json" "$2" <<'PY'
import json, sys
lb = json.load(open(sys.argv[1]))
for m in lb.get("models", []):
    c = (m.get("results") or {}).get(sys.argv[2])
    if c and c.get("run_id"):
        print(c["run_id"])
PY
}

digest() {  # deck_dir src_dir -> graded surface sha (same module as gate E)
    python3 -c "import sys; sys.path.insert(0, sys.argv[3]); import graded_surface as g; print(g.graded_surface_digest(sys.argv[1], sys.argv[2]))" "$1" "$2" "$3"
}

case "${1:-}" in
prep)
    git -C "$R" rev-parse --verify -q "$BRANCH" >/dev/null || { echo "no branch $BRANCH"; exit 1; }
    tmp=$(mktemp -d "$STATE/branch.XXXX"); trap 'rm -rf "$tmp"' EXIT
    git -C "$R" archive "$BRANCH" benchmarks/cuda/problems-rtxpro6000/04_grid_mingru_sps \
        benchmarks/hard/problems-rtxpro6000/07_w4a16_gemm benchmarks/cuda/src benchmarks/hard/src \
        scripts/lib/graded_surface.py scripts/probe_same_buffer.py | tar -x -C "$tmp"
    : > "$STATE/expected.txt"
    for bp in cuda:04_grid_mingru_sps hard:07_w4a16_gemm; do
        b=${bp%%:*}; p=${bp#*:}
        want=$(digest "$tmp/benchmarks/$b/problems-rtxpro6000/$p" "$tmp/benchmarks/$b/src" "$tmp/scripts/lib")
        echo "$b $p $want" >> "$STATE/expected.txt"
        ssh "$HOST" "mkdir -p ~/$TR/benchmarks/$b/$DECK_REL"
        rsync -a --delete "$tmp/benchmarks/$b/problems-rtxpro6000/$p/" "$HOST:$TR/benchmarks/$b/$DECK_REL/$p/"
        got=$(ssh "$HOST" "cd ~/$TR && python3 -c \"import sys; sys.path.insert(0,'scripts/lib'); import graded_surface as g; print(g.graded_surface_digest('benchmarks/$b/$DECK_REL/$p','benchmarks/$b/src'))\"")
        [ "$got" = "$want" ] || { echo "STOP: $b/$p digest on $HOST ($got) != branch ($want); tetra src differs from the branch"; exit 1; }
        echo "ok   $b/$p overlay deck on $HOST matches branch digest ${want:0:12}"
    done
    rsync -a "$tmp/scripts/probe_same_buffer.py" "$HOST:$TR/benchmarks/cuda/outputs/recheck-deck/probe_same_buffer.py"
    { cells cuda 04_grid_mingru_sps; for x in $EXTRA_CUDA; do echo "$x"; done; } | sort -u > "$STATE/cuda.txt"
    cells hard 07_w4a16_gemm | sort -u > "$STATE/hard.txt"
    for b in cuda hard; do
        missing=$(ssh "$HOST" "cd ~/$TR/benchmarks/$b/outputs/runs && for r in $(tr '\n' ' ' < "$STATE/$b.txt"); do [ -f \$r/result.json ] && [ -f \$r/solution.py ] || echo \$r; done")
        echo "$b: $(wc -l < "$STATE/$b.txt" | tr -d ' ') cells, missing on $HOST: ${missing:-none}"
        [ -z "$missing" ] || { echo "STOP: pull or locate the missing archives first"; exit 1; }
    done
    scp -q "$STATE/cuda.txt" "$STATE/hard.txt" "$STATE/expected.txt" "$HOST:$TR/benchmarks/cuda/$DECK_REL/../"
    echo "prep done; next: $0 launch <gpu> once that GPU shows no compute PIDs"
    ;;
launch)
    GPU=${2:?usage: launch <gpu>}
    busy=$(ssh "$HOST" "nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader")
    [ -z "$busy" ] || { echo "STOP: GPU $GPU on $HOST has compute PIDs: $busy (never touch another job)"; exit 1; }
    stamp=$(date +%Y%m%d_%H%M%S)
    # The remote job: lease the GPU, cuda 04 then hard 07 check-only replays, then
    # the three overwrite probes, each in its archived workspace.
    ssh "$HOST" "cat > ~/$TR/benchmarks/cuda/outputs/recheck-deck/job.sh" <<EOF
#!/usr/bin/env bash
set -uo pipefail
export PATH=\$HOME/.local/bin:/usr/local/cuda-13.3/bin:\$PATH KBH_CUDA_HOME=/usr/local/cuda-13
T=\$HOME/$TR; OUT=\$T/benchmarks/cuda/outputs/recheck-deck/run-$stamp; mkdir -p \$OUT
exec >> \$OUT/job.log 2>&1
echo "=== \$(date '+%F %T %Z') start gpu$GPU"
for b in cuda hard; do
  ( cd \$T/benchmarks/\$b && KBH_REGRADE_GPU=$GPU KBH_REGRADE_CHECK_ONLY=1 KBH_REGRADE_DECK=$DECK_REL \
      ./scripts/regrade_sequential.sh \$(sed 's#^#outputs/runs/#' \$T/benchmarks/cuda/outputs/recheck-deck/\$b.txt) ) > \$OUT/recheck_\$b.log 2>&1
  echo "=== \$(date '+%F %T %Z') recheck \$b exit \$?"
done
for spec in ${PROBES[*]}; do
  b=\${spec%%:*}; rest=\${spec#*:}; rid=\${rest%%:*}; mode=\${rest#*:}
  rd=\$T/benchmarks/\$b/outputs/runs/\$rid; pd=\$(ls -d \$rd/repo/problems/*/ | head -1)
  ( cd \$pd && CUDA_VISIBLE_DEVICES=$GPU TORCH_EXTENSIONS_DIR=\$rd/cache/torch_extensions_probe \
      uv run python \$T/benchmarks/cuda/outputs/recheck-deck/probe_same_buffer.py \$mode ) > \$OUT/probe_\$mode.log 2>&1
  echo "=== \$(date '+%F %T %Z') probe \$mode exit \$?: \$(grep -h 'cos(' \$OUT/probe_\$mode.log | tr '\n' ' ')"
  source \$T/scripts/lib/strip_run_venv.sh && strip_run_venv \$rd
done
echo "=== \$(date '+%F %T %Z') RECHECK_DONE"; touch \$OUT/DONE
EOF
    ssh "$HOST" "chmod +x ~/$TR/benchmarks/cuda/outputs/recheck-deck/job.sh && cd ~/$TR && setsid nohup overnight-compute run --agent kb-recheck-04-07-gpu$GPU --resource gpu$GPU --ttl 30m --heartbeat 5m --timeout 2m -- benchmarks/cuda/outputs/recheck-deck/job.sh > benchmarks/cuda/outputs/recheck-deck/launcher-$stamp.log 2>&1 < /dev/null & echo launched"
    echo "run-$stamp" > "$STATE/last_run"
    echo "log: $HOST:~/$TR/benchmarks/cuda/outputs/recheck-deck/run-$stamp/job.log"
    ;;
status)
    run=$(cat "$STATE/last_run"); ssh "$HOST" "tail -20 ~/$TR/benchmarks/cuda/outputs/recheck-deck/$run/job.log; ls ~/$TR/benchmarks/cuda/outputs/recheck-deck/$run/"
    ;;
pull)
    run=$(cat "$STATE/last_run")
    ssh "$HOST" "test -f ~/$TR/benchmarks/cuda/outputs/recheck-deck/$run/DONE" || { echo "not done yet; $0 status"; exit 1; }
    rsync -a "$HOST:$TR/benchmarks/cuda/outputs/recheck-deck/$run/" "$STATE/$run/"
    for b in cuda hard; do
        files=$(mktemp); while read -r rid; do printf '%s\n' "$rid/result.json" "$rid/check.log" "$rid/check.contended.log"; done < "$STATE/$b.txt" > "$files"
        rsync -a --ignore-missing-args --files-from="$files" "$HOST:$TR/benchmarks/$b/outputs/runs/" "$R/benchmarks/$b/outputs/runs/"; rm -f "$files"
    done
    grep -hE "correct=|FAIL|probe|RECHECK_DONE" "$STATE/$run"/*.log | tail -60
    echo "next: worktree of $BRANCH -> python3 scripts/check_publish_gates.py --list-stale hard cuda (must be empty), annotate any FAIL, then merge"
    ;;
*) sed -n 2,19p "$0"; exit 2 ;;
esac
