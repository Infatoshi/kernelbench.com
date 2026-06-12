# kernelbench.com — operator recipes. Run from repo root on Anvil.
# `just` with no args lists recipes.

default:
    @just --list

# Full 6-problem sweep of one harness+model, parallel containers, 2700s budget.
sweep harness model effort="":
    cd benchmarks/hard && ./scripts/sweep_deck.sh "{{harness}}" "{{model}}" "{{effort}}"

# One problem (smoke / rerun). budget overridable via BUDGET_SECONDS env.
run harness model problem effort="":
    cd benchmarks/hard && KBH_AGENT_CONTAINER=1 uv run kbh run "{{harness}}" "{{model}}" "problems/{{problem}}" "{{effort}}"

# Rebuild leaderboard + transcript viewers from the run archives (no push).
publish:
    cd benchmarks/hard && ./scripts/publish_v2.sh

# Preview the site locally (view from Mac over Tailscale: anvil:3000).
dev:
    npm run dev

build:
    npm run build

# Deploy: publish data, commit, push (Vercel auto-builds). Pass a message.
deploy msg:
    just publish
    git add -A benchmarks/hard/results public/runs app
    git -c user.email=elliot@arledge.net commit -m "{{msg}}"
    git push origin master
