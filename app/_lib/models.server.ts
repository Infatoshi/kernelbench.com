import { readFile } from "node:fs/promises"
import { join } from "node:path"
import type { ModelIndex } from "./models"

// Server-side loader for the model index. Kept separate from ./models so the
// shared types/transforms stay importable from client components.

/** Current competitive models shown on the canonical RTX PRO 6000 homepage boards.
 * When a newer model ships, add it here and move the superseded sibling to
 * RETIRED_MODEL_SLUGS below. Historical model pages and audit logs still load
 * the full index. */
export const LIVE_MODEL_SLUGS = new Set([
  "claude-fable-5-2",
  "claude-haiku-5-5",
  "claude-opus-5-5",
  "claude-sonnet-5-5",
  "deepseek-flash",
  "gemini-4-argon-high",
  "gemini-4-carbon-high",
  "gemini-4-neon-xhigh",
  "glm-5.3",
  "gpt-6.2-astra",
  "qwen4-max",
])

/** Models with published cells that are deliberately off the main homepage
 * boards (superseded siblings or retired entries). Every slug in models.json
 * must be here or in LIVE_MODEL_SLUGS (or CHART_HIDDEN_SLUGS);
 * scripts/check_publish_gates.py enforces this on publish. */
export const RETIRED_MODEL_SLUGS = new Set([
  "claude-fable-5",
  "claude-fable-5-1",
  "claude-opus-4-8",
  "claude-opus-5",
  "claude-sonnet-5",
  "composer-2.5-fast",
  "deepseek-v4-flash-0731",
  "deepseek-v4-pro",
  "deepseek-v4.1-flash-1017",
  "gemini-3.8-flash-high",
  "glm-5.2",
  "gpt-5.6-sol",
  "gpt-6-astra-pro",
  "gpt-6-luna",
  "gpt-6-sol",
  "gpt-6.1-sol",
  "grok-4.5",
  "grok-4.6",
  "grok-4.7",
  "hy3",
  "inkling",
  "k3.1-pro",
  "kinetic-0715",
  "kinetic-0715-1m",
  "longcat-2.0",
  "minimax-m3",
  "muse-spark-1.3",
  "ox-alpha",
  "qwen3.8-max",
  "qwen3.8-max-preview",
])

// No module-level cache: Next dev (and prod workers) keep one module graph per
// route segment, so a `cached ??=` here pins each page to whatever models.json
// said at that segment's first request — the roster visibly desyncs across
// pages after a publish. The file is ~1 MB; reading it per request is noise.
export function loadAllModelIndex(): Promise<ModelIndex> {
  return readFile(join(process.cwd(), "public/data/models.json"), "utf8").then(
    (raw) => JSON.parse(raw) as ModelIndex,
  )
}

export function loadModelIndex(): Promise<ModelIndex> {
  return loadAllModelIndex().then((idx) => {
    idx.models = idx.models.filter((m) => LIVE_MODEL_SLUGS.has(m.slug))
    return idx
  })
}
