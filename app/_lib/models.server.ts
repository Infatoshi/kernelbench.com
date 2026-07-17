import { readFile } from "node:fs/promises"
import { join } from "node:path"
import type { ModelIndex } from "./models"

// Server-side loader for the model index. Kept separate from ./models so the
// shared types/transforms stay importable from client components.

/** Models pulled from the site entirely (every chart, /models, model pages).
 *  Filtered at load so `kb publish` regenerating models.json can't
 *  resurrect them. */
const REMOVED_MODEL_SLUGS = new Set(["gemini-3.1-pro-preview", "gemini-3.5-flash"])

// No module-level cache: Next dev (and prod workers) keep one module graph per
// route segment, so a `cached ??=` here pins each page to whatever models.json
// said at that segment's first request — the roster visibly desyncs across
// pages after a publish. The file is ~1 MB; reading it per request is noise.
export function loadModelIndex(): Promise<ModelIndex> {
  return readFile(join(process.cwd(), "public/data/models.json"), "utf8").then(
    (raw) => {
      const idx = JSON.parse(raw) as ModelIndex
      idx.models = idx.models.filter((m) => !REMOVED_MODEL_SLUGS.has(m.slug))
      return idx
    },
  )
}
