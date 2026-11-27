import { permanentRedirect } from "next/navigation"

// The per-bench pages collapsed into the homepage decks (2026-07-22): one
// site, one board per bench section, one page per run. Kept as a permanent
// redirect so old external links (posts, README) consolidate onto /.

export default function Page() {
  permanentRedirect("/#multi")
}
