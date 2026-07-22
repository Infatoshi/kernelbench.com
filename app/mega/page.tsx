import { redirect } from "next/navigation"

// The per-bench pages collapsed into the homepage decks (2026-07-22): one
// site, one board per bench section, one page per run. Kept as a redirect so
// old external links (posts, README) still land on the right section.

export default function Page() {
  redirect("/#mega")
}
