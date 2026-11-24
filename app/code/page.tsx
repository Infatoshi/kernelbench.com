import { Suspense } from "react"
import type { Metadata } from "next"
import CodeViewer from "./viewer"

export const metadata: Metadata = {
  title: "code viewer — kernelbench.com",
  robots: { index: false },
}

export default function CodePage() {
  return (
    <Suspense
      fallback={
        <p className="text-sm text-[var(--color-fg-muted)]">loading…</p>
      }
    >
      <CodeViewer />
    </Suspense>
  )
}
