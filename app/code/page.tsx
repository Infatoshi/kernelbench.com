import { Suspense } from "react"
import type { Metadata } from "next"
import CodeViewer from "./viewer"

export const metadata: Metadata = {
  title: "code viewer — kernelbench.com",
  robots: { index: false },
}

export default function CodePage() {
  return (
    <div className="container mx-auto px-4 sm:px-6 max-w-7xl pt-6 pb-12">
      <Suspense
        fallback={
          <p className="text-sm text-[var(--color-fg-muted)]">loading…</p>
        }
      >
        <CodeViewer />
      </Suspense>
    </div>
  )
}
