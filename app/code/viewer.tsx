"use client"

import { useEffect, useState } from "react"
import { useSearchParams } from "next/navigation"

const PRISM_CDN = "https://cdn.jsdelivr.net/npm/prismjs@1.29.0"

// Normalize through URL so encoded dot segments (%2e%2e) can't escape the
// allowed prefixes, and return the normalized same-origin pathname.
function normalizeAllowed(f: string): string | null {
  let url: URL
  try {
    url = new URL(f, "https://kernelbench.com")
  } catch {
    return null
  }
  if (url.origin !== "https://kernelbench.com") return null
  const path = decodeURIComponent(url.pathname)
  if (path.includes("..")) return null
  if (!path.startsWith("/data/") && !path.startsWith("/runs/")) return null
  return url.pathname
}

function languageFor(f: string) {
  const path = f.replace(/\.txt$/, "")
  if (path.endsWith(".py")) return "python"
  if (/\.(cu|cuh|cpp|cc|h|hpp)$/.test(path)) return "cpp"
  if (path.endsWith(".json")) return "json"
  if (/\.(yaml|yml)$/.test(path)) return "yaml"
  return "none"
}

declare global {
  interface Window {
    Prism?: { highlightAll: () => void }
  }
}

function loadScript(src: string) {
  return new Promise<void>((resolve, reject) => {
    const s = document.createElement("script")
    s.src = src
    s.onload = () => resolve()
    s.onerror = () => reject(new Error(`failed to load ${src}`))
    document.body.appendChild(s)
  })
}

export default function CodeViewer() {
  const raw = useSearchParams().get("f") ?? ""
  const f = normalizeAllowed(raw) ?? ""
  const [text, setText] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!f) return
    let cancelled = false
    fetch(f)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.text()
      })
      .then((t) => {
        if (!cancelled) setText(t)
      })
      .catch((e) => {
        if (!cancelled) setError(String(e))
      })
    return () => {
      cancelled = true
    }
  }, [f])

  useEffect(() => {
    if (text === null) return
    if (window.Prism) {
      window.Prism.highlightAll()
      return
    }
    loadScript(`${PRISM_CDN}/components/prism-core.min.js`)
      .then(() => loadScript(`${PRISM_CDN}/plugins/autoloader/prism-autoloader.min.js`))
      .then(() => window.Prism?.highlightAll())
      .catch(() => {}) // highlighting is cosmetic; plain text is fine
  }, [text])

  if (!f) {
    return (
      <p className="text-sm text-[var(--color-bad)]">
        invalid or missing file path.
      </p>
    )
  }

  const name = f.split("/").pop() ?? f

  return (
    <section>
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1 mb-4">
        <h1 className="font-mono text-base text-[var(--color-fg-bright)] break-all">
          {name}
        </h1>
        <a href={f} className="text-sm" title="open the raw file">
          raw
        </a>
      </div>
      {error ? (
        <p className="text-sm text-[var(--color-bad)]">
          failed to load {f}: {error}
        </p>
      ) : text === null ? (
        <p className="text-sm text-[var(--color-fg-muted)]">loading…</p>
      ) : (
        <pre className="code-viewer-pre">
          <code className={`language-${languageFor(f)}`}>{text}</code>
        </pre>
      )}
    </section>
  )
}
