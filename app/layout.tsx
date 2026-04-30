import type { Metadata } from "next"
import { JetBrains_Mono } from "next/font/google"
import "./globals.css"

const mono = JetBrains_Mono({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-mono-loaded",
})

export const metadata: Metadata = {
  metadataBase: new URL("https://kernelbench.com"),
  title: "KernelBench",
  description:
    "GPU kernel engineering benchmarks for autonomous LLM coding agents. v3 (multi-GPU, 2026-Q1) and v-Hard (single Blackwell, 2026-Q2).",
  openGraph: {
    title: "KernelBench",
    description:
      "GPU kernel engineering benchmarks for autonomous LLM coding agents.",
    url: "https://kernelbench.com",
    siteName: "KernelBench",
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={mono.variable}>
      <body className="min-h-screen">
        <Header />
        <main className="container mx-auto px-4 sm:px-6 max-w-5xl py-10">
          {children}
        </main>
        <Footer />
      </body>
    </html>
  )
}

function Header() {
  return (
    <header className="border-b border-[var(--color-border)]">
      <div className="container mx-auto max-w-5xl px-4 sm:px-6 py-4 flex items-center justify-between gap-6">
        <a href="/" className="font-bold text-[var(--color-fg-bright)] no-underline">
          ./kernelbench
        </a>
        <nav className="flex items-center gap-5 text-sm">
          <a href="/v3">v3</a>
          <a href="/hard">v-hard</a>
          <a href="/runs">runs</a>
          <a
            href="https://github.com/Infatoshi/kernelbench.com"
            target="_blank"
            rel="noreferrer"
          >
            github
          </a>
        </nav>
      </div>
    </header>
  )
}

function Footer() {
  return (
    <footer className="border-t border-[var(--color-border)] mt-16">
      <div className="container mx-auto max-w-5xl px-4 sm:px-6 py-6 text-xs text-[var(--color-fg-muted)] flex flex-col sm:flex-row gap-2 sm:items-center sm:justify-between">
        <span>
          built by{" "}
          <a href="https://elliotarledge.com">elliot arledge</a>
        </span>
        <span>
          source:{" "}
          <a href="https://github.com/Infatoshi/KernelBench-Hard">
            github.com/Infatoshi/KernelBench-Hard
          </a>
        </span>
      </div>
    </footer>
  )
}
