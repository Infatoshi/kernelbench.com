import type { Metadata } from "next"
import { JetBrains_Mono } from "next/font/google"
import "./globals.css"
import { ThemeToggle } from "./theme-toggle"

const mono = JetBrains_Mono({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-mono-loaded",
})

export const metadata: Metadata = {
  metadataBase: new URL("https://kernelbench.com"),
  title: "kernelbench.com: Agentic GPU Kernel Benchmark Results",
  description:
    "Open agentic GPU kernel benchmark results, repositories, transcripts, and datasets.",
  authors: [{ name: "Elliot Arledge", url: "https://elliotarledge.com" }],
  creator: "Elliot Arledge",
  publisher: "kernelbench.com",
  keywords: [
    "GPU kernels",
    "CUDA",
    "benchmark",
    "coding agents",
    "LLM evaluation",
    "agentic GPU kernels",
  ],
  openGraph: {
    title: "kernelbench.com: Agentic GPU Kernel Benchmark Results",
    description:
      "Open agentic GPU kernel benchmark results, repositories, transcripts, and datasets.",
    url: "https://kernelbench.com",
    siteName: "kernelbench.com",
  },
  other: {
    citation_title:
      "kernelbench.com: Agentic GPU Kernel Benchmark Results and Run Artifacts",
    citation_author: "Arledge, Elliot",
    citation_publication_date: "2026",
    citation_online_date: "2026",
    citation_fulltext_html_url: "https://kernelbench.com",
    citation_keywords:
      "GPU kernels; CUDA; autonomous coding agents; LLM evaluation; benchmark",
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={mono.variable} data-theme="dark" suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `try{var t=localStorage.getItem("kb-theme");document.documentElement.dataset.theme=t==="light"?"light":"dark"}catch(e){document.documentElement.dataset.theme="dark"}`,
          }}
        />
      </head>
      <body className="min-h-screen">
        <Header />
        <main className="container mx-auto px-4 sm:px-6 max-w-7xl py-10">
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
      <div className="container mx-auto max-w-7xl px-4 sm:px-6 py-4 flex flex-wrap items-center justify-between gap-x-6 gap-y-3">
        <a
          href="/"
          className="font-semibold text-[var(--color-fg-bright)] no-underline"
        >
          kernelbench.com
        </a>
        <div className="ml-auto flex items-center gap-3 sm:gap-5">
          <nav className="flex items-center gap-3 sm:gap-5 text-sm text-[var(--color-fg-muted)]">
            <a href="/v3">v3</a>
            <a href="/hard">hard</a>
            <a href="/runs">runs</a>
            <a href="/#cite">cite</a>
            <a href="/blog">blog</a>
            <a
              href="https://github.com/Infatoshi/kernelbench.com"
              target="_blank"
              rel="noreferrer"
            >
              github
            </a>
          </nav>
          <ThemeToggle />
        </div>
      </div>
    </header>
  )
}

function Footer() {
  return (
    <footer className="border-t border-[var(--color-border)] mt-16">
      <div className="container mx-auto max-w-7xl px-4 sm:px-6 py-6 text-xs text-[var(--color-fg-muted)] flex flex-col sm:flex-row gap-2 sm:items-center sm:justify-between">
        <span>
          built by{" "}
          <a href="https://elliotarledge.com">elliot arledge</a>
          {" · "}
          <a href="mailto:infatoshi@gmail.com">infatoshi@gmail.com</a>
        </span>
        <span>
          source:{" "}
          <a href="https://github.com/Infatoshi/kernelbench.com">
            github.com/Infatoshi/kernelbench.com
          </a>
        </span>
      </div>
    </footer>
  )
}
