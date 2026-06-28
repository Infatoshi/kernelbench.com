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
    <html lang="en" className={mono.variable} data-theme="dark">
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
            <a href="/mega">mega</a>
            <a href="/hard">hard</a>
            <a
              href="https://github.com/Infatoshi/kernelbench.com"
              target="_blank"
              rel="noreferrer"
              className="github-icon-link no-underline"
              aria-label="GitHub repository"
              title="GitHub repository"
            >
              <svg
                viewBox="0 0 24 24"
                aria-hidden="true"
                className="github-icon"
                fill="currentColor"
              >
                <path d="M12 2C6.48 2 2 6.58 2 12.26c0 4.53 2.87 8.37 6.84 9.73.5.1.68-.22.68-.49 0-.24-.01-1.04-.01-1.89-2.78.62-3.37-1.22-3.37-1.22-.45-1.19-1.11-1.5-1.11-1.5-.91-.64.07-.63.07-.63 1 .07 1.53 1.06 1.53 1.06.9 1.57 2.35 1.12 2.92.86.09-.67.35-1.12.63-1.38-2.22-.26-4.55-1.14-4.55-5.07 0-1.12.39-2.03 1.03-2.75-.1-.26-.45-1.3.1-2.71 0 0 .84-.28 2.75 1.05A9.29 9.29 0 0 1 12 6.98c.85 0 1.71.12 2.51.34 1.91-1.33 2.75-1.05 2.75-1.05.55 1.41.2 2.45.1 2.71.64.72 1.03 1.63 1.03 2.75 0 3.94-2.34 4.81-4.57 5.06.36.32.68.94.68 1.9 0 1.38-.01 2.49-.01 2.83 0 .27.18.59.69.49A10.05 10.05 0 0 0 22 12.26C22 6.58 17.52 2 12 2Z" />
              </svg>
            </a>
          </nav>
        </div>
      </div>
    </header>
  )
}

function Footer() {
  return (
    <footer className="border-t border-[var(--color-border)] mt-16">
      <div className="container mx-auto max-w-7xl px-4 sm:px-6 py-6 text-xs text-[var(--color-fg-muted)] flex flex-col gap-4">
        <div className="flex flex-col sm:flex-row gap-2 sm:items-center sm:justify-between">
          <span>
            built by{" "}
            <a href="https://elliotarledge.com">elliot arledge</a>
            {" · "}
            <a href="mailto:elliot@arledge.net">elliot@arledge.net</a>
          </span>
          <span>
            source:{" "}
            <a href="https://github.com/Infatoshi/kernelbench.com">
              github.com/Infatoshi/kernelbench.com
            </a>
          </span>
        </div>
        <p className="text-[var(--color-fg-muted)] leading-relaxed">
          Disclaimer: This site is not affiliated with or endorsed by the
          authors of Stanford KernelBench. It is an independent website and hub
          for benchmark runs made by Elliot Arledge (
          <a href="https://x.com/elliotarledge">x.com/elliotarledge</a>
          ).
        </p>
      </div>
    </footer>
  )
}
