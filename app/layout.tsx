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
        <main className="container mx-auto px-4 sm:px-6 max-w-7xl py-10">
          {children}
        </main>
        <Footer />
      </body>
    </html>
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
