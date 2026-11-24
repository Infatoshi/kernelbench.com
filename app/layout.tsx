import type { Metadata } from "next"
import Link from "next/link"
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
    <html
      lang="en"
      className={mono.variable}
      data-theme="dark"
      suppressHydrationWarning
    >
      <body className="min-h-screen" suppressHydrationWarning>
        <div
          className="kb-mast"
          style={{
            background: "#000000",
            borderBottom: "1px solid #333333",
            padding: "14px 24px",
          }}
        >
          <Link
            href="/"
            className="kb-brand no-underline"
            style={{
              fontFamily:
                'var(--font-mono-loaded), "SF Mono", ui-monospace, "JetBrains Mono", Menlo, Consolas, monospace',
              fontSize: "15px",
              fontWeight: 600,
              letterSpacing: "-0.01em",
              color: "#ffffff",
            }}
          >
            kernelbench.com
          </Link>
        </div>
        {children}
      </body>
    </html>
  )
}
