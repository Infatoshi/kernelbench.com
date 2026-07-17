import type { Metadata } from "next"
import { JetBrains_Mono } from "next/font/google"
import { SiteNav } from "@/app/_components/site-nav"
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
        <SiteNav />
        <main className="container mx-auto px-4 sm:px-6 max-w-7xl pt-8 pb-12">
          {children}
        </main>
        <Footer />
      </body>
    </html>
  )
}

function Footer() {
  return (
    <footer className="site-footer">
      <div className="site-footer-inner">
        <span className="site-footer-line">
          built by <a href="https://elliotarledge.com">elliot arledge</a>
          {" · "}
          <a href="mailto:elliot@arledge.net">elliot@arledge.net</a>
          {" · "}
          <a href="https://github.com/Infatoshi/kernelbench.com">source</a>
        </span>
        <span className="site-footer-dim">
          independent site — not affiliated with Stanford KernelBench
        </span>
      </div>
    </footer>
  )
}
