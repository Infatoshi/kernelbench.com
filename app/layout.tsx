import type { Metadata } from "next"
import Link from "next/link"
import { JetBrains_Mono } from "next/font/google"
import { SiteNav } from "@/app/_components/site-nav"
import "./globals.css"
import { ContactLink } from "./_components/contact-link"

const mono = JetBrains_Mono({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-mono-loaded",
})

export const metadata: Metadata = {
  metadataBase: new URL("https://kernelbench.com"),
  title: {
    default: "kernelbench.com — Audited Agentic GPU Kernel & CUDA Optimization Leaderboard",
    template: "%s · kernelbench.com",
  },
  description:
    "Independent, reward-hack audited leaderboard for frontier AI models optimizing GPU kernels and CUDA inference operators. Every cell includes redacted kernel source, per-shape hardware roofline measurements, audit verdicts, and full HuggingFace agent transcripts.",
  authors: [{ name: "Elliot Arledge", url: "https://elliotarledge.com" }],
  creator: "Elliot Arledge",
  publisher: "kernelbench.com",
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-snippet": -1,
      "max-image-preview": "large",
      "max-video-preview": -1,
    },
  },
  keywords: [
    "GPU kernel optimization benchmark",
    "CUDA kernel leaderboard",
    "agentic GPU kernels",
    "LLM inference engine optimization",
    "reward-hack audited benchmark",
    "fused MoE CUDA kernel",
    "DeepSeek NSA kernel",
    "Kimi-Linear megakernel",
    "frontier coding agents",
    "hardware roofline GPU benchmark",
    "kernelbench.com",
  ],
  openGraph: {
    type: "website",
    title: "kernelbench.com — Audited Agentic GPU Kernel & CUDA Optimization Leaderboard",
    description:
      "Independent, reward-hack audited GPU kernel and CUDA optimization benchmark for frontier coding agents. Full kernel source, hardware rooflines, and agent transcripts for every published result.",
    url: "https://kernelbench.com",
    siteName: "kernelbench.com",
  },
  twitter: {
    card: "summary",
    title: "kernelbench.com — Audited Agentic GPU Kernel & CUDA Optimization Leaderboard",
    description:
      "Reward-hack audited CUDA and fused megakernel benchmark for frontier AI models. Full source code, per-shape rooflines, and HuggingFace agent transcripts.",
    creator: "@elliotarledge",
  },
  other: {
    citation_title:
      "kernelbench.com: Audited Agentic GPU Kernel & CUDA Optimization Benchmark",
    citation_author: "Arledge, Elliot",
    citation_publication_date: "2026",
    citation_online_date: "2026",
    citation_fulltext_html_url: "https://kernelbench.com",
    citation_keywords:
      "GPU kernels; CUDA; inference engine optimization; autonomous coding agents; reward-hack audit; hardware roofline; benchmark",
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    // suppressHydrationWarning: browser extensions (Dark Reader, etc.) inject
    // attributes on <html>/<body> before React hydrates; without this the
    // Mac preview shows a noisy hydration overlay that is not app state.
    <html lang="en" className={mono.variable} data-theme="dark" suppressHydrationWarning>
      <body className="min-h-screen" suppressHydrationWarning>
        <SiteNav />
        <main className="container mx-auto px-4 sm:px-6 max-w-7xl pt-6 pb-12">
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
          <ContactLink />
          {" · "}
          <Link href="/models">models</Link>
          {" · "}
          <Link href="/runs">runs</Link>
          {" · "}
          <Link href="/hard">hard archive</Link>
          {" · "}
          <a href="/llms.txt">llms.txt</a>
          {" · "}
          <a href="https://github.com/Infatoshi/kernelbench.com">source</a>
        </span>
        <span className="site-footer-dim">
          independent canonical benchmark — not affiliated with Stanford KernelBench
        </span>
      </div>
    </footer>
  )
}
