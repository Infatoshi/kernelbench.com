/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  // IMPORTANT: do NOT set allowedDevOrigins unless the list is complete for
  // every host the Mac might use (LAN / Tailscale / ZeroTier / short name).
  // When the key is present Next switches from warn → block and 403s /_next/*
  // for any unlisted Origin ("Unauthorized"), which looks like a blank page
  // or a Next.js error overlay from the Mac. Leaving it unset keeps remote
  // preview working on any network path (LAN / Tailscale / localhost).
  experimental: {
    // Persist the Turbopack module graph across runs — warm `next dev` skips
    // recompiling unchanged subtrees. (ForBuild exists but is canary-gated in
    // 16.0.10; revisit on upgrade.)
    turbopackFileSystemCacheForDev: true,
  },
  // Old KernelBench-v3 site paths — product is retired from the website.
  async redirects() {
    return [
      { source: "/v3", destination: "/", permanent: true },
      { source: "/v3/:path*", destination: "/", permanent: true },
      { source: "/blog/v3", destination: "/blog", permanent: true },
      { source: "/blog/v3/:path*", destination: "/blog", permanent: true },
      { source: "/kernelbench-v3", destination: "/", permanent: true },
      { source: "/kernelbench-v3/:path*", destination: "/", permanent: true },
      { source: "/data/v3/:path*", destination: "/", permanent: true },
      // Mega transcript viewers used to ship as public/runs/<run_id>.html; the
      // traces live only on HF now (2026-09-24), so old links go there.
      {
        source: "/runs/20260721_143203_codex_gpt-5.6-sol_02_kimi_linear_decode.html",
        destination:
          "https://huggingface.co/datasets/Infatoshi/kernelbench-mega-traces/blob/main/b200/20260721_143203_codex_gpt-5.6-sol_02_kimi_linear_decode.jsonl",
        permanent: true,
      },
      {
        source: "/runs/:rid(\\d{8}_\\d{6}_.+_kimi_linear_decode)\\.html",
        destination: "https://huggingface.co/datasets/Infatoshi/kernelbench-mega-traces/blob/main/:rid.jsonl",
        permanent: true,
      },
    ]
  },
  // data.ts reads leaderboards/annotations with dynamic fs paths, so the file
  // tracer can't narrow the pattern and matches the whole repo — including the
  // large gitignored benchmarks/*/outputs run archives (thin ~20G on Mac,
  // fat caches on workers). Scope the trace to the small result/data dirs
  // the pages actually read. Vercel is unaffected either way (archives
  // aren't in the checkout).
  outputFileTracingExcludes: {
    "*": [
      "benchmarks/*/outputs/**",
      "benchmarks/*/problems*/**",
      "benchmarks/v3/**",
      "benchmarks/multi/**",
      "runs/**",
      "media/**",
      "kbtool/**",
      "environments/**",
    ],
  },
  outputFileTracingIncludes: {
    "*": [
      "benchmarks/hard/results/**",
      "benchmarks/cuda/results/**",
      "benchmarks/mega/results/**",
      "public/data/**",
      "public/runs/**",
    ],
  },
}

export default nextConfig
