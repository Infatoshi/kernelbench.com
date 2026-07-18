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
  // preview working on any network path to anvil:3000.
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
    ]
  },
  // data.ts reads leaderboards/annotations with dynamic fs paths, so the file
  // tracer can't narrow the pattern and matches the whole repo — including the
  // ~186G gitignored benchmarks/*/outputs run archives that exist only on
  // anvil (3.7M files, ~4min of every local build). Scope the trace to the
  // small result/data dirs the pages actually read. Vercel is unaffected
  // either way (archives aren't in the checkout).
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
