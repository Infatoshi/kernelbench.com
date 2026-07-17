/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
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
