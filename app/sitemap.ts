import type { MetadataRoute } from "next"

const routes = [
  { path: "", priority: 1.0 },
  { path: "/hard", priority: 0.9 },
  { path: "/mega", priority: 0.9 },
  { path: "/cuda", priority: 0.85 },
  { path: "/multi", priority: 0.8 },
  { path: "/v3", priority: 0.8 },
  { path: "/code", priority: 0.7 },
  { path: "/blog", priority: 0.7 },
  { path: "/blog/hard", priority: 0.6 },
  { path: "/blog/v3", priority: 0.6 },
]

export default function sitemap(): MetadataRoute.Sitemap {
  const baseUrl = "https://kernelbench.com"
  const lastModified = new Date("2026-07-06")

  return routes.map(({ path, priority }) => ({
    url: `${baseUrl}${path}`,
    lastModified,
    changeFrequency: "weekly",
    priority,
  }))
}
