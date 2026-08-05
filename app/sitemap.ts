import type { MetadataRoute } from "next"

const routes = [
  { path: "", priority: 1.0 },
  { path: "/models", priority: 0.9 },
  { path: "/code", priority: 0.7 },
]

export default function sitemap(): MetadataRoute.Sitemap {
  const baseUrl = "https://kernelbench.com"
  const lastModified = new Date("2026-07-17")

  return routes.map(({ path, priority }) => ({
    url: `${baseUrl}${path}`,
    lastModified,
    changeFrequency: "weekly" as const,
    priority,
  }))
}
