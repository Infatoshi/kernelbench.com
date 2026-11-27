import type { MetadataRoute } from "next"
import { loadAllModelIndex, RETIRED_MODEL_SLUGS } from "./_lib/models.server"
import { collectPublicRunEntries, SITE_URL } from "./_lib/seo.server"

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const idx = await loadAllModelIndex()
  const parsedDate = idx.generated ? new Date(idx.generated) : new Date()
  const lastModified = Number.isNaN(parsedDate.getTime()) ? new Date() : parsedDate

  const staticEntries: MetadataRoute.Sitemap = [
    {
      url: SITE_URL,
      lastModified,
      changeFrequency: "daily",
      priority: 1.0,
    },
    {
      url: `${SITE_URL}/models`,
      lastModified,
      changeFrequency: "daily",
      priority: 0.9,
    },
    {
      url: `${SITE_URL}/runs`,
      lastModified,
      changeFrequency: "daily",
      priority: 0.85,
    },
    {
      url: `${SITE_URL}/llms.txt`,
      lastModified,
      changeFrequency: "daily",
      priority: 0.75,
    },
    {
      url: `${SITE_URL}/llms-full.txt`,
      lastModified,
      changeFrequency: "daily",
      priority: 0.7,
    },
    {
      url: `${SITE_URL}/hard`,
      lastModified: new Date("2026-11-23T00:00:00Z"),
      changeFrequency: "monthly",
      priority: 0.45,
    },
  ]

  const modelEntries: MetadataRoute.Sitemap = idx.models.map((m) => {
    const retired = RETIRED_MODEL_SLUGS.has(m.slug)
    return {
      url: `${SITE_URL}/models/${m.slug}`,
      lastModified,
      changeFrequency: retired ? "monthly" : "weekly",
      priority: retired ? 0.5 : 0.8,
    }
  })

  const runEntries: MetadataRoute.Sitemap = collectPublicRunEntries(idx).map((r) => ({
    url: `${SITE_URL}/runs/${r.gpu}/${r.runId}`,
    lastModified,
    changeFrequency: r.bench === "hard" ? "yearly" : "monthly",
    priority: r.bench === "hard" ? 0.35 : 0.65,
  }))

  return [...staticEntries, ...modelEntries, ...runEntries]
}
