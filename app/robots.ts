import type { MetadataRoute } from "next"

export default function robots(): MetadataRoute.Robots {
  return {
    rules: [
      {
        userAgent: "*",
        allow: "/",
      },
      {
        userAgent: [
          "Googlebot",
          "Google-Extended",
          "OAI-SearchBot",
          "GPTBot",
          "ClaudeBot",
          "Claude-SearchBot",
          "PerplexityBot",
          "Applebot",
        ],
        allow: "/",
      },
    ],
    sitemap: "https://kernelbench.com/sitemap.xml",
    host: "https://kernelbench.com",
  }
}
