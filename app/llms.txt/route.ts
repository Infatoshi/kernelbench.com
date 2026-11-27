import { renderLlmsTxt } from "@/app/_lib/seo.server"

export const dynamic = "force-static"

export async function GET() {
  const body = await renderLlmsTxt(false)
  return new Response(body, {
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
      "Cache-Control": "public, max-age=3600, s-maxage=86400",
    },
  })
}
