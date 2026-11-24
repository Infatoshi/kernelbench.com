import { readFile, readdir } from "node:fs/promises"
import { join } from "node:path"
import { LIVE_MODEL_SLUGS, loadAllModelIndex } from "./models.server"
import type { AuditOutcome, Bench, ModelCell } from "./models"

// Opaque cwd indirection prevents Turbopack from globbing gitignored archives.
const REPO_ROOT = process.cwd().split("/").join("/")

const CANON = "rtxpro6000"
const GPU_ORDER = ["rtxpro6000", "h100"] as const
const ALL_GPUS = new Set(["rtxpro6000", "h100", "b200"])
const ACTIVE_BENCHES = new Set(["mega", "cuda"])

const LAB_LOGO: Record<string, string> = {
  Anthropic: "anthropic",
  OpenAI: "openai",
  Google: "google",
  xAI: "xai",
  DeepSeek: "deepseek",
  "Moonshot AI": "moonshot",
  "Z.ai": "zai",
  MiniMax: "minimax",
  Meituan: "meituan",
  NVIDIA: "nvidia",
  Alibaba: "alibabacloud",
  Cursor: "cursor",
  Xiaomi: "xiaomi",
  Meta: "meta",
}

const PROBLEM_LABELS: Record<string, string> = {
  "01_fp8_gemm": "FP8 GEMM",
  "02_kda_cutlass": "KDA CUTLASS",
  "03_paged_attention": "Paged Attention",
  "05_topk_bitonic": "TopK Bitonic",
  "06_sonic_moe_swiglu": "Sonic MoE",
  "07_w4a16_gemm": "W4A16 GEMM",
  "02_kimi_linear_decode": "Kimi-Linear Decode",
  "01_glm52_fused_moe": "GLM-5.3 Fused MoE",
  "02_deepseek_nsa": "DeepSeek NSA",
  "03_megaqwen_decode": "MegaQwen Decode",
  "01_rl_grid_ppo": "RL Grid PPO",
  "01_rmsnorm_residual": "RMSNorm Residual",
  "02_online_softmax": "Online Softmax",
  "09_fmha_preattn_mrope": "FMHA Pre-attn MRoPE",
  "10_patch_embed_conv3d_gemm": "Patch-Embed Conv3D GEMM",
}

const SHORT_LABELS: Record<string, string> = {
  "02_kimi_linear_decode": "Mega",
  "01_glm52_fused_moe": "Fused MoE",
  "02_deepseek_nsa": "NSA",
  "03_megaqwen_decode": "MegaQwen",
}

const FLAGGED = new Set([
  "reward_hack",
  "contamination",
  "contaminated",
  "rubric_leak",
  "suspect",
  "megakernel_not_authentic",
])

export interface SketchpadPayload {
  generated: string
  canon: string
  site: string
  gpus: { id: string; label: string }[]
  gpu_labels: Record<string, string>
  problems: Record<Bench, string[]>
  plabel: Record<string, string>
  pshort: Record<string, string>
  methodology: string
  models: Record<string, { n: string; l: string; lg?: string; st: "cur" | "ret" }>
  boards: Record<Bench, Record<string, BoardEntry[]>>
  runs: Record<string, RunEntry>
  detail: Record<string, DetailEntry>
  code: Record<string, string>
  log: Record<string, string[]>
  logos: Record<string, string>
  census: Record<string, number>
}

interface BoardEntry {
  m: string
  pa: number
  to: number
  pf: number | null
  h: string | null
  e: string | null
  c: Record<string, string>
}

interface RunEntry {
  b: Bench
  m: string
  p: string
  g?: string | null
  on?: number
  s?: number | null
  ms?: number | null
  v?: string | null
  ok?: number
  o?: string | null
  fw?: string | null
  t?: number | null
  tr?: string | null
  su?: string | null
  ctx?: Record<string, number>
  tok?: number | null
  sum?: string | null
  pg?: boolean
  oh?: string | null
  oe?: string | null
  og?: string | null
}

interface DetailEntry {
  regime?: string | null
  pt?: number | null
  pb?: number | null
  st: {
    agent_s?: number | null
    total_s?: number | null
    output_tokens?: number | null
    input_tokens?: number | null
    cost_usd?: number | null
  }
  sh: {
    l?: string
    tf?: number
    gb?: number
    ms?: number
    cu?: number
    mu?: number
    bd?: string
  }[]
}

async function readOptional(path: string): Promise<string | null> {
  try {
    return await readFile(path, "utf8")
  } catch {
    return null
  }
}

export async function loadSketchpadPayload(): Promise<SketchpadPayload> {
  const mi = await loadAllModelIndex()

  const models: SketchpadPayload["models"] = {}
  for (const m of mi.models) {
    models[m.slug] = {
      n: m.name,
      l: m.lab || "",
      lg: LAB_LOGO[m.lab],
      st: LIVE_MODEL_SLUGS.has(m.slug) ? "cur" : "ret",
    }
  }

  const runs: Record<string, RunEntry> = {}
  const boards: SketchpadPayload["boards"] = {
    mega: { rtxpro6000: [], h100: [] },
    cuda: { rtxpro6000: [], h100: [] },
    hard: { rtxpro6000: [], h100: [] },
  }

  const activeRids: { gpu: string; rid: string }[] = []
  const solUrls = new Map<string, string>()

  const addCell = (
    rid: string,
    bench: Bench,
    slug: string,
    pid: string,
    gpu: string,
    c: ModelCell,
  ) => {
    const r = (runs[rid] ??= { b: bench, m: slug, p: pid })
    Object.assign(r, {
      b: bench,
      m: slug,
      p: pid,
      g: gpu,
      on: 1,
      s: c.score ?? null,
      ms: c.latency_ms ?? null,
      v: c.verdict ?? null,
      ok: c.valid ? 1 : 0,
      o: c.outcome ?? null,
      fw: c.framework ?? null,
      t: c.elapsed_seconds ?? null,
      tr: c.trace_url ?? null,
      su: c.solution_url ?? null,
    })
    if (c.ctx) {
      r.ctx = c.ctx
      r.tok = c.tok_s ?? null
    }
    if (ACTIVE_BENCHES.has(bench)) {
      activeRids.push({ gpu, rid })
      if (c.solution_url) solUrls.set(rid, c.solution_url)
    }
  }

  const addOutcome = (bench: Bench, slug: string, o: AuditOutcome) => {
    const rid = o.run_id
    const r = (runs[rid] ??= {
      b: bench,
      m: slug,
      p: o.problem || "unknown",
    })
    r.sum = o.summary || null
    r.pg = o.publish_grade ?? undefined
    r.oh = o.harness || null
    r.oe = o.effort || null
    if (o.gpu && ALL_GPUS.has(o.gpu)) {
      r.og = o.gpu
    }
    if (!r.on) {
      r.v = o.verdict ?? null
      r.s = o.score ?? null
      r.ms = o.latency_ms ?? null
      r.tr = o.trace_url ?? null
      r.su = o.solution_url ?? null
      r.ok = 0
    }
  }

  for (const m of mi.models) {
    const slug = m.slug
    for (const bench of ["mega", "cuda", "hard"] as const) {
      const b = m.benches[bench]
      if (!b) continue
      const views: [string, typeof b | NonNullable<typeof b.gpus>[string]][] = [
        [CANON, b],
        ...Object.entries(b.gpus ?? {}),
      ]
      for (const [gpu, gb] of views) {
        const isShownGpu = (GPU_ORDER as readonly string[]).includes(gpu)
        const cells = gb.cells ?? {}
        const entry: BoardEntry = {
          m: slug,
          pa: gb.passed ?? 0,
          to: gb.total_problems ?? 0,
          pf: gb.perf ?? null,
          h: gb.harness ?? null,
          e: gb.effort || null,
          c: {},
        }
        for (const [pid, c] of Object.entries(cells)) {
          if (!c.run_id) continue
          entry.c[pid] = c.run_id
          addCell(c.run_id, bench, slug, pid, gpu, c)
        }
        if (isShownGpu && Object.keys(entry.c).length > 0) {
          boards[bench][gpu].push(entry)
        }
      }
      for (const o of b.outcomes ?? []) {
        addOutcome(bench, slug, o)
      }
    }
  }

  // Load per-shape rundetails and redacted kernel excerpts for active Mega/CUDA runs.
  const detail: Record<string, DetailEntry> = {}
  const detailGpu = new Map<string, string>()
  const seenDetail = new Set<string>()
  await Promise.all(
    activeRids.map(async ({ gpu, rid }) => {
      const key = `${gpu}/${rid}`
      if (seenDetail.has(key)) return
      seenDetail.add(key)
      const prefix = gpu === CANON ? "" : `${gpu}/`
      const raw = await readOptional(
        join(REPO_ROOT, "public/data/rundetail", `${prefix}${rid}.json`),
      )
      if (!raw) return
      try {
        const v = JSON.parse(raw)
        const st = v.stats ?? {}
        detail[key] = {
          regime: v.regime ?? null,
          pt: v.peak_tflops ?? null,
          pb: v.peak_bw_gb_s ?? null,
          st: {
            agent_s: st.agent_s ?? null,
            total_s: st.total_s ?? null,
            output_tokens: st.output_tokens ?? null,
            input_tokens: st.input_tokens ?? null,
            cost_usd: st.cost_usd ?? null,
          },
          sh: (v.shapes ?? []).map((s: Record<string, unknown>) => ({
            l: s.label as string | undefined,
            tf: s.tflops as number | undefined,
            gb: s.gbps as number | undefined,
            ms: s.ms as number | undefined,
            cu: s.compute_util as number | undefined,
            mu: s.mem_util as number | undefined,
            bd: s.bound as string | undefined,
          })),
        }
        detailGpu.set(rid, gpu)
      } catch {
        // skip malformed rundetail
      }
    }),
  )

  const code: Record<string, string> = {}
  await Promise.all(
    Array.from(solUrls.entries()).map(async ([rid, rel]) => {
      const raw = await readOptional(
        join(REPO_ROOT, "public", rel.replace(/^\//, "")),
      )
      if (!raw) return
      const lines = raw.split(/\r?\n/)
      let snippet = lines.slice(0, 140).join("\n")
      if (lines.length > 140) {
        snippet += `\n# ... (${lines.length - 140} more lines in full kernel file)`
      }
      code[rid] = snippet.slice(0, 7000)
    }),
  )

  // Resolve off-board attempt GPU only when explicitly recorded.
  for (const [rid, r] of Object.entries(runs)) {
    const og = r.og ?? null
    delete r.og
    if (!r.on) {
      const tr = r.tr || ""
      const mm = /\/blob\/main\/(h100|b200)\//.exec(tr)
      r.g = detailGpu.get(rid) || (mm ? mm[1] : null) || og
    }
  }

  // Group every run into its model's complete audit log (Mega -> CUDA -> Hard).
  const benchIdx: Record<Bench, number> = { mega: 0, cuda: 1, hard: 2 }
  const probIdx: Record<Bench, Record<string, number>> = {
    mega: {},
    cuda: {},
    hard: {},
  }
  for (const b of ["mega", "cuda", "hard"] as const) {
    ;(mi.benches[b]?.problems ?? []).forEach((p, i) => {
      probIdx[b][p] = i
    })
  }
  const log: Record<string, string[]> = {}
  for (const [rid, r] of Object.entries(runs)) {
    ;(log[r.m] ??= []).push(rid)
  }
  for (const ids of Object.values(log)) {
    ids.sort((a, b) => {
      const ra = runs[a]
      const rb = runs[b]
      const ba = benchIdx[ra.b] - benchIdx[rb.b]
      if (ba !== 0) return ba
      const pa =
        (probIdx[ra.b][ra.p] ?? 99) - (probIdx[rb.b][rb.p] ?? 99)
      if (pa !== 0) return pa
      if (ra.p !== rb.p) return ra.p.localeCompare(rb.p)
      return b.localeCompare(a)
    })
  }

  // Audit census across active Mega and CUDA sessions.
  const census: Record<string, number> = {
    clean: 0,
    interesting: 0,
    rejected: 0,
    failed: 0,
  }
  for (const r of Object.values(runs)) {
    if (!ACTIVE_BENCHES.has(r.b)) continue
    const v = r.v
    if (!v || v === "unaudited") continue
    const g =
      v === "clean"
        ? "clean"
        : v === "interesting"
          ? "interesting"
          : FLAGGED.has(v)
            ? "rejected"
            : "failed"
    census[g] = (census[g] || 0) + 1
  }

  // Load inline lab SVG strings.
  const logos: Record<string, string> = {}
  try {
    const logoDir = join(REPO_ROOT, "public/logos/labs")
    const entries = await readdir(logoDir)
    await Promise.all(
      entries
        .filter((f) => f.endsWith(".svg"))
        .map(async (f) => {
          const raw = await readOptional(join(logoDir, f))
          if (!raw) return
          const stem = f.slice(0, -4)
          logos[stem] = raw
            .trim()
            .replace(/<title>.*?<\/title>/g, "")
            .replaceAll('fill="#eeeeee"', 'fill="currentColor"')
            .replaceAll('role="img"', 'aria-hidden="true"')
        }),
    )
  } catch {
    // missing logos dir is non-fatal
  }

  const gpuLabels = mi.benches.cuda?.gpu_labels ?? {
    rtxpro6000: "RTX PRO 6000",
    h100: "H100 PCIe",
    b200: "B200",
  }

  return {
    generated: mi.generated,
    canon: CANON,
    site: "",
    gpus: GPU_ORDER.map((g) => ({ id: g, label: gpuLabels[g] ?? g })),
    gpu_labels: gpuLabels,
    problems: {
      mega: mi.benches.mega?.problems ?? ["02_kimi_linear_decode"],
      cuda: mi.benches.cuda?.problems ?? [
        "01_glm52_fused_moe",
        "02_deepseek_nsa",
        "03_megaqwen_decode",
      ],
      hard: mi.benches.hard?.problems ?? [],
    },
    plabel: PROBLEM_LABELS,
    pshort: SHORT_LABELS,
    methodology: mi.methodology,
    models,
    boards,
    runs,
    detail,
    code,
    log,
    logos,
    census,
  }
}
