import Link from "next/link"
import { loadModelIndex, rowsForBench } from "@/app/_lib/models"
import { ModelGpuBoard, type GpuView } from "@/app/_components/model-board"

// KernelBench-Hard: per-op kernel deck. Ranked model list per GPU board;
// run-level forensics live on /runs and the per-model pages.

export default async function HardPage() {
  const idx = await loadModelIndex()
  const gpuLabels = idx.benches.hard?.gpu_labels ?? {}

  const mk = (gpu?: string): { board: ReturnType<typeof rowsForBench>["board"]; sink: ReturnType<typeof rowsForBench>["sink"] } =>
    rowsForBench(idx, "hard", gpu)

  const views: GpuView[] = [
    {
      key: "rtxpro6000",
      label: gpuLabels.rtxpro6000 ?? "RTX PRO 6000",
      blurb:
        "Unlimited time per problem. The frozen lab board — every published cell is contamination-checked and reward-hack audited.",
      ...mk(),
    },
    {
      key: "h100",
      label: gpuLabels.h100 ?? "H100 PCIe",
      ...mk("h100"),
    },
    {
      key: "b200",
      label: gpuLabels.b200 ?? "B200",
      ...mk("b200"),
    },
    {
      key: "rtx3090",
      label: gpuLabels.rtx3090 ?? "RTX 3090",
      ...mk("rtx3090"),
    },
  ]

  return (
    <div className="hard-page space-y-12">
      <section>
        <h1 className="text-3xl font-semibold tracking-tight text-[var(--color-fg-bright)] mb-3">
          hard
        </h1>
        <p className="text-sm text-[var(--color-fg)] mb-2">
          One ranked model list per GPU board &mdash; use the toggle to switch.
          Peak fraction (SOL) is measured against each GPU&rsquo;s own roofline,
          so scores are comparable within a GPU, not across GPUs.
        </p>
        <p className="text-xs text-[var(--color-fg-muted)] mb-6 max-w-4xl leading-relaxed">
          Ranked by valid passes (audited-clean correct cells), then mean
          normalized performance (cell &divide; board best per problem). The
          flagged badge counts audited sessions that failed the reward-hack
          review; it never changes the rank. Click a model for its per-problem
          cells, audit chips, and integrity record. Hardware notes for each GPU
          are below the list.
        </p>
      </section>

      <section>
        <ModelGpuBoard views={views} />
        <p className="text-xs text-[var(--color-fg)] mt-3 max-w-4xl leading-relaxed">
          Browse the{" "}
          <Link href="/runs" className="underline underline-offset-2">
            run index
          </Link>
          {" "}for transcripts, submitted solutions, checks, timing, and costs.
        </p>
      </section>

      <section>
        <h2 className="text-sm font-semibold text-[var(--color-fg-bright)] mb-3">
          hardware notes
        </h2>
        <div className="space-y-3">
          <p className="text-xs text-[var(--color-fg-muted)] max-w-4xl leading-relaxed">
            <span className="text-[var(--color-fg)] font-semibold">RTX PRO 6000.</span>{" "}
            Frontier coding agents on one unlimited autonomous session per problem:{" "}
            Opus 4.8, Fable 5, Grok 4.5, GPT-5.6 Sol, GLM-5.2, MiniMax-M3, DeepSeek
            V4 Pro, LongCat 2.0, and more. Roofline-graded; every published cell is
            contamination-checked and reward-hack audited.
          </p>
          <p className="text-xs text-[var(--color-fg-muted)] max-w-4xl leading-relaxed">
            <span className="text-[var(--color-fg)] font-semibold">H100 PCIe.</span>{" "}
            Opus 4.8, Fable 5, GLM-5.2, MiniMax-M3, DeepSeek V4 Pro, and LongCat 2.0
            on a single H100 PCIe with the same containerized harness and roofline
            grading as the Blackwell deck; peak fraction is measured against H100
            dense peaks.
          </p>
          <p className="text-xs text-[var(--color-fg-muted)] max-w-4xl leading-relaxed">
            <span className="text-[var(--color-fg)] font-semibold">B200.</span>{" "}
            The same models on a single NVIDIA B200 (SM100 Blackwell, HBM3e)
            with the identical containerized harness and roofline grading; peak
            fraction is measured against B200 dense peaks (fp8 4500, bf16 2250
            TFLOPS), so the same kernel reads as a smaller fraction of a much higher
            ceiling.
          </p>
          <p className="text-xs text-[var(--color-fg-muted)] max-w-4xl leading-relaxed">
            <span className="text-[var(--color-fg)] font-semibold">RTX 3090.</span>{" "}
            The first consumer/Ampere part in the set: a single RTX 3090 (SM86,
            GDDR6X, 936 GB/s). Ampere has no FP8 tensor cores, so the FP8 GEMM
            problem is dropped entirely &mdash; five problems, not six. The
            memory-bound cells (paged attention, top-k, W4A16) port cleanly and lead
            the board, with paged-attention decode reaching ~66% of the 936 GB/s
            ceiling; the bf16 compute cells (KDA, sonic-MoE) run against Ampere&rsquo;s
            much lower bf16 peak. GLM-5.2 is a clean 5/5 and its paged-attention
            kernel holds the board&rsquo;s top bandwidth
            fraction (~66%). Only MiniMax-M3 is partial &mdash; its sweep was
            cut short by provider rate limits.
          </p>
        </div>
      </section>
    </div>
  )
}
