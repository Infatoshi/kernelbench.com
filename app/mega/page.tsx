import Link from "next/link"
import { barsForBench } from "@/app/_lib/models"
import { loadModelIndex } from "@/app/_lib/models.server"
import { ModelGpuBoard, type GpuView } from "@/app/_components/model-board"

// KernelBench-Mega: whole-block megakernels. Ranked model list per GPU board.

const REFERENCE_URL = `/code?f=${encodeURIComponent(
  "/data/mega/code/02_kimi_linear_decode.reference.py.txt",
)}`
const BASELINE_URL = `/code?f=${encodeURIComponent(
  "/data/mega/code/02_kimi_linear_decode.baseline.py.txt",
)}`

export default async function MegaPage() {
  const idx = await loadModelIndex()
  const gpuLabels = idx.benches.mega?.gpu_labels ?? {}

  const mk = (gpu?: string): Pick<GpuView, "bars"> => ({
    bars: barsForBench(idx, "mega", gpu),
  })

  const views: GpuView[] = [
    {
      key: "rtxpro6000",
      label: gpuLabels.rtxpro6000 ?? "RTX PRO 6000 Blackwell",
      ...mk(),
    },
    { key: "h100", label: gpuLabels.h100 ?? "H100", ...mk("h100") },
    { key: "b200", label: gpuLabels.b200 ?? "B200", ...mk("b200") },
  ]

  return (
    <div className="hard-page space-y-12">
      <section>
        <h1 className="text-3xl font-semibold tracking-tight text-[var(--color-fg-bright)] mb-3">
          mega
        </h1>
        <p className="text-sm text-[var(--color-fg)] mb-2">
          KernelBench-Mega · whole-block megakernels
          <span className="ml-2 text-xs font-semibold text-[var(--color-accent)]">
            ● unlimited time · RTX PRO 6000 Blackwell + H100 + B200
          </span>
        </p>
        <p className="text-xs text-[var(--color-fg-muted)] mb-2 max-w-4xl leading-relaxed">
          KernelBench-Mega tests whole-block megakernels: instead of grading a
          single isolated op, the agent fuses an entire model block into one
          kernel. Problem <code>02_kimi_linear_decode</code> is a Kimi-Linear
          W4A16 hybrid decode (4-bit weights, bf16 activations). The headline
          metric is the{" "}
          <span className="text-[var(--color-fg)]">
            decode speedup over an optimized-PyTorch baseline
          </span>{" "}
          (e.g. <code>19.35x</code> = 19x faster than the reference), not a 0-1
          roofline fraction; <code>tok/s</code> is decode tokens per second.
          Higher is better for both, and results are reported per GPU. The
          transcript is the headline artifact: it shows the model&apos;s full
          optimization journey from baseline to the final megakernel.
        </p>
        <p className="text-xs text-[var(--color-fg-muted)] mb-6 max-w-4xl leading-relaxed">
          Each run gets a single autonomous session with unlimited wall-clock;
          models self-terminate well under three hours (the longest run so far
          is ~2.5h). One bar per model, colored by lab; models group by valid
          passes, then order by best decode speedup. The flagged badge counts
          audited sessions that failed the reward-hack or megakernel-authenticity
          review; it never changes the order.
        </p>
      </section>

      <section>
        <ModelGpuBoard views={views} />
        <p className="text-xs text-[var(--color-fg)] mt-3 max-w-4xl leading-relaxed">
          Click a model for the per-ctx breakdown (speedup at 2k / 8k / 16k
          decode context), audit chips, and its integrity record. The problem&apos;s{" "}
          <Link href={REFERENCE_URL} className="underline underline-offset-2">
            reference
          </Link>{" "}
          and{" "}
          <Link href={BASELINE_URL} className="underline underline-offset-2">
            baseline
          </Link>{" "}
          are self-hosted. Browse the{" "}
          <Link href="/runs" className="underline underline-offset-2">
            run index
          </Link>{" "}
          for transcripts and solutions, or the{" "}
          <Link
            href="https://github.com/Infatoshi/kernelbench.com/tree/master/benchmarks/mega"
            className="underline underline-offset-2"
          >
            mega benchmark source
          </Link>
          .
        </p>
      </section>
    </div>
  )
}
