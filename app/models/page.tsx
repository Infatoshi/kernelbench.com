import { columnOrder, columnsForBench, columnsForCorrectness } from "@/app/_lib/models"
import { loadModelIndex } from "@/app/_lib/models.server"
import { ModelScoreboards } from "@/app/_components/model-columns"

// All models, AA-style vertical column charts: one performance chart per
// published bench, Multi (coming soon), then the compiled correctness chart.

export const metadata = { title: "models · kernelbench" }

export default async function ModelsPage() {
  const idx = await loadModelIndex()
  const ordered = columnOrder(idx)
  const perfCharts = (["mega", "hard", "cuda"] as const).map((b) =>
    columnsForBench(idx, b, ordered),
  )
  const correctnessChart = columnsForCorrectness(idx, ordered)
  return (
    <div className="hard-page space-y-12">
      <section>
        <h1 className="text-3xl font-semibold tracking-tight text-[var(--color-fg-bright)] mb-3">
          models
        </h1>
        <p className="text-xs text-[var(--color-fg-muted)] max-w-4xl leading-relaxed">
          Frontier coding models on the kernel decks, AA-style. Performance is
          disaggregated per benchmark (Mega / Hard / CUDA, bars colored by
          lab), and the last chart is compiled correctness: the percentage of
          published problems each model gets correct across the benches it
          attempted. Each chart ranks models by its own score, best on the
          left. Click any column for per-problem cells, audit chips, and
          the model&apos;s full integrity record.
        </p>
      </section>
      <section>
        <ModelScoreboards perf={perfCharts} correctness={correctnessChart} />
      </section>
    </div>
  )
}
