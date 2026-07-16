import { loadModelIndex, rowsForIndex } from "@/app/_lib/models"
import { ModelBoard } from "@/app/_components/model-list"

// All models, one row each, ranked across every published board. Same ranking
// as the homepage Models section, with the full sink section visible.

export const metadata = { title: "models · kernelbench" }

export default async function ModelsPage() {
  const idx = await loadModelIndex()
  const { board, sink } = rowsForIndex(idx)
  return (
    <div className="hard-page space-y-12">
      <section>
        <h1 className="text-3xl font-semibold tracking-tight text-[var(--color-fg-bright)] mb-3">
          models
        </h1>
        <p className="text-xs text-[var(--color-fg-muted)] max-w-4xl leading-relaxed">
          Frontier coding models, one row each, across all published boards
          (Mega / Hard / CUDA). Ranked by benchmarks fully passed, then mean
          normalized performance (cell &divide; board best per problem). Accuracy
          first: a cell only counts when it compiles, passes correctness, and
          survives the reward-hack audit. The flagged badge shows audited
          sessions that failed that review (over all audited sessions for the
          model); it never changes the rank. Models with no valid published
          results sit below the board. Click any model for per-problem cells,
          audit chips, and its full integrity record.
        </p>
      </section>
      <section>
        <ModelBoard board={board} sink={sink} showBadges />
      </section>
    </div>
  )
}
