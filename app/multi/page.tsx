import Link from "next/link"
import { PageHead } from "@/app/_components/page-head"

// KernelBench-Multi: the multi-GPU (NVLink) sibling of the hard bench. Graded on
// busbw (achieved NVLink bus bandwidth / NVLink peak), never TFLOPS. The deck is
// six communication-dominated problems on an 8xH100 SXM (NVSwitch) node.
// No runs yet — this page stands up the coming-soon card + deck.

const REPO_TREE =
  "https://github.com/Infatoshi/kernelbench.com/blob/master/benchmarks/multi"

const MULTI_PROBLEMS = [
  { key: "01_allreduce_residual", label: "AllReduce + Residual" },
  { key: "02_reducescatter_rmsnorm", label: "ReduceScatter + RMSNorm" },
  { key: "03_allgather_fp8", label: "AllGather + fp8 Dequant" },
  { key: "04_moe_all2all", label: "MoE All-to-All" },
  { key: "05_ulysses_all2all", label: "Ulysses All-to-All" },
  { key: "06_fp8_reducescatter_grad", label: "fp8 ReduceScatter Grad" },
] as const

export default function MultiPage() {
  return (
    <div className="space-y-6">
      <PageHead
        kicker="Benchmark · Multi"
        title="The NVLink deck"
        sub={
          <>
            The multi-GPU sibling of <Link href="/hard">hard</Link>: agents
            rewrite PyTorch + NCCL collectives as fine-grained NVLink kernels on
            an 8×H100 SXM node (NVSwitch · NVLink4 · ~900 GB/s/GPU), graded on{" "}
            <strong>busbw</strong> — bus-bandwidth efficiency, never TFLOPS.{" "}
            <strong>No runs yet.</strong>
          </>
        }
        notes={
          <p>
            <strong>The metric.</strong> busbw = achieved NVLink bus bandwidth
            ÷ NVLink peak, so every problem is deliberately
            communication-dominated. Six-problem deck, all busbw-graded, each
            forbidding its bare collective (CUDA / Triton / NVSHMEM / CUDA
            symmetric memory / ParallelKittens, as long as it beats the NCCL
            baseline). Methodology and the full deck live in the{" "}
            <Link href={`${REPO_TREE}/SPEC.md`}>spec</Link>.
          </p>
        }
      />

      <div className="cell-grid">
        {MULTI_PROBLEMS.map((p) => (
          <div key={p.key} className="cell-card">
            <div className="cell-card-head">
              <span className="cell-card-problem">{p.label}</span>
              <span className="status-pill status-pill-muted">coming soon</span>
            </div>
            <div className="cell-card-links">
              <a
                className="link-chip"
                href={`${REPO_TREE}/problems-h100x8/${p.key}/reference.py`}
                target="_blank"
                rel="noreferrer"
              >
                reference
              </a>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
