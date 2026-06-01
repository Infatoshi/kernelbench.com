import Link from "next/link"

export default function BlogIndex() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="prompt cursor text-3xl font-bold text-[var(--color-fg-bright)] glow mb-3">
          writeups
        </h1>
        <p className="text-sm text-[var(--color-fg-muted)] mb-6">
          long-form posts on the design choices, the rubric leaks, the methodology behind each version.
        </p>
      </section>

      <section className="space-y-4">
        <Link
          href="/blog/hard"
          className="block box p-6 no-underline hover:border-[var(--color-fg-bright)] transition-colors"
        >
          <div className="text-xs text-[var(--color-accent)] mb-2">[ latest ]</div>
          <h2 className="text-xl font-bold text-[var(--color-fg-bright)] mb-2">
            Hard: Eight Scored Problems, Frontier Models, Two Rubric Leaks
          </h2>
          <p className="text-sm text-[var(--color-fg)] leading-relaxed">
            A focused successor to the v3 suite. One Blackwell GPU, eight scored CUDA problems, real coding-agent CLIs as the harness. Frontier models swept; only GPT-5.5 xhigh solved every problem. Two problems leak the rubric: five models all took the same bf16 shortcut on FP8 GEMM, and the only model that implemented Kahan compensated summation scored lowest of the passes.
          </p>
        </Link>

        <Link
          href="/blog/v3"
          className="block box p-6 no-underline hover:border-[var(--color-fg-bright)] transition-colors"
        >
          <div className="text-xs text-[var(--color-fg-muted)] mb-2">[ archive ]</div>
          <h2 className="text-xl font-bold text-[var(--color-fg-bright)] mb-2">
            v3: Rebuilding a GPU Kernel Benchmark from First Principles
          </h2>
          <p className="text-sm text-[var(--color-fg)] leading-relaxed">
            How discovering the original KernelBench was exploitable led to building a focused, cost-effective benchmark for evaluating LLM kernel engineering on modern architectures.
          </p>
        </Link>
      </section>
    </div>
  )
}
