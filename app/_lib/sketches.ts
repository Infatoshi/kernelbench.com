// Hand-sketched SVG doodles for the engineer's notepad look: CUDA grid,
// threadblock, warp lanes, cp.async -> smem tile -> mma.sync -> shfl.sync
// pipeline, roofline, and two margin notes. Pure deterministic vector strokes
// from a seeded PRNG (overshooting corners, doubled pen lines, slight bow).

function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function esc(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
}

class Pen {
  private r: () => number
  private o: string[] = []

  constructor(seed: number) {
    this.r = mulberry32(seed)
  }

  private j(a = 1.0): number {
    return (this.r() * 2 - 1) * a
  }

  line(
    x1: number,
    y1: number,
    x2: number,
    y2: number,
    cls = "g",
    twice = true,
    over = 2.2,
  ): void {
    const dx = x2 - x1
    const dy = y2 - y1
    const n = Math.hypot(dx, dy) || 1.0
    const ux = dx / n
    const uy = dy / n
    const passes = twice ? 2 : 1
    for (let p = 0; p < passes; p++) {
      const a = over * this.r()
      const b = over * this.r()
      const sx = x1 - ux * a + this.j(0.55)
      const sy = y1 - uy * a + this.j(0.55)
      const ex = x2 + ux * b + this.j(0.55)
      const ey = y2 + uy * b + this.j(0.55)
      const bow = this.j(Math.min(2.0, n * 0.018))
      const mx = (sx + ex) / 2 - uy * bow
      const my = (sy + ey) / 2 + ux * bow
      this.o.push(
        `<path class="${cls}" d="M${sx.toFixed(1)} ${sy.toFixed(1)}Q${mx.toFixed(1)} ${my.toFixed(1)} ${ex.toFixed(1)} ${ey.toFixed(1)}"/>`,
      )
    }
  }

  rect(x: number, y: number, w: number, h: number, cls = "g", twice = true): void {
    this.line(x, y, x + w, y, cls, twice)
    this.line(x + w, y, x + w, y + h, cls, twice)
    this.line(x + w, y + h, x, y + h, cls, twice)
    this.line(x, y + h, x, y, cls, twice)
  }

  hatch(x: number, y: number, w: number, h: number, cls = "h", gap = 6.0): void {
    let c = gap
    while (c < w + h - 1) {
      const x1 = c <= w ? x + c : x + w
      const y1 = c <= w ? y : y + c - w
      const x2 = c <= h ? x : x + c - h
      const y2 = c <= h ? y + c : y + h
      this.line(x1, y1, x2, y2, cls, false, 0)
      c += gap
    }
  }

  arrow(x1: number, y1: number, x2: number, y2: number, cls = "g"): void {
    this.line(x1, y1, x2, y2, cls, false, 0)
    const ang = Math.atan2(y2 - y1, x2 - x1)
    for (const s of [-1, 1]) {
      const a = ang + Math.PI + s * 0.5
      this.line(
        x2,
        y2,
        x2 + 7 * Math.cos(a),
        y2 + 7 * Math.sin(a),
        cls,
        false,
        0,
      )
    }
  }

  circle(cx: number, cy: number, r: number, cls = "g"): void {
    for (let i = 0; i < 2; i++) {
      this.o.push(
        `<ellipse class="${cls}" cx="${(cx + this.j(0.6)).toFixed(1)}" cy="${(cy + this.j(0.6)).toFixed(1)}" rx="${(r + this.j(0.7)).toFixed(1)}" ry="${(r + this.j(0.7)).toFixed(1)}" transform="rotate(${this.j(25).toFixed(0)} ${cx} ${cy})"/>`,
      )
    }
  }

  text(
    x: number,
    y: number,
    s: string,
    cls = "t",
    anchor = "start",
    rot: number | null = null,
  ): void {
    const angle = rot === null ? this.j(1.4) : rot
    this.o.push(
      `<text class="${cls}" x="${x}" y="${y}" text-anchor="${anchor}" transform="rotate(${angle.toFixed(1)} ${x} ${y})">${esc(s)}</text>`,
    )
  }

  svg(): string {
    return this.o.join("")
  }
}

function lane(
  p: Pen,
  x: number,
  y: number,
  label: string,
  hot = false,
  w = 246,
  n = 32,
): void {
  p.text(x - 50, y + 16, label, hot ? "tk" : "t")
  p.rect(x, y, w, 22, hot ? "k" : "g")
  for (let i = 0; i < n; i++) {
    const tx = x + 8 + (i * (w - 16)) / (n - 1)
    p.line(tx, y + 6, tx, y + 16, hot ? "kt" : "f", false, 0)
  }
}

function pipeline(p: Pen, x: number, y: number, title = true): void {
  if (title) p.text(x, y - 8, "per warp:", "t")
  const a = y + 38
  p.rect(x + 4, y + 18, 40, 40, "g")
  for (const k of [1, 2]) {
    p.line(x + 4 + k * 13.3, y + 19, x + 4 + k * 13.3, y + 57, "f", false, 0)
    p.line(x + 5, y + 18 + k * 13.3, x + 43, y + 18 + k * 13.3, "f", false, 0)
  }
  p.text(x + 24, y + 75, "gmem", "t", "middle")
  p.arrow(x + 51, a, x + 93, a)
  p.text(x + 72, y + 9, "cp.async", "t", "middle")
  p.rect(x + 100, y + 16, 44, 44, "k")
  p.hatch(x + 100, y + 16, 44, 44)
  p.text(x + 122, y + 77, "smem tile", "tk", "middle")
  p.arrow(x + 151, a, x + 193, a)
  p.text(x + 172, y + 9, "mma.sync", "t", "middle")
  p.rect(x + 200, y + 22, 32, 32, "g")
  p.line(x + 216, y + 23, x + 216, y + 53, "f", false, 0)
  p.line(x + 201, a, x + 231, a, "f", false, 0)
  p.text(x + 216, y + 71, "acc", "t", "middle")
  p.arrow(x + 239, a, x + 281, a)
  p.text(x + 260, y + 9, "shfl.sync", "t", "middle")
  p.circle(x + 298, a, 10, "k")
  p.text(x + 298, a + 5, "Σ", "tk", "middle", 0)
}

function drawHero(): string {
  const p = new Pen(11)
  p.text(18, 30, "grid · 4×3 blocks", "t")
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 4; c++) {
      const hot = r === 1 && c === 3
      const x = 18 + c * 38
      const y = 42 + r * 38
      p.rect(x, y, 30, 30, hot ? "k" : "g2", hot)
      if (hot) p.hatch(x, y, 30, 30, "h", 7)
    }
  }
  p.text(34, 172, "blockIdx (3,1)", "tk")
  p.arrow(128, 164, 144, 116, "kd")
  p.line(163, 80, 235, 44, "d", false, 0)
  p.line(163, 110, 235, 198, "d", false, 0)
  p.text(238, 30, "threadblock · 4 warps × 32 threads", "t")
  p.rect(236, 42, 312, 158, "g")
  p.text(296, 59, "tid 0", "t")
  p.text(540, 59, "tid 31", "t", "end")
  p.line(334, 55, 494, 55, "d", false, 0)
  for (let i = 0; i < 4; i++) {
    lane(p, 294, 64 + i * 33, `warp ${i}`, i === 0)
  }
  pipeline(p, 240, 234)
  p.line(30, 196, 30, 306, "g", false)
  p.line(30, 306, 198, 306, "g", false)
  p.line(32, 300, 104, 228, "g")
  p.line(104, 228, 192, 228, "g")
  p.text(118, 220, "roofline", "t")
  p.circle(150, 266, 3.2, "k")
  p.line(150, 261, 150, 232, "kd", false, 0)
  p.text(158, 264, "% of peak", "tk")
  p.text(38, 208, "perf", "t")
  p.text(114, 323, "flops / byte", "t", "middle")
  return p.svg()
}

function drawStrip(): string {
  const p = new Pen(5)
  p.text(60, 13, "tid 0", "t")
  p.text(320, 13, "tid 31", "t", "end")
  p.line(98, 9, 272, 9, "d", false, 0)
  lane(p, 58, 18, "warp 0", true, 264)
  pipeline(p, 10, 50, false)
  return p.svg()
}

function drawMega(): string {
  const p = new Pen(23)
  p.text(8, 18, "megakernel:", "t")
  const ops = ["embed", "attn", "mlp", "sample"]
  ops.forEach((name, i) => {
    const y = 38 + i * 46
    p.rect(54, y, 84, 28, "g")
    p.text(96, y + 19, name, "t", "middle")
    if (i < 3) p.arrow(96, y + 31, 96, y + 44, "g")
  })
  p.rect(38, 28, 116, 196, "kd", false)
  p.text(48, 244, "= 1 launch", "tk")
  return p.svg()
}

function drawCuda(): string {
  const p = new Pen(41)
  p.text(8, 18, "double buffer:", "t")
  p.text(8, 42, "cp.async", "t")
  for (let k = 0; k < 4; k++) {
    p.rect(8 + k * 46, 50, 38, 20, "g")
  }
  for (let k = 0; k < 3; k++) {
    const x = 54 + k * 46
    p.rect(x, 106, 38, 20, "k")
    p.hatch(x, 106, 38, 20, "h", 7)
    p.arrow(27 + k * 46, 73, x + 14, 103, "d")
  }
  p.text(54, 144, "mma.sync", "tk")
  p.text(8, 170, "load k+1 while", "t")
  p.text(8, 188, "k computes", "t")
  return p.svg()
}

export interface SketchSet {
  hero: string
  strip: string
  mega: string
  cuda: string
}

export function buildSketches(): SketchSet {
  return {
    hero: drawHero(),
    strip: drawStrip(),
    mega: drawMega(),
    cuda: drawCuda(),
  }
}
