"use client"

import { useEffect, useRef } from "react"
import type { SketchpadPayload } from "@/app/_lib/sketchpad.server"
import type { SketchSet } from "@/app/_lib/sketches"
import "@/app/sketchpad.css"

export function KernelBenchBoard({
  data,
  sketches,
}: {
  data: SketchpadPayload
  sketches: SketchSet
}) {
  const rootRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const root = rootRef.current
    if (!root) return

    const D = data
    const CANON = D.canon
    const GPU_IDS = new Set(D.gpus.map((g) => g.id))
    const GPU: Record<string, string> = Object.assign(
      {},
      D.gpu_labels || {},
      Object.fromEntries(D.gpus.map((g) => [g.id, g.label])),
    )
    const MEGA = "02_kimi_linear_decode"
    const NSA = "02_deepseek_nsa"
    const PCOLOR: Record<string, string> = {
      "02_kimi_linear_decode": "var(--green)",
      "01_glm52_fused_moe": "var(--green)",
      "02_deepseek_nsa": "var(--blue)",
      "03_megaqwen_decode": "var(--violet)",
    }
    const FLAG = new Set([
      "reward_hack",
      "contamination",
      "contaminated",
      "rubric_leak",
      "suspect",
      "megakernel_not_authentic",
    ])
    const VLABEL: Record<string, string> = {
      clean: "Clean",
      interesting: "Interesting",
      reward_hack: "Reward hack",
      contamination: "Contamination",
      contaminated: "Contamination",
      rubric_leak: "Rubric leak",
      suspect: "Suspect",
      megakernel_not_authentic: "Not a real megakernel",
      bug: "Bug",
      fail: "Failed",
      unaudited: "Not audited",
      genuine_check_failure: "Check failure",
      provider_failure_no_solution: "Provider failure",
      timeout_unverified: "Timed out, unverified",
      harness_limited: "Harness limited",
    }
    const OLABEL: Record<string, string> = {
      wrong: "wrong output",
      build: "build failed",
      timeout: "timed out",
      check_timeout: "check timed out",
      empty: "no solution",
      credits: "provider failure",
      incomplete: "incomplete",
      ungraded: "ungraded",
      other: "failed",
    }
    const FWLABEL: Record<string, string> = {
      ptx: "CUDA + inline PTX",
      cuda_raw: "Raw CUDA",
      cuda: "CUDA",
      cuda_wmma: "CUDA WMMA",
      triton: "Triton",
      eager: "Eager PyTorch",
    }
    const BLABEL: Record<string, string> = {
      mega: "Mega",
      cuda: "CUDA",
      hard: "Hard archive, frozen",
    }
    const GLABEL: Record<string, string> = {
      ok: "clean",
      note: "interesting",
      bad: "rejected",
      dead: "failed",
      un: "not audited",
    }
    const MONTHS = [
      "Jan", "Feb", "Mar", "Apr", "May", "Jun",
      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    const $ = <T extends HTMLElement = HTMLElement>(s: string, r: ParentNode = root) =>
      r.querySelector<T>(s)
    const $$ = <T extends HTMLElement = HTMLElement>(s: string, r: ParentNode = root) =>
      Array.from(r.querySelectorAll<T>(s))
    const esc = (s: unknown) =>
      String(s == null ? "" : s).replace(
        /[&<>"']/g,
        (c) =>
          ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
            c
          ] || c,
      )

    const state = {
      gpu: CANON,
      tab: "all",
      open: null as string | null,
      openFocus: "audit" as "audit" | "kernel",
      fold: false,
      hard: false,
      older: false,
      nav: [] as string[],
      navIdx: -1,
      full: false,
    }
    const popState = {
      rid: null as string | null,
      rids: [] as string[],
      anchor: null as HTMLElement | null,
      anchorX: null as number | null,
      hideTimer: 0 as ReturnType<typeof setTimeout> | number,
      switchTimer: 0 as ReturnType<typeof setTimeout> | number,
      overPop: false,
      lastMouse: null as { x: number; y: number } | null,
    }

    const fx = (v: number) => (v >= 10 ? v.toFixed(1) : v.toFixed(2)) + "×"
    const fpct = (v: number) => (v * 100).toFixed(2) + "%"
    const fms = (v: number) => (v < 1 ? v.toFixed(3) : v.toFixed(2)) + " ms"
    const fint = (v: number) => Math.round(v).toLocaleString("en-US")
    const fdur = (s: number) => {
      const h = Math.floor(s / 3600)
      const m = Math.round((s % 3600) / 60)
      return h ? `${h}h ${m}m` : `${m}m`
    }
    const ftok = (v: number) =>
      v >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : v >= 1e3 ? `${Math.round(v / 1e3)}k` : String(v)
    const fdate = (rid: string) => {
      const m = /^(\d{4})(\d{2})(\d{2})_/.exec(rid)
      return m ? `${+m[3]} ${MONTHS[+m[2] - 1]} ${m[1]}` : ""
    }
    const plural = (n: number, w: string) => `${n} ${w}${n === 1 ? "" : "s"}`
    const niceStep = (x: number) => {
      const p = Math.pow(10, Math.floor(Math.log10(x)))
      const f = x / p
      return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * p
    }
    const scale = (top: number) => {
      if (!(top > 0)) return { max: 1, step: 0.25 }
      const step = niceStep(top / 4)
      return { max: Math.ceil(top / step - 1e-9) * step, step }
    }

    const vgroup = (v: string | null | undefined) =>
      v === "clean"
        ? "ok"
        : v === "interesting"
          ? "note"
          : v && FLAG.has(v)
            ? "bad"
            : v == null || v === "unaudited"
              ? "un"
              : "dead"
    const vclass = (v: string | null | undefined) => {
      const g = vgroup(v)
      return g === "un" ? "dead" : g
    }
    const vicon = (c: string) =>
      ({ ok: "✓", note: "◆", bad: "✕", dead: "○" })[c] || "○"

    type RunRec = SketchpadPayload["runs"][string]
    const inDeck = (r: RunRec) => D.problems[r.b].includes(r.p)
    const kindOf = (r: RunRec) =>
      !inDeck(r) ? "raw" : r.b === "mega" ? "x" : r.p === NSA && r.ms != null ? "ms" : "pct"
    const valText = (r: RunRec) => {
      const k = kindOf(r)
      if (k === "ms" && r.ms != null) return fms(r.ms)
      if (r.s == null) return ""
      return k === "x" ? fx(r.s) : k === "raw" ? r.s.toFixed(3) : fpct(r.s)
    }
    const cellState = (r: RunRec | undefined) => {
      if (!r) return { k: "none", label: "not run", note: false }
      if (r.ok) return { k: "ok", label: "valid", note: r.v === "interesting" }
      if ((r.v && FLAG.has(r.v)) || (r.o && FLAG.has(r.o))) {
        return { k: "flag", label: (VLABEL[r.v || ""] || r.v || "rejected").toLowerCase(), note: false }
      }
      if (r.o === "pass" && r.v === "unaudited") return { k: "fail", label: "not audited", note: false }
      if (r.o === "pass" && r.v === "bug") return { k: "fail", label: "failed audit", note: false }
      if (r.o === "pass") return { k: "fail", label: "not publish-grade", note: false }
      return { k: "fail", label: OLABEL[r.o || ""] || r.o || "failed", note: false }
    }
    const plabel = (p: string) => D.plabel[p] || p
    const logoHTML = (m: SketchpadPayload["models"][string]) =>
      m.lg && D.logos[m.lg]
        ? `<span class="lg">${D.logos[m.lg]}</span>`
        : `<span class="lg mono">${esc((m.l || m.n).charAt(0))}</span>`
    const board = (b: "mega" | "cuda" | "hard", g: string) =>
      (D.boards[b] && D.boards[b][g]) || []
    const canon = () => state.gpu === CANON
    const visible = (bench: "mega" | "cuda" | "hard") => {
      const ents = board(bench, state.gpu)
      if (!canon()) return ents
      return ents.filter((e) => D.models[e.m]?.st === "cur")
    }
    const hasMsCache: Record<string, boolean> = {}
    const hasMs = (gpu: string) => {
      if (!(gpu in hasMsCache)) {
        hasMsCache[gpu] = board("cuda", gpu).some((e) => {
          const r = D.runs[e.c[NSA]]
          return Boolean(r && r.ms != null)
        })
      }
      return hasMsCache[gpu]
    }

    const ON_BOARD = new Set<string>()
    ;(["mega", "cuda", "hard"] as const).forEach((b) =>
      Object.values(D.boards[b]).forEach((arr) =>
        arr.forEach((e) => ON_BOARD.add(e.m)),
      ),
    )
    const ORPHANS = Object.keys(D.log)
      .filter((s) => !ON_BOARD.has(s))
      .sort()

    const pillCache: Record<string, string> = {}
    const pillHTML = (slug: string) => {
      if (slug in pillCache) return pillCache[slug]
      const rj = (D.log[slug] || [])
        .filter((id) => D.runs[id]?.v && FLAG.has(D.runs[id].v!))
        .sort()
        .reverse()
      let h = ""
      if (rj.length) {
        const by: Record<string, number> = {}
        rj.forEach((id) => {
          const v = D.runs[id].v || ""
          const k = (VLABEL[v] || v).toLowerCase()
          by[k] = (by[k] || 0) + 1
        })
        const tipText = `${plural(rj.length, "run")} rejected by audit: ${Object.keys(by).map((k) => `${by[k]} ${k}`).join(", ")}. Open the audit log.`
        h = `<button class="pill" data-open="${esc(rj[0])}" data-tip="${esc(tipText)}" aria-label="${esc(tipText)}">✕ ${rj.length} rejected</button>`
      }
      return (pillCache[slug] = h)
    }

    const axisHTML = (sc: { max: number; step: number }, fmt: (v: number) => string) => {
      const n = Math.round(sc.max / sc.step)
      let h = ""
      for (let i = 1; i <= n; i++) {
        h += `<span style="left:${((i / n) * 100).toFixed(3)}%">${fmt(i * sc.step)}</span>`
      }
      return `<div class="axis" aria-hidden="true">${h}</div>`
    }
    const trackHTML = (
      segs: { w: number; c: string; open?: string }[],
      sc: { max: number; step: number },
      ref: number | null,
    ) => {
      const s = segs
        .filter((x) => x.w > 0)
        .map(
          (x) =>
            `<i class="seg" style="width:${Math.min(100, x.w).toFixed(3)}%;background:${x.c}"${x.open ? ` data-pop="${esc(x.open)}"` : ""}></i>`,
        )
        .join("")
      const r =
        ref != null
          ? `<i class="ref" style="left:${((ref / sc.max) * 100).toFixed(3)}%"></i>`
          : ""
      return `<span class="track" style="background-size:${((sc.step / sc.max) * 100).toFixed(4)}% 100%"><span class="segs">${s}</span>${r}</span>`
    }
    const rowHTML = (o: {
      slug: string
      rank?: number
      open?: string | null
      rids?: string[]
      mark?: string
      why?: string
      val: string
      valCls?: string
      track: string
    }) => {
      const m = D.models[o.slug]
      const ridsAttr =
        o.rids && o.rids.length ? ` data-rids="${esc(o.rids.join(","))}"` : ""
      return `<div class="row"${o.open ? ` role="button" tabindex="0" data-pop="${esc(o.open)}"${ridsAttr}` : ""}><span class="rk">${o.rank || ""}</span>${logoHTML(m)}<span class="nm"><span class="nmt">${esc(m.n)}</span>${o.mark || ""}${pillHTML(o.slug)}${o.why || ""}</span><span class="val${o.valCls ? ` ${o.valCls}` : ""}">${o.val}</span>${o.track}</div>`
    }

    const NOTE_MARK =
      '<span class="mk note" data-tip="Audit verdict: interesting. A valid score with a technique worth reading.">◆</span>'
    const LOG_HINT =
      '<span><i class="pk">✕</i>rejected audits in that model’s log</span><span>Hover or tap any row for audit, kernel &amp; trace</span>'
    const emptyHTML = () =>
      `<p class="empty">No valid runs graded on ${esc(GPU[state.gpu])} yet.</p>`

    const singleChart = (
      bench: "mega" | "cuda",
      pid: string,
      o: {
        mul: number
        ref: number | null
        tick: (v: number) => string
        cap: string
        refLabel?: string
      },
    ) => {
      const all = board(bench, state.gpu)
      const top =
        Math.max(
          0,
          ...all.map((e) => {
            const r = D.runs[e.c[pid]]
            return r && r.ok && r.s ? r.s : 0
          }),
        ) * o.mul
      const sc = scale(top)
      const items = visible(bench).map((e) => {
        const r = D.runs[e.c[pid]]
        return { e, r, st: cellState(r) }
      })
      const list = items
        .filter((x) => x.st.k === "ok")
        .sort((a, b) => (b.r?.s || 0) - (a.r?.s || 0))
      if (!list.length) return emptyHTML()
      let h = `<div class="cap">${o.cap}</div>${axisHTML(sc, o.tick)}<div class="rows">`
      let rank = 0
      list.forEach((x) => {
        rank++
        h += rowHTML({
          slug: x.e.m,
          rank,
          open: x.e.c[pid],
          mark: x.st.note ? NOTE_MARK : "",
          val: valText(x.r),
          track: trackHTML(
            [{ w: (((x.r.s || 0) * o.mul) / sc.max) * 100, c: PCOLOR[pid] }],
            sc,
            o.ref,
          ),
        })
      })
      h += `</div><div class="legend">${o.refLabel ? `<span><i class="refk"></i>${o.refLabel}</span>` : ""}<span><i class="note">◆</i>audit note: unusual but legitimate technique</span>${LOG_HINT}</div>`
      return h
    }

    const renderMega = () => {
      const el = $("#megaChart")
      if (!el) return
      el.innerHTML = singleChart("mega", MEGA, {
        mul: 1,
        ref: 1,
        tick: (v) => `${v}×`,
        cap: `Speedup over optimized eager PyTorch on ${esc(GPU[state.gpu])}, geomean over three context lengths`,
        refLabel: "1× = eager PyTorch baseline",
      })
    }

    const renderCuda = () => {
      const probs = D.problems.cuda
      const tabsEl = $("#cudaTabs")
      if (tabsEl) {
        tabsEl.innerHTML =
          `<button role="tab" class="${state.tab === "all" ? "on" : ""}" data-tab="all">All three</button>` +
          probs
            .map(
              (p) =>
                `<button role="tab" class="${state.tab === p ? "on" : ""}" data-tab="${p}"><i class="sw" style="background:${PCOLOR[p]}"></i>${esc(plabel(p))}</button>`,
            )
            .join("")
      }
      const el = $("#cudaChart")
      if (!el) return
      if (state.tab !== "all") {
        const p = state.tab
        const ms = p === NSA && hasMs(state.gpu)
        el.innerHTML = singleChart(
          "cuda",
          p,
          ms
            ? {
                mul: 1,
                ref: 1,
                tick: (v) => `${+v.toFixed(2)}×`,
                cap: "Geomean latency, lower is better. Bar length is speedup over the frozen eager PyTorch anchor.",
                refLabel: "1× = eager PyTorch anchor",
              }
            : {
                mul: 100,
                ref: null,
                tick: (v) => `${+v.toFixed(2)}%`,
                cap: `Roofline score: share of the hardware peak on ${esc(GPU[state.gpu])}`,
              },
        )
        return
      }
      const all = board("cuda", state.gpu)
      const best: Record<string, number> = {}
      probs.forEach((p) => {
        best[p] = Math.max(
          0,
          ...all.map((e) => {
            const r = D.runs[e.c[p]]
            return r && r.ok && r.s ? r.s : 0
          }),
        )
      })
      const list = visible("cuda")
        .filter((e) => e.pa > 0)
        .slice()
        .sort((a, b) => b.pa - a.pa || (b.pf || 0) - (a.pf || 0))
      if (!list.length) {
        el.innerHTML = emptyHTML()
        return
      }
      const sc = { max: 100, step: 25 }
      let h = `<div class="cap">Deck score: each kernel scored against the board best, averaged over the three problems</div>${axisHTML(sc, (v) => String(v))}<div class="rows">`
      let lastPa: number | null = null
      let rank = 0
      list.forEach((e) => {
        if (lastPa !== null && e.pa !== lastPa) {
          h += `<div class="grp">${e.pa} of ${e.to} kernels valid</div>`
        }
        lastPa = e.pa
        rank++
        const segs: { w: number; c: string; open?: string }[] = []
        const why: string[] = []
        const rids: string[] = []
        let open: string | null = null
        probs.forEach((p) => {
          const r = D.runs[e.c[p]]
          const st = cellState(r)
          if (st.k === "ok" && r) {
            if (!open) open = e.c[p]
            rids.push(e.c[p])
            segs.push({
              w: ((r.s || 0) / best[p] / 3) * 100,
              c: PCOLOR[p],
              open: e.c[p],
            })
          } else {
            why.push(
              `${esc(D.pshort[p])} ${st.k === "flag" ? `<b>${esc(st.label)}</b>` : esc(st.label)}`,
            )
          }
        })
        if (!open) open = e.c[probs.find((p) => e.c[p]) || ""] || null
        h += rowHTML({
          slug: e.m,
          rank,
          open,
          rids,
          why: why.length ? `<span class="why">${why.join(" · ")}</span>` : "",
          val: e.pf != null ? (e.pf * 100).toFixed(1) : "–",
          track: trackHTML(segs, sc, null),
        })
      })
      h += `</div><div class="legend">${probs.map((p) => `<span><i class="sw" style="background:${PCOLOR[p]}"></i>${esc(plabel(p))}</span>`).join("")}${LOG_HINT}</div>`
      el.innerHTML = h
    }

    const renderControls = () => {
      const seg = $("#gpuSeg")
      if (seg) {
        seg.innerHTML = D.gpus
          .map(
            (g) =>
              `<button class="${g.id === state.gpu ? "on" : ""}" data-gpu="${g.id}" aria-pressed="${g.id === state.gpu}">${esc(g.label)}</button>`,
          )
          .join("")
      }
      const note = $("#boardNote")
      if (!note) return
      if (canon()) {
        const megaN = visible("mega").filter(
          (e) => cellState(D.runs[e.c[MEGA]]).k === "ok",
        ).length
        const cudaN = visible("cuda").filter((e) => e.pa > 0).length
        note.textContent = `${megaN} models on Mega · ${cudaN} on CUDA · valid audited runs only`
      } else {
        const ents = board("mega", state.gpu).concat(board("cuda", state.gpu))
        let last = ""
        ents.forEach((e) =>
          Object.values(e.c).forEach((rid) => {
            if (rid > last) last = rid
          }),
        )
        const n = new Set(ents.filter((e) => e.pa > 0).map((e) => e.m)).size
        note.textContent = `Archive board · ${n} models · last graded ${fdate(last)}`
      }
    }

    const renderOlder = () => {
      const box = $("#olderBox")
      if (!box) return
      if (!state.older) {
        box.innerHTML = ""
        return
      }
      const rows = Object.keys(D.models)
        .filter((s) => D.models[s].st !== "cur" && (D.log[s] || []).length > 0)
        .map((s) => {
          const m = D.models[s]
          const ids = D.log[s] || []
          const me = board("mega", CANON).find((e) => e.m === s)
          const mr = me && D.runs[me.c[MEGA]]
          const ce = board("cuda", CANON).find((e) => e.m === s)
          const bestRid =
            (mr && mr.ok && me?.c[MEGA]) ||
            (ce &&
              Object.values(ce.c).find((id) => D.runs[id] && D.runs[id].ok)) ||
            ids[0]
          return {
            s,
            m,
            ids,
            mr: mr && mr.ok ? mr : null,
            ce: ce && ce.pa > 0 ? ce : null,
            bestRid,
          }
        })
        .sort(
          (a, b) =>
            (b.mr?.s || 0) - (a.mr?.s || 0) ||
            (b.ce?.pf || 0) - (a.ce?.pf || 0) ||
            a.m.n.localeCompare(b.m.n),
        )
      let h =
        '<p class="note" style="margin:0 0 10px;max-width:74ch">Superseded siblings and retired models kept off the main board. Click any model to open its full audit log.</p>'
      h +=
        '<div class="hard"><table><thead><tr><th>Model</th><th>Lab</th><th>Mega</th><th>CUDA</th><th>Runs</th></tr></thead><tbody data-list>'
      rows.forEach((x) => {
        h += `<tr><td><button class="hm" data-open="${esc(x.bestRid)}">${logoHTML(x.m)}${esc(x.m.n)}</button></td><td class="un">${esc(x.m.l)}</td><td class="num">${x.mr && x.mr.s != null ? fx(x.mr.s) : '<span class="dd">–</span>'}</td><td class="num">${x.ce && x.ce.pf != null ? `${(x.ce.pf * 100).toFixed(1)} (${x.ce.pa}/${x.ce.to})` : '<span class="dd">–</span>'}</td><td class="num un">${x.ids.length}</td></tr>`
      })
      h += "</tbody></table></div>"
      box.innerHTML = h
    }

    const renderMethod = () => {
      const c = D.census
      const order: [string, string, string, string][] = [
        ["clean", "ok", "Clean", "clean"],
        ["interesting", "note", "Interesting: valid, unusual technique", "interesting"],
        ["rejected", "bad", "Rejected: hacks, leaks, contamination", "rejected"],
        ["failed", "dead", "Broken: bugs, build and provider failures", "broken"],
      ]
      const total = order.reduce((a, o) => a + (c[o[0]] || 0), 0)
      const col: Record<string, string> = {
        ok: "var(--ok)",
        note: "var(--warn)",
        bad: "var(--bad)",
        dead: "var(--dead)",
      }
      const line = $("#censusLine")
      if (line) {
        line.innerHTML =
          `<span class="lead"><b>${fint(total)}</b><span class="wide">Mega and CUDA </span>sessions audited. Hover any <span class="wide">model </span>row for audit, kernel &amp; trace.</span>` +
          order
            .map(
              (o) =>
                `<span><i class="ic ${o[1]}">${vicon(o[1])}</i> ${c[o[0]] || 0} ${o[3]}</span>`,
            )
            .join("")
      }
      const cbox = $("#censusBox")
      if (cbox && total > 0) {
        cbox.innerHTML =
          `<div class="census">${order.map((o) => `<i style="width:${(((c[o[0]] || 0) / total) * 100).toFixed(3)}%;background:${col[o[1]]}" data-tip="${esc(o[2].split(":")[0])}: ${c[o[0]] || 0} sessions"></i>`).join("")}</div>` +
          `<div class="cleg">${order.map((o) => `<span><i class="ic ${o[1]}">${vicon(o[1])}</i><b>${c[o[0]] || 0}</b>${esc(o[2])}</span>`).join("")}</div>`
      }
      const rt = $("#ruleText")
      if (rt) rt.textContent = D.methodology
    }

    const renderHard = () => {
      const box = $("#hardBox")
      if (!box) return
      if (!state.hard) {
        box.innerHTML = ""
        return
      }
      const probs = D.problems.hard
      const every = board("hard", state.gpu)
        .slice()
        .sort((a, b) => b.pa - a.pa || (b.pf || 0) - (a.pf || 0))
      const list = every.filter((e) => e.pa > 0)
      const none = every.length - list.length
      let h = `<p class="note" style="margin:0 0 10px;max-width:74ch">Six per-op kernels on the ${esc(GPU[state.gpu])} board, share of hardware peak in %. ${list.length} models with a valid kernel${none ? `, ${none} more had none` : ""}. Every cell opens that model’s audit log.</p>`
      h += `<div class="hard"><table><thead><tr><th>Model</th><th>Valid</th>${probs.map((p) => `<th>${esc(plabel(p))}</th>`).join("")}</tr></thead><tbody data-list>`
      list.forEach((e) => {
        const m = D.models[e.m]
        h += `<tr><td><span class="hm">${logoHTML(m)}${esc(m.n)}</span></td><td class="num">${e.pa}/${e.to}</td>${probs
          .map((p) => {
            const rid = e.c[p]
            const r = D.runs[rid]
            const st = cellState(r)
            if (!r) return '<td><span class="hc dd">·</span></td>'
            const txt =
              st.k === "ok" && r.s != null
                ? (r.s * 100).toFixed(1)
                : st.k === "flag"
                  ? "✕"
                  : r.o === "pass" && r.s != null
                    ? (r.s * 100).toFixed(1)
                    : "–"
            const cls =
              st.k === "ok"
                ? ""
                : st.k === "flag"
                  ? "bad"
                  : r.o === "pass" && r.s != null
                    ? "un"
                    : "dd"
            return `<td><button class="${cls}" data-open="${esc(rid)}">${txt}</button></td>`
          })
          .join("")}</tr>`
      })
      h += "</tbody></table></div>"
      const link = (slug: string, rid: string) =>
        `<button data-open="${esc(rid)}">${esc(D.models[slug].n)}</button>`
      const rest = every.filter((e) => !(e.pa > 0))
      if (rest.length) {
        h += `<p class="note hmore">No valid kernel on this board: ${rest.map((e) => link(e.m, Object.values(e.c).sort().pop() || "")).join(", ")}.</p>`
      }
      if (ORPHANS.length) {
        h += `<p class="note hmore">Audited, never on a board: ${ORPHANS.map((s) => link(s, D.log[s][0])).join(", ")}.</p>`
      }
      box.innerHTML = h
    }

    const shapesHTML = (
      r: RunRec,
      det: SketchpadPayload["detail"][string] | null,
    ) => {
      if (r.b === "mega") {
        if (!r.ctx) return ""
        const keys = Object.keys(r.ctx)
        const sc = scale(Math.max(...keys.map((k) => r.ctx![k])))
        return `<div class="psec"><h3>Per context length<span>speedup over eager PyTorch</span></h3>${keys
          .map(
            (k) =>
              `<div class="srow"><span class="sl">${fint(+k.replace("ctx", ""))} tokens of context</span><span class="sv">${fx(r.ctx![k])}</span><span class="track" style="grid-column:1/-1;background-size:${((sc.step / sc.max) * 100).toFixed(3)}% 100%"><i class="seg" style="width:${((r.ctx![k] / sc.max) * 100).toFixed(2)}%;background:var(--green)"></i></span></div>`,
          )
          .join("")}</div>`
      }
      if (!det || !det.sh || !det.sh.length) return ""
      const lab = (s: { l?: string }, i: number) =>
        s.l ? esc(s.l) : `Shape ${i + 1}`
      if (r.p === NSA) {
        const sc = scale(Math.max(...det.sh.map((s) => s.ms || 0)))
        return `<div class="psec"><h3>Per shape<span>latency in ms, lower is better</span></h3>${det.sh
          .map(
            (s, i) =>
              `<div class="srow"><span class="sl num">${lab(s, i)}</span><span class="sv">${s.ms != null ? fms(s.ms) : "–"}</span><span class="track" style="grid-column:1/-1;background-size:${((sc.step / sc.max) * 100).toFixed(3)}% 100%"><i class="seg" style="width:${(((s.ms || 0) / sc.max) * 100).toFixed(2)}%;background:var(--blue)"></i></span></div>`,
          )
          .join("")}</div>`
      }
      const pc = (v?: number) => {
        const x = (v || 0) * 100
        return `${x >= 100 ? fint(x) : x.toFixed(1)}%`
      }
      const bar = (name: string, v: number | undefined, c: string) =>
        `<div class="sb"><span class="sbk">${name}<b>${pc(v)}</b></span><span class="track" style="background-size:25% 100%"><i class="seg" style="width:${Math.min(100, (v || 0) * 100).toFixed(2)}%;background:${c}"></i></span></div>`
      const over = det.sh.some((s) => (s.cu || 0) > 1 || (s.mu || 0) > 1)
      return `<div class="psec"><h3>Per shape roofline<span>share of hardware peak, bars run 0 to 100%</span></h3><div class="legend" style="margin:0 0 6px"><span><i class="sw" style="background:var(--green)"></i>compute, of ${fint(det.pt || 0)} TFLOPS</span><span><i class="sw" style="background:var(--neutral)"></i>memory, of ${fint(det.pb || 0)} GB/s</span></div>${det.sh
        .map(
          (s, i) =>
            `<div class="shp"><div class="sh"><span class="sl">${lab(s, i)}</span><span class="sv">${s.ms != null ? fms(s.ms) : ""}${s.bd ? `<small>${esc(s.bd)}-bound</small>` : ""}</span></div>${bar("compute", s.cu, "var(--green)")}${bar("memory", s.mu, "var(--neutral)")}</div>`,
        )
        .join("")}${over ? '<p class="fn">Above 100%: the roofline model counts more traffic for that shape than the hardware peak. The bar stops at 100%.</p>' : ""}</div>`
    }

    const detailHTML = (rid: string) => {
      const r = D.runs[rid]
      const gpu = r.g || null
      const det = gpu ? D.detail[`${gpu}/${rid}`] || null : null
      const code = D.code[rid]
      const st = cellState(r)
      const kind = kindOf(r)
      const vc = vclass(r.v)
      const ent = board(r.b, gpu || CANON).find((e) => e.m === r.m)
      let big: string
      let sub: string
      const unit =
        kind === "x"
          ? "speedup over eager PyTorch"
          : kind === "ms"
            ? `geomean latency${r.s != null ? ` · ${r.s.toFixed(2)}× the eager anchor` : ""}`
            : "of hardware peak"
      if (r.on && st.k === "ok") {
        big = `<div class="big">${valText(r)}</div>`
        sub = unit
      } else {
        const word = !r.on
          ? VLABEL[r.v || ""] || r.v || "Attempt"
          : st.label.charAt(0).toUpperCase() + st.label.slice(1)
        big = `<div class="big word">${esc(word)}</div>`
        const v = valText(r)
        sub =
          kind === "raw"
            ? `${v ? `archived score ${v}` : "no score"}, problem since removed from the deck`
            : v
              ? `${v} measured, ${!r.on ? "not the board cell" : st.k === "flag" ? "rejected by audit" : "not counted"}`
              : r.on
                ? "no score"
                : "earlier attempt, not on the board"
      }
      sub = `${esc(plabel(r.p))}${gpu ? ` · ${esc(GPU[gpu])}` : ""} · ${sub}`
      const facts: string[] = []
      const add = (k: string, v: string | null | undefined) => {
        if (v != null && v !== "") facts.push(`<div>${k}<b>${esc(v)}</b></div>`)
      }
      add("Harness", (ent && r.on ? ent.h : null) || r.oh)
      add("Effort", (ent && r.on ? ent.e : null) || r.oe)
      add("Kernel", r.fw ? FWLABEL[r.fw] : null)
      if (r.tok) add("Throughput", `${fint(r.tok)} tok/s`)
      const secs = (det && det.st && det.st.agent_s) || r.t
      if (secs) add("Session", fdur(secs))
      if (det && det.st && det.st.output_tokens != null) {
        add("Output tokens", ftok(det.st.output_tokens))
      }
      if (det && det.st && det.st.cost_usd != null) {
        add("Cost", `$${det.st.cost_usd.toFixed(2)}`)
      }
      add("Run", fdate(rid))
      let h = `<div class="headline">${big}<div class="bigsub">${sub}</div><div class="vchip ${vc}"><i class="ic ${vc}">${vicon(vc)}</i>Audit: ${esc(VLABEL[r.v || ""] || r.v || "none")}</div></div>`
      h += `<div class="facts">${facts.join("")}</div>`
      h += `<div class="psec"><h3>Audit summary</h3>${
        r.sum
          ? `<p class="sum${state.full ? "" : " clamp"}" id="sumText">${esc(r.sum)}</p>${
              r.sum.length > 420
                ? `<button class="more" data-more>${state.full ? "Show less" : "Read the full audit"}</button>`
                : ""
            }`
          : '<p class="sum" style="color:var(--ink3)">No audit annotation exists for this run, so it is shown but never scored.</p>'
      }</div>`
      h += shapesHTML(r, det)
      if (code) {
        const lines = code.replace(/\n$/, "").split("\n")
        h += `<div class="psec" id="kernelSec"><h3>Kernel solution<span>first ${lines.length} lines, redacted</span><button class="copy-btn" data-copy="${esc(rid)}">Copy</button></h3><div class="codebox"><pre>${lines
          .map(
            (l) =>
              `<span class="cl${/^\s*(#|\/\/|\/\*|\*)/.test(l) ? " cm" : ""}">${esc(l) || " "}</span>`,
          )
          .join("")}</pre></div></div>`
      }
      const links: string[] = []
      if (r.tr) {
        links.push(
          `<a href="${esc(r.tr)}" target="_blank" rel="noopener">Agent trace on HuggingFace ↗</a>`,
        )
      }
      if (r.su) {
        links.push(
          `<a href="${esc(D.site + r.su)}" target="_blank" rel="noopener">Full kernel file ↗</a>`,
        )
      }
      if (links.length) h += `<div class="links">${links.join("")}</div>`
      return h
    }

    const renderSheet = (keepScroll = false) => {
      const sheet = $("#sheet")
      const panel = $("#panel")
      if (!sheet || !panel) return
      const rid = state.open
      const r = rid ? D.runs[rid] : null
      if (!r || !rid) {
        sheet.hidden = true
        document.documentElement.classList.remove("kb-lock")
        return
      }
      const m = D.models[r.m]
      const log = D.log[r.m] || [rid]
      const cnt: Record<string, number> = {}
      log.forEach((id) => {
        const g = vgroup(D.runs[id].v)
        cnt[g] = (cnt[g] || 0) + 1
      })
      const life = m.st === "cur" ? "" : "archived"
      let h = `<div class="ph">${logoHTML(m)}<span class="pn">${esc(m.n)}</span><span class="pl">${esc([m.l, life].filter(Boolean).join(" · "))}</span><span class="sp"><button class="kb" data-step="-1" aria-label="Previous model"${state.navIdx > 0 ? "" : " disabled"}>←</button><button class="kb" data-step="1" aria-label="Next model"${state.navIdx >= 0 && state.navIdx < state.nav.length - 1 ? "" : " disabled"}>→</button><button class="kb" data-close aria-label="Close audit log"><small>esc</small>✕</button></span></div><div class="pb">`
      h += `<div class="lhead"><span class="lt">Audit log</span><span class="ln">${plural(log.length, "run")} on record, every GPU, newest first in each problem</span></div>`
      h += `<div class="chips">${["ok", "note", "bad", "dead", "un"]
        .filter((g) => cnt[g])
        .map(
          (g) =>
            `<span><i class="ic ${g === "un" ? "dead" : g}">${g === "un" ? "·" : vicon(g)}</i><b>${cnt[g]}</b>${GLABEL[g]}</span>`,
        )
        .join("")}</div>`
      let last: string | null = null
      log.forEach((id) => {
        const x = D.runs[id]
        const key = `${x.b}|${x.p}`
        if (key !== last) {
          last = key
          h += `<div class="lgrp"><b>${esc(plabel(x.p))}</b><span>${BLABEL[x.b]}${inDeck(x) ? "" : " · problem removed from the deck"}</span></div>`
        }
        const open = id === rid && !state.fold
        const c = vclass(x.v)
        const s = cellState(x)
        const val = x.on && s.k === "fail" ? s.label : valText(x)
        h += `<div class="lent${open ? " open" : ""}${x.v && FLAG.has(x.v) ? " rej" : ""}"><button class="lrow" data-run="${esc(id)}" aria-expanded="${open}"><i class="car">›</i><span class="ld">${fdate(id)}</span><span class="lg2">${x.g ? esc(GPU[x.g] || x.g) : ""}</span><span class="lv"><i class="ic ${c}">${vicon(c)}</i>${esc(VLABEL[x.v || ""] || x.v || "Not audited")}${x.on ? `<span class="ontag">board</span>` : ""}</span><span class="lb">${esc(val || "")}</span></button>${open ? `<div class="ldet">${detailHTML(id)}</div>` : ""}</div>`
      })
      h += "</div>"
      const y = panel.scrollTop
      panel.innerHTML = h
      sheet.hidden = false
      document.documentElement.classList.add("kb-lock")
      if (keepScroll) panel.scrollTop = y
      else if (state.openFocus === "kernel") {
        const ks = $("#kernelSec", panel)
        if (ks) panel.scrollTop = Math.max(0, ks.offsetTop - 72)
      } else {
        const ent = $(".lent.open", panel)
        const top = ent ? ent.offsetTop : 0
        panel.scrollTop = top > panel.clientHeight * 0.45 ? top - 132 : 0
      }
    }

    const popWrap = $("#popWrap")!
    const popEl = $("#pop")!
    const clearPopTimers = () => {
      if (popState.hideTimer) {
        clearTimeout(popState.hideTimer)
        popState.hideTimer = 0
      }
      if (popState.switchTimer) {
        clearTimeout(popState.switchTimer)
        popState.switchTimer = 0
      }
    }
    const hidePop = () => {
      clearPopTimers()
      if (popState.anchor) popState.anchor.classList.remove("pop-active")
      popState.rid = null
      popState.rids = []
      popState.anchor = null
      popState.anchorX = null
      popState.overPop = false
      popWrap.hidden = true
    }
    const schedulePopHide = () => {
      if (popState.overPop) return
      if (popState.hideTimer) clearTimeout(popState.hideTimer)
      popState.hideTimer = setTimeout(() => {
        if (!popState.overPop) hidePop()
      }, 220)
    }
    const positionPop = () => {
      if (popWrap.hidden || !popState.anchor) return
      const r = popState.anchor.getBoundingClientRect()
      const pw = popWrap.offsetWidth
      const ph = popWrap.offsetHeight
      const vw = window.innerWidth
      const vh = window.innerHeight
      const targetX =
        popState.anchorX != null
          ? popState.anchorX - pw * 0.42
          : r.left + Math.min(180, r.width * 0.25)
      const left = Math.min(vw - pw - 12, Math.max(12, targetX))
      let top = r.bottom - 4
      if (top + ph > vh - 8 && r.top - ph + 4 >= 8) top = r.top - ph + 4
      else if (top + ph > vh - 8) top = Math.max(8, vh - ph - 8)
      popWrap.style.left = `${Math.round(left)}px`
      popWrap.style.top = `${Math.round(top)}px`
    }
    const renderPop = (reposition = true) => {
      const rid = popState.rid
      const r = rid ? D.runs[rid] : null
      if (!r || !rid) {
        hidePop()
        return
      }
      const m = D.models[r.m]
      const gpu = r.g || state.gpu
      const det = gpu ? D.detail[`${gpu}/${rid}`] || null : null
      const ent = board(r.b, gpu || CANON).find((e) => e.m === r.m)
      const vc = vclass(r.v)
      const hasCode = Boolean(D.code[rid])
      const totalRuns = (D.log[r.m] || []).length
      const meta: string[] = []
      if (r.fw && FWLABEL[r.fw]) meta.push(FWLABEL[r.fw])
      if (r.tok) meta.push(`${fint(r.tok)} tok/s`)
      const hName = (ent && r.on ? ent.h : null) || r.oh
      if (hName) meta.push(hName)
      const secs = (det && det.st && det.st.agent_s) || r.t
      if (secs) meta.push(fdur(secs))
      if (det && det.st && det.st.cost_usd != null) {
        meta.push(`$${det.st.cost_usd.toFixed(2)}`)
      }
      const dt = fdate(rid)
      if (dt) meta.push(dt)
      const probPills =
        popState.rids.length > 1
          ? `<div class="pop-probs">${popState.rids
              .map((id) => {
                const x = D.runs[id]
                return `<button class="pop-p${id === rid ? " on" : ""}" data-pop-rid="${esc(id)}"><i class="sw" style="background:${PCOLOR[x.p] || "var(--green)"}"></i>${esc(D.pshort[x.p] || plabel(x.p))} ${esc(valText(x))}</button>`
              })
              .join("")}</div>`
          : ""
      let h = `<div class="pop-top">${logoHTML(m)}<span class="pop-m">${esc(m.n)}</span><span class="pop-ver ${vc}"><i class="ic ${vc}">${vicon(vc)}</i>${esc(VLABEL[r.v || ""] || r.v || "Audit")}</span><button class="pop-x" data-pop-close aria-label="Close menu">✕</button></div>${probPills}`
      h += `<div class="pop-score"><span class="pop-v">${esc(valText(r) || VLABEL[r.v || ""] || "—")}</span><span class="pop-prob">${esc(plabel(r.p))}${gpu ? ` · ${esc(GPU[gpu] || gpu)}` : ""}</span></div>`
      if (meta.length) {
        h += `<div class="pop-meta">${meta.map((x) => `<span>${esc(x)}</span>`).join("")}</div>`
      }
      if (r.sum) h += `<p class="pop-blurb">${esc(r.sum)}</p>`
      h += '<div class="pop-acts">'
      h += `<button class="pop-btn pri" data-open="${esc(rid)}" data-focus="audit">Audit report</button>`
      if (hasCode) {
        h += `<button class="pop-btn" data-open="${esc(rid)}" data-focus="kernel">Kernel solution</button>`
      } else if (r.su) {
        h += `<a class="pop-btn" href="${esc(D.site + r.su)}" target="_blank" rel="noopener">Kernel solution ↗</a>`
      }
      if (r.tr) {
        h += `<a class="pop-btn" href="${esc(r.tr)}" target="_blank" rel="noopener">Agent trace ↗</a>`
      }
      h += "</div>"
      h += `<div class="pop-foot">${plural(totalRuns, "run")} in model audit log</div>`
      popEl.innerHTML = h
      popWrap.hidden = false
      if (reposition) positionPop()
    }
    const showPop = (rid: string, anchor: HTMLElement, clientX?: number) => {
      if (!D.runs[rid] || state.open) return
      clearPopTimers()
      const row = (anchor.closest(".row") as HTMLElement | null) || anchor
      const sameRow = popState.anchor === row && !popWrap.hidden
      const rids = row.dataset?.rids
        ? row.dataset.rids.split(",").filter(Boolean)
        : [rid]
      if (popState.anchor && popState.anchor !== row) {
        popState.anchor.classList.remove("pop-active")
      }
      popState.rid = rid
      popState.rids = rids.includes(rid) ? rids : [rid]
      popState.anchor = row
      if (!sameRow) popState.anchorX = typeof clientX === "number" ? clientX : null
      row.classList.add("pop-active")
      renderPop(!sameRow)
    }
    const movingTowardPop = (e: MouseEvent) => {
      if (popWrap.hidden || !popState.lastMouse) return false
      const pr = popWrap.getBoundingClientRect()
      const py = popState.lastMouse.y
      const nx = e.clientX
      const ny = e.clientY
      const popBelow = pr.top >= py - 10
      const popAbove = pr.bottom <= py + 10
      const towardY = (popBelow && ny >= py - 2) || (popAbove && ny <= py + 2)
      const nearX = nx >= pr.left - 36 && nx <= pr.right + 36
      return towardY && nearX
    }

    let lastFocus: HTMLElement | null = null
    const writeHash = () => {
      const p = new URLSearchParams()
      if (state.gpu !== CANON) p.set("gpu", state.gpu)
      if (state.tab !== "all") p.set("tab", state.tab)
      if (state.older) p.set("older", "1")
      if (state.hard) p.set("hard", "1")
      if (state.open) p.set("open", state.open)
      const s = p.toString()
      try {
        history.replaceState(
          null,
          "",
          s ? `#${s}` : location.pathname + location.search,
        )
      } catch {
        // ignore history restrictions
      }
    }
    const openRun = (
      rid: string,
      from: HTMLElement | null,
      focusMode: "audit" | "kernel" = "audit",
    ) => {
      if (!D.runs[rid]) return
      const navSource =
        popState.anchor && !from?.closest("[data-list]") ? popState.anchor : from
      hidePop()
      if (navSource) {
        const box = navSource.closest("[data-list]")
        const row = (navSource.closest(".row") as HTMLElement | null) || navSource
        if (box) {
          const rows = $$(".row[data-pop], .row[data-open]", box)
          state.nav = rows.map((x) => x.dataset.pop || x.dataset.open || "")
          state.navIdx = rows.indexOf(row)
        } else {
          state.nav = []
          state.navIdx = -1
        }
        if (!state.open) lastFocus = row
      }
      state.open = rid
      state.openFocus = focusMode
      state.fold = false
      state.full = false
      renderSheet()
      writeHash()
      const c = $("#panel [data-close]")
      if (c) c.focus({ preventScroll: true })
    }
    const closeSheet = () => {
      if (!state.open) return
      state.open = null
      renderSheet()
      writeHash()
      if (lastFocus && document.contains(lastFocus)) {
        lastFocus.focus({ preventScroll: true })
      }
      lastFocus = null
    }
    const stepRun = (d: number) => {
      const i = state.navIdx + d
      if (state.navIdx < 0 || i < 0 || i >= state.nav.length) return
      state.navIdx = i
      state.open = state.nav[i]
      state.openFocus = "audit"
      state.fold = false
      state.full = false
      renderSheet()
      writeHash()
      const c = $("#panel [data-close]")
      if (c) c.focus({ preventScroll: true })
    }
    const pickRun = (rid: string) => {
      if (rid === state.open) {
        state.fold = !state.fold
        renderSheet(true)
      } else {
        state.open = rid
        state.openFocus = "audit"
        state.fold = false
        state.full = false
        renderSheet()
        writeHash()
      }
      const b = $(`#panel .lrow[data-run="${rid}"]`)
      if (b) b.focus({ preventScroll: true })
    }

    const readHash = () => {
      const raw = location.hash.slice(1)
      if (raw === "mega" || raw === "cuda") return raw
      const p = new URLSearchParams(raw)
      const g = p.get("gpu")
      state.gpu = g && GPU_IDS.has(g) ? g : CANON
      const t = p.get("tab")
      state.tab = t && D.problems.cuda.includes(t) ? t : "all"
      state.hard = p.get("hard") === "1"
      state.older = p.get("older") === "1"
      const o = p.get("open")
      state.open = o && D.runs[o] ? o : null
      state.fold = false
      state.full = p.get("full") === "1"
      const hd = $<HTMLDetailsElement>("#hardD")
      if (hd) hd.open = state.hard
      const od = $<HTMLDetailsElement>("#olderD")
      if (od) od.open = state.older
      const md = $<HTMLDetailsElement>("#methodD")
      if (md && p.get("method") === "1") md.open = true
      return p.get("at")
    }

    const renderBoards = () => {
      hidePop()
      renderControls()
      renderMega()
      renderCuda()
      if (state.older) renderOlder()
      if (state.hard) renderHard()
    }

    const tip = $("#tip")!
    const canHover = window.matchMedia("(hover: hover)").matches
    const placeTip = (e: MouseEvent) => {
      if (tip.hidden) return
      const w = tip.offsetWidth
      const h = tip.offsetHeight
      const x = Math.min(window.innerWidth - w - 10, e.clientX + 14)
      tip.style.left = `${Math.max(8, x)}px`
      tip.style.top = `${e.clientY + 18 + h > window.innerHeight - 8 ? e.clientY - h - 12 : e.clientY + 18}px`
    }

    const onPopEnter = () => {
      popState.overPop = true
      clearPopTimers()
    }
    const onPopLeave = () => {
      popState.overPop = false
      schedulePopHide()
    }
    const onClick = (e: MouseEvent) => {
      const target = e.target as HTMLElement | null
      if (!target) return
      const t = target.closest<HTMLElement>(
        "[data-gpu],[data-tab],[data-open],[data-pop],[data-pop-rid],[data-pop-close],[data-run],[data-close],[data-step],[data-more],[data-copy]",
      )
      if (!t) {
        if (target.id === "sheet") closeSheet()
        else if (!popWrap.hidden && !target.closest("#popWrap")) hidePop()
        return
      }
      const d = t.dataset
      if (d.gpu) {
        state.gpu = d.gpu
        renderBoards()
        writeHash()
      } else if (d.tab) {
        state.tab = d.tab
        renderCuda()
        writeHash()
      } else if ("popClose" in d) {
        hidePop()
      } else if (d.popRid) {
        popState.rid = d.popRid
        renderPop(false)
      } else if (d.open) {
        openRun(d.open, t, d.focus === "kernel" ? "kernel" : "audit")
      } else if (d.pop) {
        const row = (t.closest(".row") as HTMLElement | null) || t
        if (!popWrap.hidden && popState.anchor === row && popState.rid === d.pop && !canHover) {
          hidePop()
        } else {
          showPop(d.pop, t, e.clientX)
        }
      } else if (d.run) {
        pickRun(d.run)
      } else if ("close" in d) {
        closeSheet()
      } else if (d.step) {
        stepRun(+d.step)
      } else if ("more" in d) {
        state.full = !state.full
        const s = $("#sumText")
        if (s) s.classList.toggle("clamp", !state.full)
        t.textContent = state.full ? "Show less" : "Read the full audit"
      } else if (d.copy && D.code[d.copy]) {
        try {
          navigator.clipboard.writeText(D.code[d.copy])
          t.textContent = "Copied"
          setTimeout(() => {
            t.textContent = "Copy"
          }, 1400)
        } catch {
          // ignore clipboard error
        }
      }
    }
    const onClickCapture = () => {
      tip.hidden = true
    }
    const onOlderToggle = (e: Event) => {
      const open = (e.target as HTMLDetailsElement).open
      if (state.older === open) {
        if (state.older && !$("#olderBox")?.firstChild) renderOlder()
        return
      }
      state.older = open
      renderOlder()
      writeHash()
    }
    const onHardToggle = (e: Event) => {
      const open = (e.target as HTMLDetailsElement).open
      if (state.hard === open) {
        if (state.hard && !$("#hardBox")?.firstChild) renderHard()
        return
      }
      state.hard = open
      renderHard()
      writeHash()
    }
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null
      const typing = target ? /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName) : false
      if (e.key === "Escape") {
        if (state.open) {
          e.preventDefault()
          closeSheet()
        } else if (!popWrap.hidden) {
          e.preventDefault()
          hidePop()
        }
        return
      }
      if (typing || e.metaKey || e.ctrlKey || e.altKey) return
      if (state.open) {
        if (e.key === "ArrowRight" || e.key === "j") {
          e.preventDefault()
          stepRun(1)
        } else if (e.key === "ArrowLeft" || e.key === "k") {
          e.preventDefault()
          stepRun(-1)
        }
        return
      }
      const row = target?.closest<HTMLElement>(".row[data-pop]")
      if (row && target === row) {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault()
          if (row.dataset.pop) showPop(row.dataset.pop, row)
        } else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
          const rows = $$(".row[data-pop]")
          const i = rows.indexOf(row) + (e.key === "ArrowDown" ? 1 : -1)
          if (rows[i]) {
            e.preventDefault()
            rows[i].focus()
          }
        }
      }
    }
    const onMouseOver = (e: MouseEvent) => {
      if (!canHover) return
      const target = e.target as HTMLElement | null
      if (!target) return
      if (target.closest("#popWrap")) {
        popState.overPop = true
        clearPopTimers()
      } else {
        const pt = target.closest<HTMLElement>("[data-pop]")
        if (pt && !state.open && !target.closest("[data-open]")) {
          const row = (pt.closest(".row") as HTMLElement | null) || pt
          const rid = pt.dataset.pop || ""
          if (popState.anchor === row && !popWrap.hidden) {
            clearPopTimers()
            if (popState.rid !== rid) showPop(rid, pt, e.clientX)
          } else if (!popWrap.hidden && movingTowardPop(e)) {
            if (popState.switchTimer) clearTimeout(popState.switchTimer)
            const cx = e.clientX
            popState.switchTimer = setTimeout(() => {
              if (!popState.overPop) showPop(rid, pt, cx)
            }, 260)
          } else {
            showPop(rid, pt, e.clientX)
          }
        } else if (!pt && !popWrap.hidden) {
          schedulePopHide()
        }
      }
      const t = target.closest<HTMLElement>("[data-tip]")
      if (!t) {
        tip.hidden = true
        return
      }
      tip.textContent = t.dataset.tip || ""
      tip.hidden = false
      placeTip(e)
    }
    const onMouseMove = (e: MouseEvent) => {
      popState.lastMouse = { x: e.clientX, y: e.clientY }
      placeTip(e)
    }
    const onScrollOrResize = () => {
      if (!popWrap.hidden) positionPop()
    }
    const onHashChange = () => {
      readHash()
      renderBoards()
      renderOlder()
      renderHard()
      renderSheet()
    }

    popWrap.addEventListener("mouseenter", onPopEnter)
    popWrap.addEventListener("mouseleave", onPopLeave)
    root.addEventListener("click", onClick)
    root.addEventListener("click", onClickCapture, true)
    $("#olderD")?.addEventListener("toggle", onOlderToggle)
    $("#hardD")?.addEventListener("toggle", onHardToggle)
    document.addEventListener("keydown", onKeyDown)
    root.addEventListener("mouseover", onMouseOver)
    document.addEventListener("mousemove", onMouseMove)
    window.addEventListener("scroll", onScrollOrResize, { passive: true })
    window.addEventListener("resize", onScrollOrResize)
    window.addEventListener("hashchange", onHashChange)

    const at = readHash()
    const genNote = $("#genNote")
    if (genNote) genNote.textContent = `Board data generated ${D.generated.slice(0, 10)}.`
    renderBoards()
    renderMethod()
    renderOlder()
    renderHard()
    renderSheet()
    if (at) {
      const targetEl = document.getElementById(at)
      if (targetEl) {
        requestAnimationFrame(() =>
          targetEl.scrollIntoView({ behavior: "instant", block: "start" }),
        )
      }
    }

    return () => {
      clearPopTimers()
      document.documentElement.classList.remove("kb-lock")
      popWrap.removeEventListener("mouseenter", onPopEnter)
      popWrap.removeEventListener("mouseleave", onPopLeave)
      root.removeEventListener("click", onClick)
      root.removeEventListener("click", onClickCapture, true)
      $("#olderD")?.removeEventListener("toggle", onOlderToggle)
      $("#hardD")?.removeEventListener("toggle", onHardToggle)
      document.removeEventListener("keydown", onKeyDown)
      root.removeEventListener("mouseover", onMouseOver)
      document.removeEventListener("mousemove", onMouseMove)
      window.removeEventListener("scroll", onScrollOrResize)
      window.removeEventListener("resize", onScrollOrResize)
      window.removeEventListener("hashchange", onHashChange)
    }
  }, [data])

  return (
    <div className="kb-shell" ref={rootRef}>
      <header className="hero">
        <svg
          className="sk sk-hero"
          viewBox="0 0 560 330"
          aria-hidden="true"
          dangerouslySetInnerHTML={{ __html: sketches.hero }}
        />
        <div className="col hero-in">
          <h1>
            Which model writes the{" "}
            <span className="ul">
              fastest
              <svg
                viewBox="0 0 100 10"
                preserveAspectRatio="none"
                aria-hidden="true"
              >
                <path d="M2 5.5C24 2.4 62 7.6 98 3.4" />
                <path d="M6 8.2C36 5.6 70 8.8 95 6.6" />
              </svg>
            </span>{" "}
            GPU kernels?
          </h1>
          <p className="lede">
            Coding agents get a real GPU and unlimited time. Every score is
            regraded in isolation and audited before it ships.
          </p>
          <p className="cline" id="censusLine" />
          <svg
            className="sk sk-strip"
            viewBox="0 0 328 134"
            aria-hidden="true"
            dangerouslySetInnerHTML={{ __html: sketches.strip }}
          />
        </div>
      </header>

      <div className="bar">
        <div className="col">
          <div className="seg-ctl" id="gpuSeg" role="group" aria-label="GPU" />
          <span className="note" id="boardNote" />
        </div>
      </div>

      <main className="col">
        <section className="blk" id="mega">
          <svg
            className="sk doodle l"
            viewBox="0 0 200 250"
            aria-hidden="true"
            dangerouslySetInnerHTML={{ __html: sketches.mega }}
          />
          <h2>Mega</h2>
          <p className="sub">
            One fused megakernel: a full Kimi-Linear W4A16 decode step.
          </p>
          <div className="chart" id="megaChart" data-list="" />
        </section>

        <section className="blk" id="cuda">
          <svg
            className="sk doodle r"
            viewBox="0 0 200 198"
            aria-hidden="true"
            dangerouslySetInnerHTML={{ __html: sketches.cuda }}
          />
          <h2>CUDA</h2>
          <p className="sub">
            Three hand-written CUDA kernels. Triton, DSLs and pure PyTorch are
            rejected.
          </p>
          <div className="tabs" id="cudaTabs" role="tablist" />
          <div className="chart" id="cudaChart" data-list="" />
        </section>

        <div className="discs">
          <details className="disc" id="methodD">
            <summary>
              Method
              <span>how a run is written, regraded, audited and ranked</span>
            </summary>
            <div className="dbody">
              <div className="steps">
                <div className="step">
                  <i />
                  <b>Write</b>
                  <span>An agent gets a live GPU and unlimited time.</span>
                </div>
                <div className="step">
                  <i />
                  <b>Regrade</b>
                  <span>The final kernel runs alone on a quiet GPU.</span>
                </div>
                <div className="step">
                  <i />
                  <b>Audit</b>
                  <span>Kernel and transcript are read end to end.</span>
                </div>
                <div className="step">
                  <i />
                  <b>Publish</b>
                  <span>Score, code and agent trace go public.</span>
                </div>
              </div>
              <div id="censusBox" />
              <p className="rule">
                <b>Ranking.</b> <span id="ruleText" />
              </p>
            </div>
          </details>

          <details className="disc" id="olderD">
            <summary>
              Older &amp; retired models
              <span>
                superseded siblings and earlier models off the main board
              </span>
            </summary>
            <div className="dbody" id="olderBox" />
          </details>

          <details className="disc" id="hardD">
            <summary>
              Hard archive
              <span>
                deprecated six-op deck, frozen 2026-11-23, out of the rankings
              </span>
            </summary>
            <div className="dbody" id="hardBox" />
          </details>
        </div>
      </main>

      <footer className="kb-foot">
        <div className="col">
          <div className="fl">
            <span>
              Independent benchmark by{" "}
              <a
                href="https://elliotarledge.com"
                target="_blank"
                rel="noopener noreferrer"
              >
                Elliot Arledge
              </a>
              . Not affiliated with Stanford KernelBench.
            </span>
            <a
              href="https://github.com/Infatoshi/kernelbench.com"
              target="_blank"
              rel="noopener noreferrer"
            >
              Source
            </a>
            <span id="genNote" />
          </div>
        </div>
      </footer>

      <div
        className="sheet"
        id="sheet"
        hidden
        role="dialog"
        aria-modal="true"
        aria-label="Model audit log"
      >
        <div className="panel" id="panel" />
      </div>
      <div className="pop-wrap" id="popWrap" hidden>
        <div className="pop" id="pop" role="dialog" aria-label="Run quick menu" />
      </div>
      <div className="tip" id="tip" hidden />
    </div>
  )
}
