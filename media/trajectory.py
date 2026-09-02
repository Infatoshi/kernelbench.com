"""Annotated optimization trajectory for one run: the highlight chart.

Reproduces the Fable 5 Mega 18.7x chart for any archive. Y is the score the
agent measured (Mega: speedup over reference decode; Hard/CUDA: fraction of
roofline). X is wall clock; the bottom axis marks estimated cumulative output
tokens at each checkpoint. Shaded band = measurement + design before the
first passing benchmark. Rose point = a regression the agent measured.

Points come from two places, merged:

1. Automatic. Every `benchmark.py` the agent ran inside the session
   (`peak_fraction:` line in the tool result), the first `check.py` PASS, and
   the first baseline timing line. Works for every harness the viewer parses
   (claude, codex, kimi, cursor, droid, grok).
2. Audit checkpoints. Optional `trajectory:` list in
   `benchmarks/<bench>/results/annotations/<run_id>.yaml` (schema in benchmarks/hard/AGENTS.md), or
   a `--checkpoints file.yaml`. Each item: `t` (minutes), `score`, `label`,
   optional `kind` (`baseline | bench | regress | final`). The audit already
   reads the whole trace; the auditor lists the timed moves it saw there. A
   checkpoint within 0.6 min of an automatic point labels that point instead
   of adding a new one.

Usage:
    uv run --no-project --with matplotlib,numpy,pyyaml python media/trajectory.py \
        benchmarks/mega/outputs/runs/<run_id> [--out media/<run_id>_trajectory.png] \
        [--checkpoints extra.yaml] [--title "..."]

Prints the merged point table so the numbers on the PNG can be checked
against the trace. Inspect the PNG before it goes anywhere.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MEDIA = Path(__file__).resolve().parent
ROOT = MEDIA.parent
sys.path.insert(0, str(MEDIA))
from kbh_theme import C, apply  # noqa: E402

PEAK_RE = re.compile(r"peak_fraction:\s*([0-9]+\.[0-9]+)")
PASS_RE = re.compile(r"(?:^|\\n|\n)PASS\b")
BASELINE_RE = re.compile(r"baseline[^\n]*?([0-9]+\.[0-9]+)\s*ms(?:/tok)?", re.I)


@dataclass
class Point:
    t_min: float
    score: float
    tokens: int
    kind: str = "bench"  # baseline | check | bench | regress | final
    label: str = ""

    @property
    def labeled(self) -> bool:
        return bool(self.label)


# ---------------------------------------------------------------- archive ---

def pick_transcript(run_dir: Path) -> Path | None:
    """Same preference order as benchmarks/hard/scripts/traces_to_hf.py."""
    p = run_dir / "codex_session.jsonl"
    if p.exists() and p.stat().st_size > 0:
        return p
    hist: list[Path] = []
    for base in (run_dir / "agent_home" / ".grok" / "sessions", run_dir / "grok_session"):
        if base.is_dir():
            hist.extend(base.rglob("chat_history.jsonl"))
    if hist:
        return max(hist, key=lambda q: q.stat().st_size)
    stderr = run_dir / "stderr.log"
    if stderr.exists() and stderr.stat().st_size > 0 and b"OpenAI Codex" in stderr.read_bytes()[:4096]:
        return stderr
    for name in ("transcript.jsonl", "transcript.txt"):
        p = run_dir / name
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def load_session(run_dir: Path, bench_dir: Path):
    sys.path.insert(0, str(bench_dir))
    from src.viewer.parsers import parse  # bench-local viewer

    tp = pick_transcript(run_dir)
    if tp is None:
        sys.exit(f"no transcript in {run_dir}")
    return parse(tp), tp


def pretty_model(model_id: str) -> str:
    try:
        models = json.loads((ROOT / "public/data/models.json").read_text())
    except OSError:
        return model_id
    if isinstance(models, dict):
        models = models.get("models", list(models.values()))
    tail = model_id.split("/")[-1].lower()
    for m in models:
        if isinstance(m, dict) and str(m.get("slug", "")).lower() == tail:
            return str(m.get("name") or model_id)
    return model_id


def pretty_problem(problem: str) -> str:
    return re.sub(r"^\d+_", "", problem).replace("_", " ")


# ------------------------------------------------------------ extraction ---

def extract(session, result: dict) -> tuple[list[Point], float, int, list[tuple[float, int]]]:
    """Return (auto points, session minutes, total output tokens, (t, tokens) curve)."""
    start = result.get("start_epoch")
    t0 = datetime.fromtimestamp(start, tz=timezone.utc) if start else None
    if t0 is None:
        t0 = next((e.timestamp for e in session.events if e.timestamp), None)
    total_tokens = int(((result.get("usage") or {}).get("output_tokens")) or 0)
    elapsed_min = float(result.get("elapsed_seconds") or 0) / 60

    # Token proxy: characters the agent emitted, scaled to the billed total.
    chars = 0
    curve: list[tuple[float, int]] = []  # (minutes, chars) after every event
    cmd_by_call: dict[str, str] = {}
    raw: list[tuple[float, float, str, int, str]] = []  # (t, score, kind, chars, label)
    last_t = 0.0
    seen_check = False
    seen_baseline = False
    for e in session.events:
        if e.timestamp and t0:
            ts = e.timestamp if e.timestamp.tzinfo else e.timestamp.replace(tzinfo=timezone.utc)
            last_t = max(last_t, (ts - t0).total_seconds() / 60)
        chars += len(e.text or "") + len(e.reasoning or "")
        for tc in e.tool_calls:
            chars += len(json.dumps(tc.args, default=str))
            if tc.call_id:
                cmd_by_call[tc.call_id] = json.dumps(tc.args, default=str)
        curve.append((last_t, chars))
        tr = e.tool_result
        if not tr:
            continue
        cmd = cmd_by_call.get(tr.call_id or "", "")
        body = tr.content or ""
        # benchmark.py output signature: a numeric peak_fraction line plus the
        # RESULT verdict. Agents often run it in the background and cat the log,
        # so the command text is not a reliable tell; the output is.
        hits = PEAK_RE.findall(body)
        if hits and "RESULT:" in body:
            score = float(hits[-1])
            if score > 0:
                raw.append((last_t, score, "bench", chars, ""))
            continue
        if "check.py" in cmd and not seen_check and PASS_RE.search(body):
            seen_check = True
            raw.append((last_t, float("nan"), "check", chars, "check.py passes"))
            continue
        if not seen_baseline:
            m = BASELINE_RE.search(body)
            if m:
                seen_baseline = True
                raw.append((last_t, 1.0, "baseline", chars, f"baseline timed:\n{m.group(1)} ms floor"))

    if elapsed_min <= 0:
        elapsed_min = last_t
    scale = (total_tokens / chars) if (total_tokens and chars) else 0.25
    pts = [Point(t, s, int(c * scale), k, lbl) for (t, s, k, c, lbl) in raw]
    tok_curve = [(t, int(c * scale)) for t, c in curve]
    return pts, elapsed_min, total_tokens or int(chars * scale), tok_curve


def load_checkpoints(bench_dir: Path, run_id: str, extra: Path | None) -> list[dict]:
    import yaml

    items: list[dict] = []
    ann = bench_dir / "results" / "annotations" / f"{run_id}.yaml"
    if ann.exists():
        data = yaml.safe_load(ann.read_text()) or {}
        items.extend(data.get("trajectory") or [])
    if extra:
        data = yaml.safe_load(Path(extra).read_text()) or {}
        items.extend(data.get("trajectory") if isinstance(data, dict) else data)
    return items


def merge(auto: list[Point], checkpoints: list[dict], scale_tokens) -> list[Point]:
    pts = [p for p in auto if p.kind != "check" or not np.isnan(p.score)]
    check_t = [p.t_min for p in auto if p.kind == "check"]
    for cp in checkpoints:
        t = float(cp["t"])
        score = cp.get("score", cp.get("speedup"))
        label = str(cp.get("label", "")).replace("\\n", "\n")
        kind = cp.get("kind", "bench")
        near = [p for p in pts if abs(p.t_min - t) <= 0.6]
        if near:
            p = min(near, key=lambda q: abs(q.t_min - t))
            if label:
                p.label = label
            if kind != "bench":
                p.kind = kind
            if score is not None:
                p.score = float(score)
        else:
            if score is None:
                continue
            pts.append(Point(t, float(score), scale_tokens(t), kind, label))
    pts.sort(key=lambda p: p.t_min)
    # regressions: a measured drop is rose unless the audit tagged it otherwise
    for prev, cur in zip(pts, pts[1:]):
        if cur.kind == "bench" and cur.score < prev.score * 0.97:
            cur.kind = "regress"
    if pts:
        pts[-1].kind = "final"
        if not pts[-1].label:
            pts[-1].label = "final"
    # the first passing benchmark gets a default label
    firsts = [p for p in pts if p.kind in ("bench", "final") and p.score > 0]
    if firsts and not firsts[0].label:
        firsts[0].label = "first passing benchmark"
    _ = check_t
    return pts


# ------------------------------------------------------------- rendering ---

def fmt_tokens(n: int) -> str:
    return f"{n / 1000:.0f}k" if n < 1_000_000 else f"{n / 1e6:.2f}M"


def render(pts: list[Point], *, out: Path, title: str, subtitle: str, unit: str,
           session_min: float, total_tokens: int, ref_label: str | None) -> None:
    apply()
    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.80, bottom=0.15)

    xs = [p.t_min for p in pts]
    ys = [p.score for p in pts]
    ymax = max(ys) * 1.18 if ys else 1.0
    ax.set_xlim(0, session_min * 1.02)
    ax.set_ylim(0, ymax)

    # design phase: up to the first real benchmark
    first_bench = next((p for p in pts if p.kind in ("bench", "regress", "final") and p.score > 0), None)
    if first_bench and first_bench.t_min > session_min * 0.08:
        ax.axvspan(0, first_bench.t_min, color=C["surface"], zorder=0)
        share = first_bench.t_min / session_min
        ax.text(first_bench.t_min / 2, ymax * 0.97,
                f"measurement + design: {share:.0%} of session",
                ha="center", va="top", color=C["fg_muted"], fontsize=8.5)

    if unit == "x":
        ax.axhline(1.0, color=C["fg_dim"], linewidth=0.9, linestyle="--")
        ax.text(session_min * 0.01, 1.0 + ymax * 0.015, ref_label or "reference = 1x",
                color=C["fg_dim"], fontsize=8)

    # baseline -> first kernel: dashed, nothing existed in between
    if pts and pts[0].kind == "baseline" and len(pts) > 1:
        ax.plot(xs[:2], ys[:2], color=C["accent"], linewidth=1.4, linestyle=(0, (4, 3)),
                alpha=0.55, zorder=3)
        solid_from = 1
    else:
        solid_from = 0
    ax.plot(xs[solid_from:], ys[solid_from:], color=C["accent"], linewidth=1.8, zorder=3)
    for p in pts:
        if p.kind == "regress":
            ax.plot(p.t_min, p.score, "o", color=C["bad"], markersize=6, zorder=5)
        elif p.kind == "final":
            ax.plot(p.t_min, p.score, "o", color=C["accent"], markersize=11, zorder=5,
                    markeredgecolor=C["fg_bright"], markeredgewidth=1.4)
        else:
            ax.plot(p.t_min, p.score, "o", color=C["accent"], markersize=5.5, zorder=4)

    # labels: box-collision layout in data coordinates. Candidates sit close
    # to the point (above, below, left, right); the first one that fits inside
    # the axes without crossing another label, the caption, or the curve wins.
    labeled = [p for p in pts if p.labeled]
    xmax = session_min * 1.02
    char_w = xmax / 128.0      # monospace 8.2pt on a 12.5in axis, with slack
    line_h = ymax * 0.048
    placed_boxes: list[tuple[float, float, float, float]] = []
    if first_bench and first_bench.t_min > session_min * 0.08:
        cap_w = len("measurement + design: 64% of session") * char_w
        cx = first_bench.t_min / 2
        placed_boxes.append((cx - cap_w / 2, ymax * 0.90, cx + cap_w / 2, ymax * 0.99))
    if unit == "x":
        placed_boxes.append((0, 0.9, session_min * 0.01 + 24 * char_w, 1.0 + ymax * 0.06))
    # dense samples along the curve so labels never sit on the line
    if len(xs) > 1:
        sx = np.linspace(xs[0], xs[-1], 400)
        sy = np.interp(sx, xs, ys)
        curve = np.column_stack([sx, sy])
    else:
        curve = np.array([[xs[0], ys[0]]])

    def free(box) -> bool:
        x0, y0, x1, y1 = box
        if x0 < 0 or x1 > xmax or y0 < 0 or y1 > ymax:
            return False
        for bx0, by0, bx1, by1 in placed_boxes:
            if x0 < bx1 and x1 > bx0 and y0 < by1 and y1 > by0:
                return False
        mx, my = xmax * 0.004, ymax * 0.012
        hit = (curve[:, 0] > x0 - mx) & (curve[:, 0] < x1 + mx) & (curve[:, 1] > y0 - my) & (curve[:, 1] < y1 + my)
        return not hit.any()

    for p in labeled:
        lines = p.label.split("\n")
        w = max(len(ln) for ln in lines) * char_w
        h = line_h * len(lines)
        dx, dy = xmax * 0.012, ymax * 0.05
        cands = []
        for k in range(1, 10):
            for yy in (p.score + dy * k, p.score - dy * k - h):
                for xx in (p.t_min + dx, p.t_min - w - dx, p.t_min - w / 2,
                           p.t_min - w - dx * 8, p.t_min - w - dx * 16, p.t_min + dx * 8,
                           p.t_min - w - dx * 26):
                    cands.append((xx, yy))
        cands.sort(key=lambda c: ((c[0] + w / 2 - p.t_min) / xmax) ** 2 * 2.2
                   + ((c[1] + h / 2 - p.score) / ymax) ** 2)
        box = None
        for x0, y0 in cands:
            x0 = min(max(x0, 0.0), xmax - w)
            y0 = min(max(y0, 0.0), ymax - h)
            b = (x0, y0, x0 + w, y0 + h)
            if free(b):
                box = b
                break
        if box is None:  # last resort: nearest in-bounds candidate
            x0 = min(max(p.t_min + dx, 0.0), xmax - w)
            y0 = min(max(p.score + dy, 0.0), ymax - h)
            box = (x0, y0, x0 + w, y0 + h)
        placed_boxes.append(box)
        ax.annotate(p.label, xy=(p.t_min, p.score), xytext=(box[0], box[1]),
                    fontsize=8.2, color=C["fg"], ha="left", va="bottom",
                    arrowprops={"arrowstyle": "-", "color": C["fg_dim"], "linewidth": 0.7,
                                "shrinkA": 0, "shrinkB": 3},
                    zorder=6)

    ax.set_ylabel("speedup over reference" if unit == "x" else "fraction of roofline",
                  color=C["fg_muted"], fontsize=10)
    ax.grid(axis="y", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # bottom axis: estimated cumulative output tokens at checkpoints
    tick_pts = [p for p in pts if p.labeled or p.kind == "final"]
    ticks, labels = [], []
    for p in tick_pts:
        if ticks and p.t_min - ticks[-1] < session_min * 0.035:
            continue
        ticks.append(p.t_min)
        labels.append(fmt_tokens(p.tokens))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=9, color=C["fg_muted"])
    ax.set_xlabel(f"cumulative output tokens, est. from the trace (session total {total_tokens:,})",
                  color=C["fg_muted"], fontsize=9.5)

    # top axis: wall clock
    top = ax.secondary_xaxis("top")
    step = next(st for st in (5, 10, 15, 30, 60, 120, 240) if session_min / st <= 8)
    hours = np.arange(0, session_min + 1, step)
    top.set_xticks(hours)
    top.set_xticklabels([f"{int(h // 60)}h{int(h % 60):02d}" for h in hours], fontsize=9,
                        color=C["fg_muted"])
    top.set_xlabel(f"wall clock (session total {int(session_min // 60)}h{int(session_min % 60):02d}m)",
                   color=C["fg_muted"], fontsize=9.5)
    top.spines["top"].set_color(C["border"])

    fig.text(0.06, 0.945, title, fontsize=13, color=C["fg_bright"], ha="left", va="top")
    fig.text(0.06, 0.895, subtitle, fontsize=8.8, color=C["fg_muted"], ha="left", va="top")
    fig.savefig(out, dpi=170, facecolor=C["bg"])
    plt.close(fig)


# ------------------------------------------------------------------ main ---

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--out")
    ap.add_argument("--checkpoints", help="extra YAML with a trajectory: list")
    ap.add_argument("--title")
    ap.add_argument("--subtitle")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    if not (run_dir / "result.json").exists():
        sys.exit(f"{run_dir} has no result.json")
    bench_dir = run_dir.parents[2]  # outputs/runs/<id> -> benchmarks/<bench>
    bench = bench_dir.name
    result = json.loads((run_dir / "result.json").read_text())
    run_id = run_dir.name

    session, tp = load_session(run_dir, bench_dir)
    auto, session_min, total_tokens, tok_curve = extract(session, result)
    curve_t = np.array([t for t, _ in tok_curve] or [0.0])
    curve_tok = np.array([n for _, n in tok_curve] or [0])

    def scale_tokens(t: float) -> int:
        return int(np.interp(t, curve_t, curve_tok))

    pts = merge(auto, load_checkpoints(bench_dir, run_id, Path(args.checkpoints) if args.checkpoints else None),
                scale_tokens)
    if not pts:
        sys.exit("no benchmark points found in the transcript and no checkpoints given")

    unit = "x" if bench == "mega" else "frac"
    model = pretty_model(str(result.get("model", "")))
    problem = pretty_problem(str(result.get("problem", "")))
    peak = float(result.get("peak_fraction") or pts[-1].score)
    peak_s = f"{peak:.1f}x" if unit == "x" else f"{peak:.3f} of roofline"
    gpu = result.get("gpu_name") or ""
    title = args.title or f"One run, annotated — {model} on {problem}: {peak_s}"
    subtitle = args.subtitle or " · ".join(
        s for s in (f"kernelbench.com/{bench}", gpu, f"{session.harness} harness",
                    "every point = the agent benchmarking its own kernel") if s)
    out = Path(args.out) if args.out else MEDIA / f"{run_id}_trajectory.png"

    print(f"transcript: {tp.relative_to(ROOT) if tp.is_relative_to(ROOT) else tp}")
    print(f"{'min':>7} {'score':>8} {'tok(est)':>9}  kind      label")
    for p in pts:
        print(f"{p.t_min:7.1f} {p.score:8.3f} {p.tokens:>9,}  {p.kind:<9} {p.label.replace(chr(10), ' / ')}")
    render(pts, out=out, title=title, subtitle=subtitle, unit=unit, session_min=session_min,
           total_tokens=total_tokens, ref_label="reference decode = 1x" if bench == "mega" else None)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
