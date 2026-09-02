"""Fail if any http(s) URL in an X article source does not return 200.

Usage:
    uv run --no-project python media/check_article_links.py media/X-article-v4flash/00_article.txt
    uv run --no-project python media/check_article_links.py --all

Never invent a Hugging Face path. Catalog `trace_url` is a hint only —
the site can bake `h100/` for a cell that was never pushed. HEAD/GET the
final href before draft. For HF `/blob/main/` also GET `/resolve/main/`.
A 404 or an "unresolved" / "Entry not found" page is a refuse.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

MEDIA = Path(__file__).resolve().parent
ROOT = MEDIA.parent
CATALOG = ROOT / "public/data/catalog.json"
URL_RE = re.compile(r"https?://[^\s)\]>\"']+")
IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
COVER_RE = re.compile(r"^cover:\s*(\S+)", re.M)
UA = {"User-Agent": "kernelbench-article-linkcheck/1"}
DEAD_MARKERS = ("unresolved", "entry not found", "sorry, we can't find the page")


def extract_urls(text: str) -> list[str]:
    seen: list[str] = []
    for raw in URL_RE.findall(text):
        url = raw.rstrip(".,);")
        if url not in seen:
            seen.append(url)
    return seen


def catalog_trace_index() -> dict[str, str]:
    if not CATALOG.exists():
        return {}
    data = json.loads(CATALOG.read_text())
    cells = data.get("cells") or []
    out: dict[str, str] = {}
    for cell in cells:
        rid = cell.get("run_id")
        url = cell.get("trace_url")
        if rid and url:
            out[rid] = url
    return out


def resolve_twin(url: str) -> str | None:
    if "huggingface.co" in url and "/blob/main/" in url:
        return url.replace("/blob/main/", "/resolve/main/", 1)
    return None


def probe(url: str, timeout: float = 20.0) -> tuple[int, str]:
    body = ""
    last = 0
    for method in ("HEAD", "GET"):
        req = urllib.request.Request(url, headers=UA, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                last = int(resp.status)
                if method == "GET":
                    body = resp.read(8000).decode("utf-8", "replace")
                return last, body
        except urllib.error.HTTPError as e:
            last = int(e.code)
            if method == "HEAD" and e.code in {403, 405}:
                continue
            try:
                body = e.read(8000).decode("utf-8", "replace")
            except Exception:
                body = ""
            return last, body
        except Exception:
            if method == "GET":
                return 0, body
    return last, body


def href_ok(url: str) -> tuple[bool, int, str]:
    status, body = probe(url)
    if status != 200:
        return False, status, "http"
    low = body.lower()
    if any(m in low for m in DEAD_MARKERS):
        return False, status, "unresolved-page"
    twin = resolve_twin(url)
    if twin:
        t_status, t_body = probe(twin)
        if t_status != 200:
            return False, t_status, "resolve"
        t_low = t_body.lower()
        if any(m in t_low for m in DEAD_MARKERS):
            return False, t_status, "resolve-unresolved"
    return True, status, "ok"


def local_images(path: Path, text: str) -> list[str]:
    missing: list[str] = []
    names = list(IMG_RE.findall(text))
    cover = COVER_RE.search(text)
    if cover:
        names.append(cover.group(1))
    for name in names:
        if name.startswith("http"):
            continue
        if not (path.parent / name).is_file():
            missing.append(name)
    return missing


def check_file(path: Path) -> list[tuple[str, int]]:
    text = path.read_text()
    urls = extract_urls(text)
    dead: list[tuple[str, int]] = []
    traces = catalog_trace_index()
    for url in urls:
        ok, status, why = href_ok(url)
        mark = "OK" if ok else "DEAD"
        print(f"{mark:4} {status or '-':>3}  {why:18}  {url}")
        if not ok:
            dead.append((url, status))
        if "/blob/main/h100/" in url:
            stem = Path(url).stem
            if stem in traces and traces[stem] != url:
                print(f"     catalog has {traces[stem]}")
    for name in local_images(path, text):
        print(f"DEAD   -  missing-image      {name}")
        dead.append((name, 0))
    return dead


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", type=Path)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    paths = list(args.paths)
    if args.all:
        paths.extend(sorted(MEDIA.glob("X-article-*/00_article*.txt")))
    if not paths:
        ap.error("pass article paths or --all")
    failed = False
    for path in paths:
        print(f"==== {path} ====")
        dead = check_file(path)
        if dead:
            failed = True
            print(f"REFUSE {path}: {len(dead)} dead href(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
