#!/usr/bin/env python3
"""Create the parsing test corpus from a curated public-domain PDF set.

Binary and large test fixtures should be reproducible from a script rather than
existing only as opaque committed blobs. This script (re)builds the MinerU-based
corpus under ``tests/parsing/data/`` from five public PDFs.

What this does:
    1. Downloads 5 public PDFs into scripts/.fixture_pdfs/ (cached).
    2. Runs MinerU on each → tests/parsing/data/mineru/<stem>...
    3. For the integration-test PDF, also produces .html and .docx derivatives
       (via pypandoc) into tests/parsing/data/{html,pandoc}/.

Requirements:
    - MinerU + its backend peer deps (torch, torchvision, transformers,
      onnxruntime, accelerate, and on Apple Silicon also mlx/mlx-vlm). These
      live behind PyPI extras and are bundled in this project as the
      `fixture_regen` dependency group:
          uv sync --group fixture_regen
    - pypandoc-binary (already a base project dependency).
    - GPU recommended. On Apple Silicon, `mineru[mlx]` gives MPS-accelerated VLM,
      so `-b hybrid-auto-engine` (default on darwin) is the high-quality path.
      The lighter `-b pipeline` backend also works on CPU, just slower.

Usage (from repo root):
    uv sync --group fixture_regen                              # one-time setup
    uv run python scripts/create_test_fixtures.py              # download + run
    uv run python scripts/create_test_fixtures.py --skip-download
    uv run python scripts/create_test_fixtures.py --backend pipeline
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MINERU_DIR = REPO / "tests" / "parsing" / "data" / "mineru"
HTML_DIR = REPO / "tests" / "parsing" / "data" / "html"
PANDOC_DIR = REPO / "tests" / "parsing" / "data" / "pandoc"
STAGING = REPO / "scripts" / ".fixture_pdfs"
MINERU_STAGING = STAGING / "_mineru_out"


@dataclass
class Source:
    url: str
    stem: str  # neutral filename root, e.g. "tesla-q4-2024-update"
    full_dir: bool = False  # if True, keep full mineru output dir under <stem>/hybrid_auto/
    also_html: bool = False  # if True, also produce <stem>.html via pandoc
    also_docx: bool = False  # if True, also produce <stem>.docx via pandoc


# Corpus: five diverse public-domain PDFs spanning genres and complexity.
#   - Tesla Q4 update     → integration fixture (mineru + html + docx, multi-format)
#   - NIST SP 800-63B     → image-bearing full hybrid_auto/ dir (deep heading hierarchy)
#   - Attention/BERT/VW   → flat mineru fixtures (smaller, varied genre)
CORPUS: list[Source] = [
    Source(
        url="https://digitalassets.tesla.com/tesla-contents/image/upload/IR/TSLA-Q4-2024-Update.pdf",
        stem="tesla-q4-2024-update",
        also_html=True,
        also_docx=True,
    ),
    Source(
        url="https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-63b.pdf",
        stem="nist-sp-800-63b",
        full_dir=True,
    ),
    Source(
        url="https://arxiv.org/pdf/1706.03762",
        stem="attention-is-all-you-need",
    ),
    Source(
        url="https://arxiv.org/pdf/1810.04805",
        stem="bert-paper",
    ),
    Source(
        url="https://www.volkswagen-group.com/en/publications/more/group-annual-report-2023-2674/download",
        stem="vw-sustainability-2023",
    ),
]


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[cache] {dest.name}")
        return
    print(f"[get  ] {url} -> {dest.name}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def find_mineru_cli() -> str:
    for name in ("mineru", "magic-pdf"):
        if shutil.which(name):
            return name
    sys.exit(
        "error: neither `mineru` nor `magic-pdf` found on PATH.\n"
        "  install via the project dependency group:\n"
        "    uv sync --group fixture_regen\n"
        "  (this pulls in mineru[pipeline]: torch, torchvision, transformers, onnxruntime, ...)"
    )


def run_mineru(pdf: Path, cli: str, backend: str) -> Path:
    """Run MinerU on `pdf` into MINERU_STAGING. Return the produced output subdir."""
    MINERU_STAGING.mkdir(parents=True, exist_ok=True)
    cmd = [cli, "-p", str(pdf), "-o", str(MINERU_STAGING), "-m", "auto", "-b", backend]
    print(f"[run  ] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    pdf_out = MINERU_STAGING / pdf.stem
    for candidate in ("hybrid_auto", "auto", "ocr", "txt"):
        p = pdf_out / candidate
        if p.exists():
            return p
    raise RuntimeError(f"MinerU produced no recognized subdir under {pdf_out}")


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def stage(src_out: Path, source: Source) -> None:
    """Copy MinerU outputs from `src_out` to their target test-fixture path.

    Renames files: every occurrence of the original PDF stem in a filename is
    replaced with `source.stem`. Subdirectories (e.g. ``images/``) are
    preserved structurally.
    """
    orig_stem = src_out.parent.name  # PDF basename used by MinerU

    if source.full_dir:
        target = MINERU_DIR / source.stem / "hybrid_auto"
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        for p in src_out.iterdir():
            new_name = p.name.replace(orig_stem, source.stem)
            _copy_tree(p, target / new_name)
        print(f"[stage] {target.relative_to(REPO)}/  ({sum(1 for _ in target.rglob('*'))} entries)")
    else:
        for suffix in ("_middle.json", "_origin.pdf"):
            src = src_out / f"{orig_stem}{suffix}"
            if not src.exists():
                print(f"[warn ] missing {src.relative_to(REPO)}")
                continue
            dst = MINERU_DIR / f"{source.stem}{suffix}"
            _copy_tree(src, dst)
            print(f"[stage] {dst.relative_to(REPO)}")

    # Pandoc derivatives from MinerU's markdown output.
    if source.also_html or source.also_docx:
        md = src_out / f"{orig_stem}.md"
        if not md.exists():
            print(f"[warn ] no markdown produced at {md} — skipping html/docx")
            return
        try:
            import pypandoc
        except ImportError:
            print("[warn ] pypandoc not importable — run `uv sync` then re-run with --skip-mineru")
            return
        if source.also_html:
            HTML_DIR.mkdir(parents=True, exist_ok=True)
            out = HTML_DIR / f"{source.stem}.html"
            pypandoc.convert_file(str(md), "html", outputfile=str(out), extra_args=["--standalone"])
            print(f"[stage] {out.relative_to(REPO)}")
        if source.also_docx:
            PANDOC_DIR.mkdir(parents=True, exist_ok=True)
            out = PANDOC_DIR / f"{source.stem}.docx"
            pypandoc.convert_file(str(md), "docx", outputfile=str(out))
            print(f"[stage] {out.relative_to(REPO)}")


def print_plan() -> None:
    print("Corpus plan:")
    for src in CORPUS:
        kind = "full hybrid_auto/" if src.full_dir else "flat mineru"
        extra = []
        if src.also_html:
            extra.append("html")
        if src.also_docx:
            extra.append("docx")
        suffix = f"  (+ {' + '.join(extra)})" if extra else ""
        print(f"  {src.stem:30s} -> {kind}{suffix}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--skip-download", action="store_true", help="reuse cached PDFs")
    ap.add_argument("--skip-mineru", action="store_true", help="skip MinerU runs (debugging)")
    # Default to hybrid-auto-engine on macOS (mineru[mlx] gives MPS-accelerated
    # VLM, the high-quality path on Apple Silicon). Fall back to the CPU-friendly
    # pipeline backend everywhere else.
    default_backend = "hybrid-auto-engine" if sys.platform == "darwin" else "pipeline"
    ap.add_argument(
        "--backend",
        default=default_backend,
        choices=("pipeline", "hybrid-auto-engine", "vlm-auto-engine"),
        help=f"MinerU backend (default for this platform: {default_backend}). "
        "`pipeline` is CPU-friendly; `hybrid-auto-engine` and `vlm-auto-engine` "
        "use VLM models (Apple Silicon: via mineru[mlx]).",
    )
    args = ap.parse_args()

    print_plan()
    STAGING.mkdir(parents=True, exist_ok=True)

    pdfs: list[tuple[Source, Path]] = []
    for src in CORPUS:
        pdf = STAGING / f"{src.stem}.pdf"
        if not args.skip_download:
            download(src.url, pdf)
        pdfs.append((src, pdf))

    if not args.skip_mineru:
        cli = find_mineru_cli()
        for src, pdf in pdfs:
            out = run_mineru(pdf, cli, args.backend)
            stage(out, src)

    print("\nDone.")


if __name__ == "__main__":
    main()
