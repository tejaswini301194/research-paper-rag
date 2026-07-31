"""Ingestion: get arXiv papers onto disk and load them into LangChain Documents.

Two entry points:
  - download_arxiv(ids): fetch PDFs by arXiv id into data/papers/
  - load_papers():        read every PDF in data/papers/ via PyPDFLoader
"""
from __future__ import annotations

from pathlib import Path

import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from . import config


def download_arxiv(arxiv_ids: list[str]) -> list[Path]:
    """Download papers by arXiv id (e.g. '1706.03762') into PAPERS_DIR.
    Returns the list of local paths."""
    config.PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for aid in arxiv_ids:
        aid = aid.strip()
        out = config.PAPERS_DIR / f"{aid.replace('/', '_')}.pdf"
        if out.exists():
            print(f"  [skip] {out.name} already downloaded")
            paths.append(out)
            continue
        url = f"https://arxiv.org/pdf/{aid}.pdf"
        print(f"  [get ] {url}")
        resp = requests.get(url, timeout=60, headers={"User-Agent": "rag-assistant/1.0"})
        resp.raise_for_status()
        out.write_bytes(resp.content)
        paths.append(out)
    return paths


def load_papers() -> dict[str, list[Document]]:
    """Load every PDF in PAPERS_DIR. Returns {filename: [page Documents]}.

    We normalize the `source` metadata to just the filename so chunk ids stay
    short and stable regardless of where the repo lives on disk.
    """
    papers: dict[str, list[Document]] = {}
    pdfs = sorted(config.PAPERS_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"No PDFs found in {config.PAPERS_DIR}. "
            "Add papers or run: python scripts/build_index.py --arxiv 1706.03762"
        )
    for pdf in pdfs:
        pages = PyPDFLoader(str(pdf)).load()
        for p in pages:
            p.metadata["source"] = pdf.name
        papers[pdf.name] = pages
        print(f"  [load] {pdf.name}: {len(pages)} pages")
    return papers
