"""Section-aware chunking for academic papers.

Naive RAG splits a document into fixed-size windows and throws away structure.
arXiv papers have strong structure (Abstract, Introduction, Methods, Results,
References...), and keeping track of *which section* a chunk came from does two
useful things:

  1. It lets the splitter avoid cutting across section boundaries, so a chunk
     never blends "Results" text with "References" text.
  2. It attaches a `section` label to every chunk's metadata, which the
     retriever can surface in citations and which makes evaluation far easier
     to interpret ("we keep missing Methods-section questions").

PDF text extraction is messy, so we detect headers heuristically with a couple
of regexes rather than relying on a perfect document model. This is good enough
in practice and degrades gracefully: if no headers are found, the whole paper
becomes one "Body" section and we still chunk it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Numbered headers: "1 Introduction", "2. Related Work", "3.1 Model", "IV. Results"
_NUMBERED = re.compile(
    r"^\s*((?:\d+(?:\.\d+)*)|(?:[IVX]+))\.?\s+([A-Z][A-Za-z0-9 \-,&]{2,60})\s*$"
)
# Bare canonical headers that often appear without numbering.
_CANONICAL = re.compile(
    r"^\s*(Abstract|Introduction|Background|Related Work|Methods?|Methodology|"
    r"Approach|Experiments?|Results?|Evaluation|Discussion|Conclusions?|"
    r"References|Acknowledgements?|Appendix)\s*$",
    re.IGNORECASE,
)


@dataclass
class Section:
    title: str
    text: str


def detect_sections(full_text: str) -> list[Section]:
    """Split a paper's raw text into (title, text) sections by scanning lines
    for header-looking lines. Everything before the first header is 'Front'."""
    lines = full_text.splitlines()
    sections: list[Section] = []
    current_title = "Front"
    buffer: list[str] = []

    def flush():
        body = "\n".join(buffer).strip()
        if body:
            sections.append(Section(title=current_title, text=body))

    for line in lines:
        stripped = line.strip()
        if not stripped:
            buffer.append(line)
            continue

        header = None
        m = _NUMBERED.match(stripped)
        if m and len(stripped) < 70:
            header = m.group(2).strip()
        elif _CANONICAL.match(stripped):
            header = stripped.title()

        if header:
            flush()                 # close the previous section
            current_title = header
            buffer = []
        else:
            buffer.append(line)

    flush()
    # If we found basically nothing, treat the document as one block.
    if len(sections) <= 1:
        return [Section(title="Body", text=full_text.strip())]
    return sections


def chunk_documents(
    pages: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Turn a list of per-page Documents (from PyPDFLoader) into section-aware,
    size-bounded chunks with rich metadata.

    Each returned chunk carries:
      - source:     the paper filename / id (from the loader metadata)
      - title:      paper title if known
      - section:    detected section name
      - chunk_id:   stable unique id (used as the retrieval target in eval)
    """
    source = pages[0].metadata.get("source", "unknown") if pages else "unknown"
    title = pages[0].metadata.get("title", source)
    full_text = "\n".join(p.page_content for p in pages)

    sections = detect_sections(full_text)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    for section in sections:
        for piece in splitter.split_text(section.text):
            piece = piece.strip()
            if len(piece) < 50:        # drop tiny fragments / orphaned headers
                continue
            idx = len(chunks)
            chunks.append(
                Document(
                    page_content=piece,
                    metadata={
                        "source": source,
                        "title": title,
                        "section": section.title,
                        "chunk_id": f"{source}::chunk_{idx}",
                    },
                )
            )
    return chunks
