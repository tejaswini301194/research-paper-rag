"""Tests for the two pieces of logic that don't need an API key: the
section-aware chunker and the retrieval metrics."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document
from src.chunking import detect_sections, chunk_documents
from src.evaluate import recall_at_k, reciprocal_rank

SAMPLE = """A Toy Paper on Widgets

Abstract
We study widgets and report strong results on the widget benchmark.

1 Introduction
Widgets are everywhere. Prior work ignored them. We do not.

2 Methods
We train a transformer on a corpus of widget descriptions using AdamW.
The learning rate is tuned on a validation split.

3 Results
Our model reaches 92.4 accuracy, beating the baseline of 80.1 points.

4 Conclusion
Widgets matter. Future work will study gadgets.

References
[1] Someone et al. Widgets. 2020.
"""


def test_detect_sections():
    secs = detect_sections(SAMPLE)
    titles = [s.title for s in secs]
    assert "Introduction" in titles, titles
    assert "Methods" in titles, titles
    assert "Results" in titles, titles
    print("detect_sections titles:", titles)


def test_chunk_metadata():
    pages = [Document(page_content=SAMPLE, metadata={"source": "toy.pdf"})]
    chunks = chunk_documents(pages, chunk_size=200, chunk_overlap=20)
    assert chunks, "expected at least one chunk"
    for c in chunks:
        assert c.metadata["source"] == "toy.pdf"
        assert c.metadata["chunk_id"].startswith("toy.pdf::chunk_")
        assert c.metadata["section"]
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    print(f"chunk_documents produced {len(chunks)} unique chunks across sections")


def test_metrics():
    ids = ["c2", "c5", "c1", "c9"]   # gold at rank 3
    assert recall_at_k(ids, "c1", k=3) == 1.0
    assert recall_at_k(ids, "c1", k=2) == 0.0
    assert recall_at_k(ids, "absent", k=4) == 0.0
    assert abs(reciprocal_rank(ids, "c1") - (1 / 3)) < 1e-9
    assert reciprocal_rank(ids, "c2") == 1.0
    assert reciprocal_rank(ids, "absent") == 0.0
    print("metrics: recall@k and reciprocal_rank behave correctly")


if __name__ == "__main__":
    test_detect_sections()
    test_chunk_metadata()
    test_metrics()
    print("\nALL TESTS PASSED")
