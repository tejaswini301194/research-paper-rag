"""Quantitative retrieval evaluation: Recall@K and MRR.

Why these two metrics:
  - Recall@K: of the queries whose answer lives in a known gold chunk, what
    fraction retrieve that chunk somewhere in the top K? Tells you whether the
    right evidence is reaching the LLM at all.
  - MRR (Mean Reciprocal Rank): rewards putting the gold chunk *high* in the
    ranking, not just somewhere in the top K. 1.0 means always rank 1; 0.5 means
    typically rank 2; and so on.

The eval set is a list of {"question", "gold_chunk_id"} records. You can write
these by hand, or generate them automatically with build_eval_set(): for each
of a sample of chunks, ask the LLM to write a question answerable only by that
chunk, and treat that chunk as the gold target. Self-bootstrapped, no labeling.

References sections are excluded from generation: they produce meta-questions
("who wrote paper X") that test citation-list lookup rather than real
research-content retrieval, which skews the metrics unrepresentatively.
"""
from __future__ import annotations

import json
import random

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from . import config


# --- Pure metric functions (unit-tested, no external calls) ------------------

def recall_at_k(retrieved_ids: list[str], gold_id: str, k: int) -> float:
    """1.0 if the gold chunk is in the top-k retrieved ids, else 0.0."""
    return 1.0 if gold_id in retrieved_ids[:k] else 0.0


def reciprocal_rank(retrieved_ids: list[str], gold_id: str) -> float:
    """1 / rank of the gold chunk (1-indexed); 0.0 if it never appears."""
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid == gold_id:
            return 1.0 / rank
    return 0.0


# --- Eval harness ------------------------------------------------------------

def evaluate(store: FAISS, eval_set: list[dict], k: int) -> dict:
    """Run retrieval for every eval record and aggregate Recall@K and MRR."""
    recalls, rrs, per_section = [], [], {}
    for rec in eval_set:
        results = store.similarity_search(rec["question"], k=max(k, 10))
        ids = [d.metadata.get("chunk_id") for d in results]
        r = recall_at_k(ids, rec["gold_chunk_id"], k)
        rr = reciprocal_rank(ids, rec["gold_chunk_id"])
        recalls.append(r)
        rrs.append(rr)
        sec = rec.get("section", "unknown")
        per_section.setdefault(sec, []).append(r)

    n = len(eval_set) or 1
    return {
        "n_queries": len(eval_set),
        "k": k,
        f"recall@{k}": round(sum(recalls) / n, 4),
        "mrr": round(sum(rrs) / n, 4),
        "recall_by_section": {
            s: round(sum(v) / len(v), 4) for s, v in sorted(per_section.items())
        },
    }


# --- Synthetic eval-set generation -------------------------------------------

_QGEN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You write one specific, self-contained question that can be answered "
            "ONLY using the passage below. Do not answer it. Output just the question.",
        ),
        ("human", "Passage:\n{passage}"),
    ]
)


def build_eval_set(store: FAISS, n: int = 30, seed: int = 0) -> list[dict]:
    """Sample n chunks from the index and generate one question each."""
    config.require_api_key()
    docstore = store.docstore._dict  # {id: Document}
    # References sections produce meta-questions ("who wrote paper X") that
    # test citation-list lookup rather than real research-content retrieval.
    docs = [
        d for d in docstore.values()
        if d.metadata.get("section", "").lower() != "references"
    ]
    random.seed(seed)
    sample = random.sample(docs, min(n, len(docs)))

    llm = ChatOpenAI(model=config.CHAT_MODEL, temperature=0.3)
    chain = _QGEN_PROMPT | llm | StrOutputParser()

    eval_set = []
    for i, doc in enumerate(sample, 1):
        question = chain.invoke({"passage": doc.page_content}).strip()
        eval_set.append(
            {
                "question": question,
                "gold_chunk_id": doc.metadata["chunk_id"],
                "section": doc.metadata.get("section", "unknown"),
            }
        )
        print(f"  [{i}/{len(sample)}] {question[:70]}...")
    return eval_set


def save_eval_set(eval_set: list[dict]) -> None:
    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    config.EVAL_FILE.write_text(json.dumps(eval_set, indent=2))


def load_eval_set() -> list[dict]:
    if not config.EVAL_FILE.exists():
        raise FileNotFoundError(
            f"No eval set at {config.EVAL_FILE}. "
            "Generate one: python scripts/run_eval.py --generate 30"
        )
    return json.loads(config.EVAL_FILE.read_text())
