# Research-Paper RAG Assistant

**[Try the live demo →](https://research-paper-rag-kqz6ad6rqv2pvyq6opx89s.streamlit.app/)**

A retrieval-augmented question-answering system for academic papers, built with LangChain v1, FAISS, and the OpenAI API. Ask questions about a PDF corpus and get grounded, cited answers — the system refuses to guess when the retrieved context doesn't contain the answer.

## Why this project

Most RAG demos stop at "it works on one document." This one is built to answer a harder question: **does the retriever actually find the right passage, and how do you know?** Every design decision here — section-aware chunking, the eval harness, the tuning pass — exists to make that measurable rather than assumed.

## What it does

- Downloads papers from arXiv by ID, or ingests any PDFs you drop in `data/papers/`
- Splits each paper along its actual section boundaries (Abstract, Introduction, Methods, Results...) instead of blind fixed-size windows, so a chunk never blends unrelated content
- Embeds chunks and builds a persisted FAISS index
- Answers questions with citations back to the source section, and explicitly declines to answer when the context doesn't support it
- Scores its own retrieval quality with Recall@K and MRR, using a self-bootstrapped eval set (no manual labeling required)

## Live demo

**[research-paper-rag-kqz6ad6rqv2pvyq6opx89s.streamlit.app](https://research-paper-rag-kqz6ad6rqv2pvyq6opx89s.streamlit.app/)**

The hosted demo runs on a single indexed paper ("Attention Is All You Need") for a clean first-time experience. The retrieval numbers below were measured separately on a harder, 6-paper mixed corpus — see [Results](#results).

## Results

Measured on a 6-paper mixed corpus (arXiv's "Attention Is All You Need" plus five unrelated ML papers spanning neuro-symbolic reasoning, mental health NLP, multi-agent systems, domain adaptation, and biophysics):

| Metric | Score |
|---|---|
| Recall@1 | 0.60 |
| Recall@3 | 0.90 |
| Recall@5 | 0.95 |
| MRR | 0.75 |

**How I got here, not just the final number:** the first pass scored Recall@5 = 0.80. Breaking the score down by section showed two problems — `References` sections were dragging the average down (citation-list lookup questions like "who wrote paper X" are a fundamentally different task from real research-content retrieval, and embeddings struggle to distinguish one short citation line from another), and `TOP_K=4` was cutting off answers that would've been found at rank 5. Excluding References from eval generation and raising `TOP_K` to 5 took Recall@5 from 0.80 to 0.95. That before/after is committed in the git history, not just asserted here.

**Honest limitations:**
- The eval questions are LLM-generated from the same chunks being retrieved, which makes them somewhat easier than real user questions — treat these numbers as an optimistic upper bound, not a guarantee of real-world performance.
- PDF text extraction is imperfect. Section-header detection is regex-based and degrades gracefully (falls back to one "Body" section) but won't be perfect on every paper's formatting — one section in testing (`EXPERIMENT ANDRESULTS`) shows a header mashed together by extraction artifacts, a case worth knowing about rather than hiding.
- A `References`-section chunk can still get retrieved by mistake for other query types; excluding it from eval generation doesn't remove it from the corpus, just from the test set.

## Architecture- **`src/chunking.py`** — detects section headers via regex (numbered like "3.1 Methods" and canonical like "Abstract"), chunks within sections so boundaries are never crossed, tags every chunk with source/title/section/chunk_id
- **`src/ingest.py`** — downloads arXiv PDFs by ID, loads any PDF in `data/papers/` via `PyPDFLoader`
- **`src/index.py`** — full build pipeline (load → chunk → embed → FAISS), plus persistence and reload
- **`src/qa.py`** — the LangChain retrieval chain; system prompt enforces grounding, citation by section, and refusal over hallucination
- **`src/evaluate.py`** — pure Recall@K / MRR metric functions, an eval harness with per-section breakdown, and a synthetic eval-set generator that asks the LLM to write one question per sampled chunk
- **`app.py`** — Streamlit UI wrapping the same retrieval chain used by the CLI, deployed on Streamlit Community Cloud

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate      # Git Bash on Windows; use .venv/bin/activate on macOS/Linux

pip install -r requirements.txt

cp .env.example .env               # then paste your OpenAI API key into .env
```

Requires an OpenAI API key with billing enabled. Costs are low: embedding a typical paper is a fraction of a cent, and each question costs roughly $0.005–$0.01 with the default models (`text-embedding-3-small`, `gpt-4o-mini`).

## Usage

```bash
# sanity check — no API key needed
python tests/test_core.py

# download and index a paper (or just run with no --arxiv flag to index whatever's in data/papers/)
python scripts/build_index.py --arxiv 1706.03762

# ask a question
python scripts/ask.py "What attention mechanism does the paper propose?"

# generate a synthetic eval set and score retrieval quality
python scripts/run_eval.py --generate 20 --k 1 3 5

# re-score an existing eval set without regenerating it
python scripts/run_eval.py --k 1 3 5

# run the web demo locally
streamlit run app.py
```

## Tech stack

LangChain v1 (`langchain-core`, `langchain-community`, `langchain-openai`, `langchain-text-splitters`), FAISS, OpenAI API (`text-embedding-3-small`, `gpt-4o-mini`), PyPDF, Streamlit.

## A note on process

This project's git history also includes a real security incident: an OpenAI API key was accidentally committed to `.env.example` during development. It was caught by GitHub's push protection before any secret reached a public remote, the key was revoked immediately, and the local git history was rebuilt from a clean state and verified (via full-history grep) before re-pushing. No secret was ever exposed on the public repo. Leaving this note in rather than scrubbing it from the narrative — mistakes happen, and how you respond to them is the more useful signal.
