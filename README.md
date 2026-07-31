# Research-Paper RAG Assistant

A retrieval-augmented Q&A system over arXiv papers. Ask natural-language
questions and get answers grounded in the actual paper text, with section-level
citations — plus a quantitative evaluation harness (Recall@K, MRR) so retrieval
quality is measured, not assumed.

**Stack:** Python · LangChain (v1) · FAISS · OpenAI API · PyPDF

---

## Why this isn't just another RAG demo

Two design choices set it apart from a tutorial pipeline:

1. **Section-aware chunking.** Instead of blindly slicing every paper into
   fixed-size windows, the chunker detects section headers (Abstract,
   Introduction, Methods, Results, ...) and chunks *within* sections. Every
   chunk carries its section label, so the assistant can cite `[Methods]` vs
   `[Results]` and the evaluation can break recall down by section.

2. **A real evaluation framework.** Retrieval quality is scored with
   **Recall@K** (is the right passage in the top K?) and **MRR** (how highly is
   it ranked?). The eval set can be bootstrapped automatically: sample chunks,
   have the LLM write a question answerable only by each chunk, and treat that
   chunk as the gold target. No manual labeling required.

---

## How it works

```
PDFs ─▶ section-aware chunk ─▶ OpenAI embeddings ─▶ FAISS index   (build once)
                                                         │
question ─────────────────────────────────────▶ similarity search ─▶ top-K chunks
                                                         │
                                          prompt (grounded + cite) ─▶ GPT ─▶ answer
```

The indexing stage runs once and persists the FAISS index to disk; querying and
evaluation reuse it.

---

## Project layout

```
research-paper-rag/
├── src/
│   ├── config.py       # all tunables (models, chunk size, K, paths)
│   ├── ingest.py       # download arXiv PDFs / load them with PyPDFLoader
│   ├── chunking.py     # ★ section-aware chunking
│   ├── index.py        # build + load the FAISS index
│   ├── qa.py           # the RAG chain (retrieval → prompt → generation)
│   └── evaluate.py     # ★ Recall@K, MRR, synthetic eval-set generation
├── scripts/
│   ├── build_index.py  # build the index (optionally download papers first)
│   ├── ask.py          # query the assistant (single question or interactive)
│   └── run_eval.py     # generate an eval set and/or score retrieval
├── tests/test_core.py  # unit tests for chunker + metrics (no API key needed)
├── data/
│   ├── papers/         # put PDFs here (gitignored)
│   └── eval/           # generated eval sets land here
├── requirements.txt    # pinned, verified versions
└── .env.example
```

---

## Setup

Requires Python 3.10+ and an OpenAI API key.

```bash
# 1. clone and enter
git clone https://github.com/<your-username>/research-paper-rag.git
cd research-paper-rag

# 2. virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. dependencies
pip install -r requirements.txt

# 4. API key
cp .env.example .env               # then edit .env and paste your key
```

---

## Usage

### 1. Build the index

Download a paper by arXiv id and index it in one step (1706.03762 is *Attention
Is All You Need*):

```bash
python scripts/build_index.py --arxiv 1706.03762
```

Or drop your own PDFs into `data/papers/` and just run:

```bash
python scripts/build_index.py
```

### 2. Ask questions

```bash
python scripts/ask.py "What attention mechanism does the paper propose?"
```

```
Multi-head scaled dot-product attention, which lets the model jointly attend to
information from different representation subspaces [Methods]. It replaces
recurrence entirely, enabling far more parallelization [Introduction].

Sources:
  - 1706.03762.pdf [Methods]
  - 1706.03762.pdf [Introduction]
```

Run it with no argument for an interactive loop.

### 3. Evaluate retrieval

Generate 30 synthetic QA pairs and score Recall@K and MRR:

```bash
python scripts/run_eval.py --generate 30 --k 1 3 5
```

```json
{
  "n_queries": 30,
  "k": 3,
  "recall@3": 0.8667,
  "mrr": 0.7944,
  "recall_by_section": {
    "Introduction": 0.95,
    "Methods": 0.86,
    "Results": 0.78
  }
}
```

Re-score an existing eval set without regenerating it:

```bash
python scripts/run_eval.py --k 1 3 5
```

### Run the tests

```bash
python tests/test_core.py
```

---

## Configuration

Everything tunable is an env var (see `.env.example`) read by `src/config.py`:
embedding model, chat model, chunk size/overlap, and top-K. Defaults favour
cost (`text-embedding-3-small`, `gpt-4o-mini`).

---

## Notes & limitations

- **PDF extraction is imperfect.** Header detection is heuristic; papers with
  unusual formatting may merge sections. If no headers are found the document
  falls back to a single block and still chunks cleanly.
- **Costs.** Embedding a typical paper is fractions of a cent; each question is
  roughly $0.005–0.01 with the default models.
- **`allow_dangerous_deserialization=True`** is required to load a FAISS index
  (it unpickles the docstore). Only load indexes you built yourself.

## License

MIT — see [LICENSE](LICENSE).
