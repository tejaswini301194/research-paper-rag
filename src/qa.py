"""The RAG question-answering chain.

Retrieval -> prompt assembly -> generation. The prompt is engineered to do
three things that matter for a research-paper assistant:
  1. Ground strictly in retrieved context (refuse rather than hallucinate).
  2. Cite the section each claim comes from, using the chunk metadata.
  3. Stay concise and technical, matching the audience for arXiv papers.
"""
from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from . import config

SYSTEM_PROMPT = """You are a precise research assistant answering questions about \
academic papers. Use ONLY the context passages provided. Each passage is tagged \
with its source paper and section.

Rules:
- If the context does not contain the answer, say so plainly. Do not invent results, \
numbers, or citations.
- Be concise and technical; assume the reader knows the field.
- After each claim, cite the section it came from in brackets, e.g. [Methods] or \
[Results]. If passages disagree, note the disagreement.
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


def format_context(docs: list[Document]) -> str:
    blocks = []
    for d in docs:
        tag = f"{d.metadata.get('source', '?')} | {d.metadata.get('section', '?')}"
        blocks.append(f"[{tag}]\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


def build_chain(store: FAISS, top_k: int | None = None):
    """Return (chain, retriever). chain.invoke(question) -> answer string."""
    retriever = store.as_retriever(search_kwargs={"k": top_k or config.TOP_K})
    llm = ChatOpenAI(model=config.CHAT_MODEL, temperature=0)

    chain = (
        {
            "context": retriever | RunnableLambda(format_context),
            "question": RunnablePassthrough(),
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def answer(store: FAISS, question: str, top_k: int | None = None) -> dict:
    """Convenience wrapper that returns the answer plus the sources used."""
    chain, retriever = build_chain(store, top_k)
    sources = retriever.invoke(question)
    return {
        "answer": chain.invoke(question),
        "sources": [
            {"source": d.metadata.get("source"), "section": d.metadata.get("section")}
            for d in sources
        ],
    }
