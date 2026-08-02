"""Streamlit demo for the Research-Paper RAG Assistant.

Run locally:   streamlit run app.py
Deploy free:   https://share.streamlit.io (Streamlit Community Cloud)
"""
import streamlit as st

from src.index import load_index
from src.qa import answer

st.set_page_config(page_title="Research-Paper RAG Assistant", page_icon="📄")

st.title("📄 Research-Paper RAG Assistant")
st.caption(
    "Section-aware RAG demo, indexed on \"Attention Is All You Need.\" "
    "Full retrieval evaluation (Recall@5 = 0.95, MRR = 0.75) was measured on a "
    "6-paper mixed corpus — see the "
    "[repo](https://github.com/tejaswini301194/research-paper-rag) for details."
)

# --- Load the index once and cache it across reruns --------------------------
@st.cache_resource
def get_store():
    return load_index()

try:
    store = get_store()
except FileNotFoundError:
    st.error(
        "No index found. This demo ships with a pre-built FAISS index — "
        "if you're seeing this, the index files weren't included in the deploy."
    )
    st.stop()

# --- Question input ------------------------------------------------------------
question = st.text_input(
    "Ask a question about the paper:",
    placeholder="e.g. What attention mechanism does the Transformer use?",
)

if st.button("Ask", type="primary") and question:
    with st.spinner("Retrieving and generating..."):
        result = answer(store, question)

    st.markdown("### Answer")
    st.write(result["answer"])

    st.markdown("### Sources")
    for s in result["sources"]:
        st.markdown(f"- `{s['source']}` — *{s['section']}*")

st.divider()
st.caption(
    "Built with LangChain v1, FAISS, and OpenAI. "
    "System prompt is instructed to refuse rather than hallucinate when "
    "context doesn't support an answer — try an off-topic question to see it."
)
