"""
app.py — Streamlit chat UI for the HR Policy Chatbot.

It's a thin client: it just sends the user's question to the FastAPI backend
(/chat) and renders the answer + the source snippets it was based on.

Run (backend must already be running on port 8000):
    cd frontend
    streamlit run app.py
"""
import os

import requests
import streamlit as st

# Where the FastAPI backend lives. Override with BACKEND_URL env var if needed.
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="HR Policy Chatbot", page_icon="💬", layout="centered")

st.title("💬 HR Policy Chatbot")
st.caption(
    "Ask me anything about company HR policies — leave, WFH, expenses, benefits. "
    "Answers come straight from the HR documents."
)

# --- Sidebar: status + example questions ---
with st.sidebar:
    st.header("About")
    st.markdown(
        "This assistant uses **RAG** (Retrieval-Augmented Generation):\n"
        "1. Finds the most relevant policy sections\n"
        "2. Asks GPT to answer using only those sections\n\n"
        "Built with FastAPI · LlamaIndex · ChromaDB · OpenAI."
    )

    # Live backend health indicator.
    try:
        ok = requests.get(f"{BACKEND_URL}/health", timeout=3).ok
        if ok:
            st.success("Backend: connected ✅")
        else:
            st.error("Backend: error ❌")
    except requests.RequestException:
        st.error("Backend: offline ❌")
        st.caption(f"Expected at {BACKEND_URL}")

    st.divider()
    st.subheader("Try asking")
    examples = [
        "How many annual leave days do I get?",
        "What is the work from home policy?",
        "How do I claim travel expenses?",
        "How do I apply for a referral bonus?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --- Helpers (defined before use, since Streamlit runs the script top-to-bottom) ---
def render_sources(sources: list[dict]) -> None:
    """Show each retrieved chunk in full, with its file + page + score."""
    with st.expander("📄 Sources"):
        for s in sources:
            page = f" · page {s['page']}" if s.get("page") else ""
            score = f" · score {s['score']}" if s.get("score") else ""
            st.markdown(f"**{s['file']}**{page}{score}")
            # Show the FULL chunk in a scrollable box so the answer line is
            # always visible, even if it sits near the end of the chunk.
            st.text(s["text"])
            st.divider()


def ask_backend(question: str) -> dict:
    resp = requests.post(
        f"{BACKEND_URL}/chat", json={"question": question}, timeout=60
    )
    resp.raise_for_status()
    return resp.json()


# --- Chat state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay history.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])


# Accept input either from the chat box or a clicked example button.
prompt = st.chat_input("Ask an HR question...")
if "pending" in st.session_state:
    prompt = st.session_state.pop("pending")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching the HR documents..."):
            try:
                data = ask_backend(prompt)
                answer = data["answer"]
                sources = data.get("sources", [])
            except requests.HTTPError as e:
                detail = e.response.json().get("detail", str(e))
                answer, sources = f"⚠️ {detail}", []
            except requests.RequestException as e:
                answer, sources = (
                    f"⚠️ Could not reach the backend at {BACKEND_URL}. "
                    f"Is it running?\n\n`{e}`",
                    [],
                )

        st.markdown(answer)
        if sources:
            render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
