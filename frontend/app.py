"""
app.py — Streamlit UI for the HR Policy Chatbot (with login + saved history).

Flow:
    Not logged in  -> show Login / Register tabs.
    Logged in      -> sidebar lists your past conversations (+ "New chat" + logout);
                      main area is the chat. Every Q&A is saved to the backend DB.

The login token lives in st.session_state, so refreshing the browser logs you
out (kept simple on purpose). Run with the backend already up on port 8000:
    cd frontend
    streamlit run app.py
"""
import os

import requests
import streamlit as st


def _resolve_backend_url() -> str:
    """Find the backend URL: env var (local) -> Streamlit secret (cloud) -> localhost."""
    url = os.getenv("BACKEND_URL")
    if url:
        return url
    try:
        return st.secrets["BACKEND_URL"]  # set in Streamlit Cloud "Secrets"
    except Exception:  # noqa: BLE001 — no secrets file locally is fine
        return "http://localhost:8000"


BACKEND_URL = _resolve_backend_url()

st.set_page_config(page_title="HR Policy Chatbot", page_icon="💬", layout="centered")


# --------------------------------------------------------------------------- #
# Backend helpers
# --------------------------------------------------------------------------- #
def _auth_headers() -> dict:
    return {"Authorization": f"Bearer {st.session_state.get('token', '')}"}


def api_post(path: str, payload: dict, auth: bool = True) -> requests.Response:
    headers = _auth_headers() if auth else {}
    return requests.post(f"{BACKEND_URL}{path}", json=payload, headers=headers, timeout=60)


def api_get(path: str) -> requests.Response:
    return requests.get(f"{BACKEND_URL}{path}", headers=_auth_headers(), timeout=30)


def api_delete(path: str) -> requests.Response:
    return requests.delete(f"{BACKEND_URL}{path}", headers=_auth_headers(), timeout=30)


def backend_online() -> bool:
    try:
        return requests.get(f"{BACKEND_URL}/health", timeout=3).ok
    except requests.RequestException:
        return False


def render_sources(sources: list[dict]) -> None:
    """Show each retrieved chunk in full, with its file + page + score."""
    if not sources:
        return
    with st.expander("📄 Sources"):
        for s in sources:
            page = f" · page {s['page']}" if s.get("page") else ""
            score = f" · score {s['score']}" if s.get("score") else ""
            st.markdown(f"**{s['file']}**{page}{score}")
            st.text(s["text"])
            st.divider()


# --------------------------------------------------------------------------- #
# Login / Register screen
# --------------------------------------------------------------------------- #
def login_screen() -> None:
    st.title("💬 HR Policy Chatbot")
    st.caption("Sign in to ask questions and keep your conversation history.")

    if not backend_online():
        st.error(f"Backend is offline. Start it and reload. (Expected at {BACKEND_URL})")
        return

    login_tab, register_tab = st.tabs(["Log in", "Create account"])

    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", use_container_width=True)
        if submitted:
            _do_auth("/login", username, password)

    with register_tab:
        with st.form("register_form"):
            username = st.text_input("Choose a username (min 3 chars)")
            password = st.text_input("Choose a password (min 6 chars)", type="password")
            submitted = st.form_submit_button("Create account", use_container_width=True)
        if submitted:
            _do_auth("/register", username, password)


def _do_auth(path: str, username: str, password: str) -> None:
    # Catch obvious problems before hitting the backend, with friendly messages.
    if len(username.strip()) < 3:
        st.error("Username must be at least 3 characters.")
        return
    if len(password) < 6:
        st.error("Password must be at least 6 characters.")
        return
    try:
        resp = api_post(path, {"username": username, "password": password}, auth=False)
        if resp.ok:
            data = resp.json()
            st.session_state.token = data["token"]
            st.session_state.username = data["username"]
            st.session_state.conversation_id = None
            st.session_state.messages = []
            st.rerun()
        else:
            st.error(_friendly_error(resp))
    except requests.RequestException as e:
        st.error(f"Could not reach the backend: {e}")


def _friendly_error(resp: requests.Response) -> str:
    """Turn a backend error response into a single readable sentence."""
    try:
        detail = resp.json().get("detail")
    except Exception:  # noqa: BLE001
        return "Something went wrong. Please try again."
    # FastAPI validation errors come back as a list of dicts; flatten them.
    if isinstance(detail, list):
        return " ".join(d.get("msg", "Invalid input.") for d in detail)
    return detail or "Something went wrong. Please try again."


def logout() -> None:
    for key in ("token", "username", "conversation_id", "messages"):
        st.session_state.pop(key, None)
    st.rerun()


# --------------------------------------------------------------------------- #
# Conversation loading
# --------------------------------------------------------------------------- #
def load_conversation(conv_id: int) -> None:
    resp = api_get(f"/conversations/{conv_id}/messages")
    if resp.ok:
        st.session_state.conversation_id = conv_id
        st.session_state.messages = resp.json()
    else:
        st.error("Could not load that conversation.")


def new_chat() -> None:
    st.session_state.conversation_id = None
    st.session_state.messages = []


# --------------------------------------------------------------------------- #
# Main chat app (logged in)
# --------------------------------------------------------------------------- #
def chat_app() -> None:
    with st.sidebar:
        st.markdown(f"👤 **{st.session_state.username}**")
        if st.button("➕ New chat", use_container_width=True):
            new_chat()
            st.rerun()

        st.divider()
        st.subheader("Your conversations")
        resp = api_get("/conversations")
        conversations = resp.json() if resp.ok else []
        if not conversations:
            st.caption("No conversations yet. Ask something to start one!")
        for c in conversations:
            is_current = c["id"] == st.session_state.get("conversation_id")
            cols = st.columns([0.8, 0.2])
            label = ("▶ " if is_current else "") + c["title"]
            if cols[0].button(label, key=f"conv_{c['id']}", use_container_width=True):
                load_conversation(c["id"])
                st.rerun()
            if cols[1].button("🗑️", key=f"del_{c['id']}", use_container_width=True):
                api_delete(f"/conversations/{c['id']}")
                if is_current:
                    new_chat()
                st.rerun()

        st.divider()
        if st.button("🚪 Log out", use_container_width=True):
            logout()

    # --- Main area ---
    st.title("💬 HR Policy Chatbot")
    st.caption(
        "Ask about HR policies — leave, WFH, expenses, benefits. Answers come "
        "straight from the HR documents."
    )

    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            render_sources(msg.get("sources", []))

    prompt = st.chat_input("Ask an HR question...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching the HR documents..."):
                answer, sources = _send_chat(prompt)
            st.markdown(answer)
            render_sources(sources)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
        st.rerun()  # refresh sidebar (e.g. a brand-new conversation appears)


def _send_chat(prompt: str):
    payload = {"question": prompt, "conversation_id": st.session_state.conversation_id}
    try:
        resp = api_post("/chat", payload)
        if resp.status_code == 401:
            st.warning("Your session expired. Please log in again.")
            logout()
            return "", []
        resp.raise_for_status()
        data = resp.json()
        st.session_state.conversation_id = data["conversation_id"]
        return data["answer"], data.get("sources", [])
    except requests.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:  # noqa: BLE001
            detail = str(e)
        return f"⚠️ {detail}", []
    except requests.RequestException as e:
        return f"⚠️ Could not reach the backend at {BACKEND_URL}. Is it running?\n\n`{e}`", []


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
if "token" not in st.session_state:
    login_screen()
else:
    chat_app()
