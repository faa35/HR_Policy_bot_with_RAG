# 💬 HR Policy Chatbot (RAG)

An AI assistant that reads your company's HR documents and answers employee
questions instantly — _"How many leave days do I get?"_, _"What's the WFH
policy?"_, _"How do I claim a referral bonus?"_ — 24/7, grounded in the actual
policy text.

This is a small, portfolio-sized version of what companies like **Leena AI**,
**Moveworks**, and **Workday Assistant** do at enterprise scale.

## 🧱 Tech Stack

| Layer            | Tool                                   |
| ---------------- | -------------------------------------- |
| RAG pipeline     | **LlamaIndex**                         |
| Vector database  | **ChromaDB** (local, zero-config)      |
| Embeddings + LLM | **OpenAI** (`text-embedding-3-small` + `gpt-4o-mini`) |
| PDF parsing      | **PyMuPDF**                            |
| Backend API      | **FastAPI**                            |
| Frontend         | **Streamlit**                          |

## 🏗️ How it works (RAG in one paragraph)

`ingest.py` reads every document in `data/hr_docs/`, splits them into chunks,
turns each chunk into an embedding (a vector), and stores them in ChromaDB.
When a user asks a question, `retriever.py` embeds the question, finds the
**top 3 most similar chunks**, and feeds them + the question to GPT, which writes
an answer using **only** that context. The Streamlit UI just talks to the FastAPI
`/chat` endpoint.

```
data/hr_docs/*.pdf ──ingest──► ChromaDB ──retrieve──► top 3 chunks ──► GPT ──► answer
```

## 📁 Project structure

```
hr-policy-chatbot/
├── backend/
│   ├── config.py            # shared settings (paths, models, chunking)
│   ├── ingest.py            # run once: docs → chunks → embeddings → ChromaDB
│   ├── retriever.py         # query: search ChromaDB → top 3 chunks → GPT answer
│   ├── main.py              # FastAPI app with /chat endpoint
│   ├── make_sample_pdfs.py  # (optional) convert sample .md policies to PDFs
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app.py               # Streamlit chat UI
│   └── Dockerfile
├── data/
│   └── hr_docs/             # drop your PDFs here (4 sample policies included)
├── chroma_store/            # auto-created by ingest.py (git-ignored)
├── .env.example             # template for your OPENAI_API_KEY
├── docker-compose.yml
└── README.md
```

## 🚀 Quick start (local)

> Requires Python 3.10+ and an OpenAI API key.

### 1. Install dependencies

```bash
cd hr-policy-chatbot
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r backend/requirements.txt
```

### 2. Add your API key

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY=sk-...`.

### 3. (Optional) Make the sample docs into real PDFs

Four sample policies are already provided as `.md` (they work as-is). To demo
with real PDFs instead:

```bash
cd backend
python make_sample_pdfs.py
```

### 4. Ingest the documents (run once)

```bash
cd backend
python ingest.py
```

You should see it load the documents and report how many chunks were indexed.

### 5. Start the backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

Test it at <http://localhost:8000/docs>.

### 6. Start the frontend (new terminal)

```bash
cd frontend
streamlit run app.py
```

Open <http://localhost:8501> and start asking questions. 🎉

## 🧪 Try these questions

- "How many annual leave days do I get, and can I carry them over?"
- "What is the work from home policy?"
- "How much is the home-office allowance?"
- "How do I claim travel expenses and what's the per diem?"
- "How does the referral bonus work?"

## 🐳 Run with Docker (optional)

```bash
docker compose up --build
# then run ingest once, e.g.:
docker compose exec backend python ingest.py
```

Frontend: <http://localhost:8501> · Backend: <http://localhost:8000/docs>

## 🔧 Using your own documents

Drop any `.pdf`, `.txt`, `.md`, or `.docx` files into `data/hr_docs/`, delete the
sample ones if you like, then re-run `python ingest.py`. That's it.

## 🌍 Deploying to the web (free)

Three free services, each handling one piece:

| Piece | Platform | Why |
| ----- | -------- | --- |
| User database | **Supabase** (Postgres) | Free hosted SQL database |
| Backend (FastAPI + vector store) | **Render** | Free always-on Python host |
| Frontend (Streamlit) | **Streamlit Community Cloud** | Free, built for Streamlit |

> ⚠️ **Not Vercel.** Vercel is for static sites + short serverless functions. This
> app needs always-on Python servers with a filesystem, which Vercel doesn't provide.

> 🧠 **The vector store needs no separate host.** It's read-only at runtime, so
> [render.yaml](render.yaml) rebuilds it at deploy time (`python backend/ingest.py`)
> from the committed docs in `data/hr_docs/`. No Pinecone/Qdrant needed.

### Step 0 — Push to GitHub
Commit everything and push. (`.env`, `chroma_store/`, and `*.db` are git-ignored —
secrets and local data stay out of the repo. Your HR docs in `data/hr_docs/` **are**
committed so the deploy can build the vector store.)

### Step 1 — Database on Supabase
1. Create a free project at [supabase.com](https://supabase.com).
2. Project Settings → **Database** → **Connection string** → **URI**. Copy it
   (use the **connection pooler** URI). It looks like:
   `postgresql://postgres.xxxx:[PASSWORD]@aws-0-region.pooler.supabase.com:6543/postgres`
3. Replace `[PASSWORD]` with your DB password. This is your `DATABASE_URL`.

### Step 2 — Backend on Render
1. At [render.com](https://render.com): **New +** → **Blueprint** → connect this repo.
   Render reads [render.yaml](render.yaml) automatically.
2. When prompted, set the secret env vars:
   - `OPENAI_API_KEY` — your OpenAI key
   - `DATABASE_URL` — the Supabase URI from Step 1
   - (`JWT_SECRET` is auto-generated; `ORGANIZATIONS` is preset.)
3. Deploy. You'll get a URL like `https://hr-policy-chatbot-api.onrender.com`.
   Check `…/health` returns `{"status":"ok"}` and `…/docs` shows the API.

### Step 3 — Frontend on Streamlit Cloud
1. At [share.streamlit.io](https://share.streamlit.io): **New app** → pick this repo.
2. Set **Main file path** to `frontend/app.py`.
3. In **Advanced settings → Secrets**, add:
   ```toml
   BACKEND_URL = "https://hr-policy-chatbot-api.onrender.com"
   ```
4. Deploy. You get a public `https://your-app.streamlit.app` — **share this link.** 🎉

### Notes
- **Free Render services sleep** after ~15 min idle; the first request then takes
  ~30–60s to wake. Fine for a portfolio; upgrade to a paid plan to keep it warm.
- **Changed your docs?** Re-deploy the Render service — the build re-runs `ingest.py`.
- **Set a strong `JWT_SECRET`** (Render's blueprint generates one automatically).

## 💡 Notes & next steps

- **Cost:** uses `gpt-4o-mini` + `text-embedding-3-small` (very cheap). Switch
  models in `.env` (`LLM_MODEL` / `EMBED_MODEL`).
- **Grounding:** the prompt forces the model to answer only from retrieved
  policy text and to say "I don't have that info" otherwise — so it won't invent
  policies.
- **Ideas to extend:** conversation memory, per-document access control, source
  page citations, an admin upload UI, or swapping OpenAI for a local model.

## ⚠️ Disclaimer

The included policies are **fictional samples** for "Acme Corp" and exist only to
demonstrate the RAG pipeline. Replace them with your own documents for real use.
