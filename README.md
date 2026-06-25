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
