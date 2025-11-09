# Math Tutor Agent

A friendly math-tutor web app with a FastAPI backend and a React + Vite frontend. It answers math questions with step‑by‑step solutions using a retrieval‑augmented pipeline (FAISS + Ollama) and falls back to web search via MCP + Tavily when your local knowledge base isn’t enough. It also supports feedback so answers can be refined and learned from in‑session.

---

## What’s inside

- **Backend**: FastAPI service that
  - builds/loads a **FAISS** index from your JSON math corpus
  - answers via **Ollama** LLM (default `llama3.2`) and **nomic‑embed‑text** embeddings
  - routes by confidence: **Knowledge Base** → **Retrieval chain** → **Web search (MCP/Tavily)**
  - exposes `/chat` and `/feedback` endpoints
- **Frontend**: React + Vite app with a small Express server that
  - serves the UI on one port
  - proxies API calls to the backend (configurable with `BACKEND_URL`)
  - ships a clean chat UI with feedback controls

---

## Quick start

### 0) Prerequisites
- **Python** 3.12+ and **pip**
- **Node.js** 18+ and **npm** (npx available)
- **Ollama** installed and running locally
  - Pull the models used by default:
    ```bash
    ollama pull llama3.2
    ollama pull nomic-embed-text
    ```
- Optional (for web fallback): **Tavily API key** and Node environment for MCP

### 1) Backend setup
```bash
# from repo root
cd backend
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# run the API
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
On first boot the service builds a FAISS index from your dataset (see **Data format**). Subsequent boots load the saved index.

### 2) Frontend setup
```bash
cd frontend
npm install
# dev server (Express + Vite)
npm run dev
```
The UI will serve on **http://localhost:5000** by default. It proxies API calls to the backend running at **http://localhost:8000** (configurable with `BACKEND_URL`).

---

## Environment variables

### Backend (FastAPI)
| Variable | Default | What it does |
|---|---|---|
| `DATA_DIR` | `~/MATH/train` | Folder containing your JSON corpus (see format below) |
| `INDEX_DIR` | `.index` next to `app.py` | Where FAISS index files are stored/loaded |
| `SIMILARITY_THRESHOLD` | `0.60` | Gate for using KB results directly |
| `TOP_K` | `3` | How many docs to retrieve from FAISS/feedback stores |
| `TAVILY_API_KEY` | unset | Enables MCP Tavily web fallback when set |

### Frontend (Express + Vite)
| Variable | Default | What it does |
|---|---|---|
| `PORT` | `5000` | Port for the UI/Express server |
| `BACKEND_URL` | `http://localhost:8000` | Where the UI proxies `/api/chat` and `/api/feedback` |

---

## Data format (your knowledge base)
Place `.json` files under `DATA_DIR` (recursively scanned). Each file should look like:
```json
{
  "problem": "Find the derivative of sin(x).",
  "solution": "d/dx[sin x] = cos x."
}
```

---

## How it answers questions
1. **Safety + routing**
2. **Knowledge Base path**
3. **Web fallback (optional)**
4. **Refinement**
5. **Feedback memory**

---

## API

### `POST /chat`
```json
{ "question": "Integrate x^2" }
```

### `POST /feedback`
```json
{
  "record_id": "<uuid-from-/chat>",
  "rating": 4,
  "comments": "Please show one intermediate step."
}
```

---

## Running in production
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2

cd frontend
npm run build
npm start
```

---

## Troubleshooting
- Backend/Frontend connectivity
- Node deps like `concurrently`
- FAISS index rebuild
- MCP/Tavily availability
- Ollama model not found

---

## Project layout
```
backend/
app.py
frontend/
  client/
  server/
```

---

## Security & guardrails
- Basic content filters
- Math-only classification
- Web answers constrained to snippets
