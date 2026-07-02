# SHL Conversational Assessment Recommender

FastAPI service that turns a vague hiring intent into a grounded shortlist of SHL Individual Test Solutions through dialogue. Built for the SHL Labs AI Intern take-home.

## Endpoints

- `GET /health` → `{"status":"ok"}`
- `POST /chat` → stateless. Request/response schema exactly as specified in the assignment.

```json
POST /chat
{ "messages": [{"role":"user","content":"Hiring a Java developer who works with stakeholders"}] }

→
{
  "reply": "...",
  "recommendations": [{"name":"...", "url":"https://www.shl.com/...", "test_type":"K"}],
  "end_of_conversation": false
}
```

`recommendations` is empty during clarification/refusal; a 1–10 item array when the agent commits to a shortlist.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Build the catalog (or copy data/catalog.sample.json → data/catalog.json for a smoke test)
python scripts/scrape_catalog.py

# 2. Pick an LLM provider and set its key
cp .env.example .env
# edit .env → set GROQ_API_KEY (or GEMINI_API_KEY / OPENROUTER_API_KEY)

# 3. Run
uvicorn app.main:app --reload --port 8000

# 4. Test
curl http://localhost:8000/health
curl -X POST http://localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hiring a mid-level Java dev, ~45min, stakeholder work"}]}'
```

## Evaluate against traces

Put the SHL-provided trace JSON files into `traces/`, then:

```bash
python scripts/eval_traces.py --traces traces/
```

Outputs per-trace Recall@10 and the mean.

## Tests

```bash
pip install pytest
python -m pytest -q
```

Smoke tests confirm schema compliance and that recommendations always come from the catalog (no hallucinated URLs) — no LLM key required.

## Deployment

- **Render** (recommended free tier): push repo, add `GROQ_API_KEY` env var, Render picks up `render.yaml` + `Dockerfile`. Health check `/health`.
- **Fly.io / Railway / HF Spaces**: use the same `Dockerfile`. Expose port 8000.

## Repo layout

```
app/
  main.py         FastAPI app (GET /health, POST /chat)
  agent.py        Planner-driven conversational agent
  retriever.py    BM25 + TF-IDF hybrid with RRF fusion
  catalog.py      Loads data/catalog.json into typed Assessments
  llm.py          Async LLM client (Groq / Gemini / OpenRouter)
  schemas.py      Pydantic request/response models
scripts/
  scrape_catalog.py   SHL Individual Test Solutions scraper → data/catalog.json
  eval_traces.py      Replay harness (Recall@10) over trace JSON files
tests/
  test_smoke.py       Schema + no-hallucination checks
data/
  catalog.sample.json  Tiny sample so tests + demos work without scraping
Dockerfile
render.yaml
requirements.txt
.env.example
APPROACH.md         2-page write-up for submission
```

See `APPROACH.md` for design rationale, prompt design, and evaluation.
