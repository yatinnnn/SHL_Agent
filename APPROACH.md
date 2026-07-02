# Approach — SHL Conversational Assessment Recommender

## Problem framing

Recruiters describe roles in fuzzy natural language; the SHL catalog is
navigated by product name and taxonomy. The agent has to bridge that gap
across a **multi-turn** conversation, while staying grounded in the
Individual Test Solutions catalog and never fabricating products or URLs.

## Design choices

**1. Stateless FastAPI, single planner LLM call per turn.**
Every `POST /chat` rebuilds context from the full `messages` history and issues
one JSON-mode planner call. The planner picks an action —
`CLARIFY | RECOMMEND | COMPARE | REFUSE | END` — and, for `RECOMMEND`, returns
**indices into a pre-retrieved candidate list**, not free text. This makes URL
fabrication structurally impossible.

**2. Hybrid retrieval (BM25 + TF-IDF, RRF fusion).**
Chose lexical + n-gram TF-IDF over dense embeddings because (a) the catalog is
small (~500 products) so recall is easy without a vector DB, (b) queries mix
proper product names (“OPQ”, “GSA”, “Verify”) with paraphrased skills, and
BM25 crushes vector search on named-entity lookups, (c) it keeps cold-start
under a second on free tiers with no model download. Reciprocal Rank Fusion
handles both regimes.

**3. Grounded planner prompt.**
The planner sees the retrieved candidates block with `test_type`, `duration`,
short description, and URL. The system prompt forbids naming items outside
that block. We validate returned `picked_indices` against the block before
constructing `Recommendation` objects — indices out of range are dropped.

**4. Hard rules that override the LLM.**
- Turn-1 vague queries force `CLARIFY` even if the LLM tries to recommend.
  This directly targets the “agent does not recommend on turn 1 for a vague
  query” behavior probe.
- `recommendations` is empty for any action other than `RECOMMEND`.
- LLM failure / missing key falls back to a deterministic top-5 retrieval
  response so the schema is always valid.

**5. Test type is a first-class field.**
We normalize SHL’s letter codes (A/B/C/D/E/K/P/S) at ingestion and expose the
full name in the retriever’s searchable text, so queries like “add personality
tests” route to `P` items during refinement.

## Prompt design

Two system messages: a static behavior charter (scope, refusal rules,
concision) and a planner instruction that specifies the JSON schema and each
action’s constraints. The user message packs `CONVERSATION`, `CANDIDATES`, and
`TURN_NUMBER / FORCE_CLARIFY` flags. History is truncated to the last 10 turns
to fit within the 8-turn cap plus buffer for latency.

## Evaluation

`scripts/eval_traces.py` simulates the SHL replay harness: an LLM plays the
persona and answers questions truthfully from a `facts` object, saying “no
preference” for unknown fields, and terminating on shortlist. We compute
per-trace **Recall@10** (name-normalized set intersection over expected
shortlist) and the mean. Smoke tests in `tests/test_smoke.py` assert schema
compliance and no hallucinated URLs — these run with **no LLM key** using the
deterministic fallback path, catching regressions in CI.

## What didn't work

- **Dense embeddings only** (sentence-transformers MiniLM): shipped a 90MB
  model that blew Render’s free-tier cold start past 60s and *hurt* recall on
  product-name queries like “OPQ”, “ADEPT-15”. Kept it out.
- **Function-calling loop** (LLM proposes retrieval queries then answers):
  added 2× latency without changing top-5 hits vs. concatenating all user
  turns into a single query. Removed.
- **Letting the LLM emit `recommendations` as free JSON**: it occasionally
  paraphrased names (“OPQ 32r Personality”), which broke exact-match recall
  against expected shortlists. Switching to `picked_indices` fixed it.

## Stack justification

FastAPI + Pydantic v2 (schema is enforced at both request and response
boundaries), scikit-learn + rank_bm25 (zero-download retrieval), httpx +
tenacity for the LLM client (async, retried, hard 20s timeout well under the
30s per-call cap). Groq (Llama 3.3 70B) as the default provider for JSON-mode
speed on the free tier; Gemini and OpenRouter supported via env swap.

## AI tools used

Cursor / Claude were used to draft boilerplate (scraper skeleton, pydantic
models, Dockerfile). All prompt logic, planner design, retrieval fusion,
force-clarify rule, and evaluation harness were designed by hand and can be
walked through line-by-line.
