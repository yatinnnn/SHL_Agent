"""Smoke tests that do NOT require an LLM key.
Run with: python -m pytest -q  (after: pip install pytest)
"""
from __future__ import annotations

import json
import os

from fastapi.testclient import TestClient

# ensure a tiny catalog exists for tests
os.environ.setdefault(
    "SHL_CATALOG_PATH",
    os.path.join(os.path.dirname(__file__), "fixture_catalog.json"),
)

FIXTURE = [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/x/java-8-new/", "test_type": "K",
     "description": "Multi-choice assessment for Java 8 fundamentals."},
    {"name": "OPQ32r", "url": "https://www.shl.com/x/opq32r/", "test_type": "P",
     "description": "Occupational personality questionnaire for workplace behavior."},
    {"name": "Verify - Numerical Reasoning", "url": "https://www.shl.com/x/verify-num/", "test_type": "A",
     "description": "Cognitive numerical reasoning ability test."},
]

with open(os.environ["SHL_CATALOG_PATH"], "w") as f:
    json.dump(FIXTURE, f)

from app.main import app  # noqa: E402

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_chat_schema_no_key():
    # Without LLM keys, agent falls back but MUST return valid schema.
    r = client.post("/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"reply", "recommendations", "end_of_conversation"}
    assert isinstance(body["reply"], str) and body["reply"]
    assert isinstance(body["recommendations"], list)
    assert isinstance(body["end_of_conversation"], bool)


def test_recommendations_are_from_catalog():
    r = client.post("/chat", json={
        "messages": [{"role": "user", "content": "Hiring a mid level Java developer with stakeholder work, ~45 min budget"}]
    })
    body = r.json()
    urls = {a["url"] for a in FIXTURE}
    for rec in body["recommendations"]:
        assert rec["url"] in urls, f"hallucinated URL: {rec['url']}"
