"""Replay-style evaluator.

Reads traces from `traces/` (JSON files matching the SHL-provided format).
Each trace is expected to look like:

  {
    "persona": "Hiring manager for ...",
    "facts": { "role": "...", "seniority": "...", ... },
    "expected_shortlist": ["Java 8 (New)", "OPQ32r", ...]
  }

The evaluator drives our agent by simulating the user turn-by-turn using the
same LLM the agent uses (persona-conditioned). It stops when the agent
returns end_of_conversation=true OR after MAX_TURNS.

Metric: mean Recall@10 over traces, where relevance is name-matching against
expected_shortlist (case-insensitive, whitespace-normalized).

Usage:
    python scripts/eval_traces.py --traces traces/
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import re
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agent import run_agent
from app.llm import complete
from app.schemas import Message

MAX_TURNS = 8


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


async def simulated_user_turn(persona: str, facts: Dict[str, Any], history: List[Message]) -> str:
    sys_p = (
        "You are simulating a hiring manager talking to an assessment recommender. "
        "Answer the assistant's LAST question truthfully using ONLY the FACTS given. "
        "If asked about something not in FACTS, say you have no preference. "
        "Keep responses under 2 sentences. If the assistant has already given a shortlist, say 'thanks, that works'."
    )
    convo = "\n".join(f"{m.role.upper()}: {m.content}" for m in history)
    user = f"PERSONA: {persona}\nFACTS: {json.dumps(facts)}\n\nCONVERSATION SO FAR:\n{convo}\n\nYour next USER message:"
    return (await complete([{"role": "system", "content": sys_p}, {"role": "user", "content": user}], temperature=0.3)).strip()


def recall_at_k(expected: List[str], recs: List[Dict[str, str]], k: int = 10) -> float:
    if not expected:
        return 0.0
    exp = {norm(x) for x in expected}
    got = {norm(r["name"]) for r in recs[:k]}
    hit = len(exp & got)
    return hit / len(exp)


async def run_trace(trace: Dict[str, Any]) -> Dict[str, Any]:
    persona = trace.get("persona") or "Hiring manager"
    facts = trace.get("facts") or {}
    expected = trace.get("expected_shortlist") or trace.get("expected") or []

    # Seed with an opening user message derived from persona (or first msg field).
    opener = trace.get("opening") or facts.get("opening") or "I need help picking an assessment."
    messages: List[Message] = [Message(role="user", content=opener)]

    final_recs: List[Dict[str, str]] = []
    for turn in range(MAX_TURNS):
        resp = await run_agent(messages)
        messages.append(Message(role="assistant", content=resp.reply))
        if resp.recommendations:
            final_recs = [r.model_dump() for r in resp.recommendations]
        if resp.end_of_conversation:
            break
        if turn == MAX_TURNS - 1:
            break
        try:
            next_user = await simulated_user_turn(persona, facts, messages)
        except Exception as e:  # noqa: BLE001
            next_user = "thanks, that works"
        messages.append(Message(role="user", content=next_user))
        if "that works" in next_user.lower() or "thanks" in next_user.lower():
            # give the agent one more turn to finalize if it hasn't
            if not final_recs:
                continue
            break

    return {
        "persona": persona,
        "expected": expected,
        "final_recs": [r["name"] for r in final_recs],
        "recall_at_10": recall_at_k(expected, final_recs, 10),
    }


async def main_async(paths: List[str]) -> None:
    results = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            trace = json.load(f)
        print(f"--- {os.path.basename(p)} ---")
        r = await run_trace(trace)
        r["file"] = os.path.basename(p)
        print(f"  recall@10 = {r['recall_at_10']:.3f}")
        print(f"  expected: {r['expected']}")
        print(f"  got:      {r['final_recs']}")
        results.append(r)
    mean = sum(r["recall_at_10"] for r in results) / max(len(results), 1)
    print("=" * 60)
    print(f"Mean Recall@10 across {len(results)} traces: {mean:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="traces", help="directory of trace JSON files")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.traces, "*.json")))
    if not files:
        print(f"No traces found in {args.traces}")
        return
    asyncio.run(main_async(files))


if __name__ == "__main__":
    main()
