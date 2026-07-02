"""Thin async LLM client. Supports Groq, Gemini, OpenRouter via env.

We keep this small on purpose: one function, JSON-mode where available, retries
on transient errors, hard timeout well under the 30s per-call cap.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")

DEFAULT_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "20"))


class LLMError(RuntimeError):
    pass


async def _post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        r = await client.post(url, headers=headers, json=payload)
        if r.status_code >= 400:
            raise LLMError(f"{r.status_code} {r.text[:300]}")
        return r.json()


async def _groq(messages: List[Dict[str, str]], json_mode: bool, temperature: float) -> str:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise LLMError("GROQ_API_KEY not set")
    payload: Dict[str, Any] = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 800,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    data = await _post_json(
        "https://api.groq.com/openai/v1/chat/completions",
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        payload,
    )
    return data["choices"][0]["message"]["content"]


async def _openrouter(messages: List[Dict[str, str]], json_mode: bool, temperature: float) -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise LLMError("OPENROUTER_API_KEY not set")
    payload: Dict[str, Any] = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 800,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    data = await _post_json(
        "https://openrouter.ai/api/v1/chat/completions",
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        payload,
    )
    return data["choices"][0]["message"]["content"]


async def _gemini(messages: List[Dict[str, str]], json_mode: bool, temperature: float) -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise LLMError("GEMINI_API_KEY not set")
    # Convert OpenAI-style messages to Gemini format
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    contents = []
    for m in messages:
        if m["role"] == "system":
            continue
        contents.append(
            {"role": "user" if m["role"] == "user" else "model",
             "parts": [{"text": m["content"]}]}
        )
    payload: Dict[str, Any] = {
        "contents": contents,
        "generationConfig": {"temperature": temperature, "maxOutputTokens": 800},
    }
    if system_parts:
        payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
    if json_mode:
        payload["generationConfig"]["responseMimeType"] = "application/json"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={key}"
    data = await _post_json(url, {"Content-Type": "application/json"}, payload)
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        raise LLMError(f"gemini bad response: {data}") from e


async def complete(
    messages: List[Dict[str, str]],
    *,
    json_mode: bool = False,
    temperature: float = 0.2,
) -> str:
    fn = {"groq": _groq, "gemini": _gemini, "openrouter": _openrouter}.get(PROVIDER)
    if fn is None:
        raise LLMError(f"unknown LLM_PROVIDER={PROVIDER}")

    last_err: Optional[Exception] = None
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.TransportError, LLMError)),
        reraise=False,
    ):
        with attempt:
            try:
                return await fn(messages, json_mode=json_mode, temperature=temperature)
            except Exception as e:  # noqa: BLE001
                last_err = e
                raise
    raise LLMError(f"LLM failed after retries: {last_err}")


# async def complete_json(messages: List[Dict[str, str]], temperature: float = 0.1) -> Dict[str, Any]:
#     raw = await complete(messages, json_mode=True, temperature=temperature)
#     try:
#         return json.loads(raw)
#     except json.JSONDecodeError:
#         # last-resort recovery: extract {...} substring
#         start = raw.find("{")
#         end = raw.rfind("}")
#         if start != -1 and end != -1:
#             return json.loads(raw[start : end + 1])
#         raise
async def complete_json(messages: List[Dict[str, str]], temperature: float = 0.1) -> Dict[str, Any]:
    raw = await complete(messages, json_mode=True, temperature=temperature)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            return json.loads(raw[start:end + 1])
        raise