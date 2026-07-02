from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


Role = Literal["user", "assistant", "system"]


class Message(BaseModel):
    role: Role
    content: str

    @field_validator("content")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v if isinstance(v, str) else str(v)


class ChatRequest(BaseModel):
    messages: List[Message] = Field(default_factory=list)


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str  # single-letter SHL code: A,B,C,D,E,K,P,S


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False


class HealthResponse(BaseModel):
    status: str = "ok"
