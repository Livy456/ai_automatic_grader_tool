"""
LLM routing: local Ollama on the GPU/private tier vs OpenAI (server-side only).

Request handlers and light Flask code must not assume where the model runs — workers
call this layer with Config resolved for their deployment (GPU host has INTERNAL_OLLAMA_URL).
Escalation uses OpenAI when configured and ESCALATE_TO_OPENAI is true.
"""
from __future__ import annotations

import json
from typing import Any, Protocol

import requests

from ..config import Config


class ChatClient(Protocol):
    def chat_json(self, messages: list[dict]) -> dict: ...


class OllamaClient:
    def __init__(self, base_url: str, model: str):
        self.base_url = (base_url or "").rstrip("/")
        self.model = model

    def chat_json(self, messages: list[dict]) -> dict:
        r = requests.post(
            f"{self.base_url}/api/chat",
            json={"model": self.model, "messages": messages, "stream": False},
            timeout=300,
        )
        r.raise_for_status()
        content = r.json()["message"]["content"]
        return json.loads(content)


class OpenAIJsonClient:
    """Server-side OpenAI only; API key never leaves backend workers."""

    def __init__(self, api_key: str, model: str):
        self._api_key = api_key
        self.model = model

    def chat_json(self, messages: list[dict]) -> dict:
        from openai import OpenAI

        client = OpenAI(api_key=self._api_key)
        # Map simple role/content messages to Chat Completions
        oa_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
        resp = client.chat.completions.create(
            model=self.model,
            messages=oa_messages,
            temperature=0.2,
        )
        content = (resp.choices[0].message.content or "").strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return json.loads(content)


def primary_ollama_client(cfg: Config) -> OllamaClient:
    base = (cfg.INTERNAL_OLLAMA_URL or cfg.OLLAMA_BASE_URL or "").strip()
    model = (cfg.OLLAMA_MODEL or "llama3.2:3b").strip()
    return OllamaClient(base, model)


def openai_client_if_configured(cfg: Config) -> OpenAIJsonClient | None:
    key = (cfg.OPENAI_API_KEY or "").strip()
    if not key:
        return None
    return OpenAIJsonClient(key, cfg.OPENAI_MODEL)


def _parse_model_spec(spec: str, cfg: Config) -> tuple[ChatClient, str] | None:
    """Parse a 'provider:model' spec into (client, label). Returns None if invalid/empty."""
    if not spec:
        return None
    if spec.startswith("openai:"):
        model_name = spec[len("openai:") :].strip()
        key = (cfg.OPENAI_API_KEY or "").strip()
        if not key or not model_name:
            return None
        return OpenAIJsonClient(key, model_name), f"openai:{model_name}"
    if spec.startswith("ollama:"):
        model_name = spec[len("ollama:") :].strip()
    else:
        # Bare model name defaults to Ollama
        model_name = spec.strip()
    if not model_name:
        return None
    base = (cfg.INTERNAL_OLLAMA_URL or cfg.OLLAMA_BASE_URL or "").strip()
    if not base:
        return None
    return OllamaClient(base, model_name), f"ollama:{model_name}"


def build_grading_clients(cfg: Config) -> list[tuple[ChatClient, str]]:
    """
    Return 1–3 (client, model_label) pairs for multi-LLM grading.
    Always includes the primary Ollama model. Slots 2 and 3 come from
    GRADING_MODEL_2 / GRADING_MODEL_3 env vars.
    """
    primary = primary_ollama_client(cfg)
    om = (cfg.OLLAMA_MODEL or "llama3.2:3b").strip()
    primary_label = f"ollama:{om}"
    clients: list[tuple[ChatClient, str]] = [(primary, primary_label)]

    for spec in (cfg.GRADING_MODEL_2, cfg.GRADING_MODEL_3):
        parsed = _parse_model_spec(spec, cfg)
        if parsed:
            clients.append(parsed)

    return clients


def maybe_escalate_grade(
    cfg: Config,
    primary: ChatClient,
    secondary: OpenAIJsonClient | None,
    rubric: list,
    assignment_prompt: str,
    submission_context: dict,
    local_result: dict,
) -> dict:
    """
    If escalation is enabled and OpenAI is configured, optionally re-run grading
    when local criteria look weak (simple heuristic — tune in prompts later).
    """
    if not secondary or not cfg.ESCALATE_TO_OPENAI:
        return local_result
    criteria = local_result.get("criteria") or []
    low = any(float(c.get("confidence", 1.0)) < 0.72 for c in criteria)
    if not low and "needs_review" not in (local_result.get("flags") or []):
        return local_result
    from .prompts import SYSTEM, GRADER

    payload = {
        "rubric": rubric,
        "assignment_prompt": assignment_prompt,
        "submission": submission_context,
        "prior_local_result": local_result,
    }
    try:
        out = secondary.chat_json(
            [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": GRADER
                    + "\nArbitrate using prior local result; return final JSON only.\n"
                    + json.dumps(payload),
                },
            ]
        )
        out["_used_openai_arbitration"] = True
        return out
    except Exception:
        return local_result
