"""
LLM routing: local Ollama on the GPU/private tier vs OpenAI (server-side only).

Request handlers and light Flask code must not assume where the model runs — workers
call this layer with Config resolved for their deployment (GPU host has INTERNAL_OLLAMA_URL).
Escalation uses OpenAI when configured and ESCALATE_TO_OPENAI is true.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Protocol

import requests

from ..config import Config

_log = logging.getLogger(__name__)

_THINK_BLOCK = re.compile(
    r"<\s*(?:think|thinking|thought|reasoning)\b[^>]*>.*?</\s*(?:think|thinking|thought|reasoning)\s*>",
    re.IGNORECASE | re.DOTALL,
)


def parse_llm_json_content(content: str) -> dict[str, Any]:
    """
    Parse JSON from chat model output. Handles common small-model issues:

    - Leading / trailing whitespace
    - Common XML-style reasoning blocks stripped before parse
    - `` ```json ... ``` `` fences
    - Preamble or trailing prose (uses first JSON value via ``raw_decode``)
    - Leading garbage before the first ``{`` (retries from first brace)
    - Unicode “smart quotes” normalized
    """
    s = (content or "").strip()
    if not s:
        raise json.JSONDecodeError("Expecting value: empty model content", s, 0)
    s = _THINK_BLOCK.sub("", s).strip()
    s = re.sub(
        r"<\s*redacted\w*\b[^>]*>.*?</\s*redacted\w*\s*>",
        "",
        s,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    s = (
        s.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )

    decoder = json.JSONDecoder()

    def _decode_slice(text: str) -> dict[str, Any]:
        obj, end = decoder.raw_decode(text)
        if not isinstance(obj, dict):
            raise ValueError("model JSON must be an object at top level")
        return obj

    try:
        return _decode_slice(s)
    except json.JSONDecodeError as first_err:
        start = s.find("{")
        if start <= 0:
            _log.debug("LLM JSON parse failed (no brace slice): %s", first_err)
            raise first_err
        try:
            return _decode_slice(s[start:])
        except json.JSONDecodeError:
            _log.debug("LLM JSON parse failed after brace slice: %s", first_err)
            raise first_err


class ChatClient(Protocol):
    def chat_json(
        self, messages: list[dict], *, temperature: float | None = None
    ) -> dict: ...


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        request_json_format: bool = True,
        timeout_sec: float = 300.0,
        keep_alive: str | None = None,
    ):
        self.base_url = (base_url or "").rstrip("/")
        self.model = model
        self._request_json_format = request_json_format
        self._timeout_sec = float(timeout_sec)
        self._keep_alive = keep_alive

    def chat_json(
        self, messages: list[dict], *, temperature: float | None = None
    ) -> dict:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if self._request_json_format:
            body["format"] = "json"
        if self._keep_alive is not None:
            body["keep_alive"] = self._keep_alive
        if temperature is not None:
            body["options"] = {"temperature": float(temperature)}
        r = requests.post(
            f"{self.base_url}/api/chat",
            json=body,
            timeout=self._timeout_sec,
        )
        if r.status_code >= 400:
            snippet = (getattr(r, "text", None) or "")[:1200].strip()
            _log.warning(
                "Ollama /api/chat HTTP %s for model=%r: %s",
                r.status_code,
                self.model,
                snippet or "(empty body)",
            )
        r.raise_for_status()
        try:
            payload = r.json()
        except json.JSONDecodeError:
            _log.warning("Ollama returned non-JSON body for model=%r", self.model)
            raise
        content = (payload.get("message") or {}).get("content") or ""
        return parse_llm_json_content(content)


class OpenAIJsonClient:
    """Server-side OpenAI only; API key never leaves backend workers."""

    def __init__(self, api_key: str, model: str):
        self._api_key = api_key
        self.model = model

    def chat_json(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> dict:
        parsed, _usage = self.chat_json_with_usage(
            messages, temperature=temperature, response_format=response_format
        )
        return parsed

    def chat_json_with_usage(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        """
        Chat Completions → parsed JSON plus token usage (when the API returns it).

        ``response_format`` is passed through when supported; on error it is dropped
        and the request is retried once for broader model compatibility.
        """
        from openai import OpenAI

        client = OpenAI(api_key=self._api_key)
        oa_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
        temp = 0.3 if temperature is None else float(temperature)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": oa_messages,
            "temperature": temp,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception:
            if response_format is not None:
                kwargs.pop("response_format", None)
                resp = client.chat.completions.create(**kwargs)
            else:
                raise
        content = resp.choices[0].message.content or ""
        usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        u = getattr(resp, "usage", None)
        if u is not None:
            usage["prompt_tokens"] = int(getattr(u, "prompt_tokens", 0) or 0)
            usage["completion_tokens"] = int(getattr(u, "completion_tokens", 0) or 0)
            usage["total_tokens"] = int(getattr(u, "total_tokens", 0) or 0)
        return parse_llm_json_content(content), usage


def _ollama_keep_alive(cfg: Config) -> str | None:
    """Return ``keep_alive`` value for Ollama requests.

    When ``OLLAMA_KEEP_ALIVE`` is set (e.g. ``"0s"``), Ollama unloads the model
    immediately after responding, freeing VRAM/RAM for the next model.  Essential
    on memory-constrained machines running multiple grading models sequentially.
    """
    return getattr(cfg, "OLLAMA_KEEP_ALIVE", None) or os.getenv("OLLAMA_KEEP_ALIVE") or None


def primary_ollama_client(cfg: Config) -> OllamaClient:
    base = (cfg.INTERNAL_OLLAMA_URL or cfg.OLLAMA_BASE_URL or "").strip()
    model = (cfg.OLLAMA_MODEL or "llama3.2:3b").strip()
    to = float(getattr(cfg, "OLLAMA_CHAT_TIMEOUT_SEC", 300))
    return OllamaClient(
        base,
        model,
        request_json_format=getattr(cfg, "OLLAMA_CHAT_JSON_FORMAT", True),
        timeout_sec=to,
        keep_alive=_ollama_keep_alive(cfg),
    )


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
    to = float(getattr(cfg, "OLLAMA_CHAT_TIMEOUT_SEC", 300))
    return (
        OllamaClient(
            base,
            model_name,
            request_json_format=getattr(cfg, "OLLAMA_CHAT_JSON_FORMAT", True),
            timeout_sec=to,
            keep_alive=_ollama_keep_alive(cfg),
        ),
        f"ollama:{model_name}",
    )


def build_grading_clients(cfg: Config) -> list[tuple[ChatClient, str]]:
    """
    Return 1–3 (client, model_label) pairs for multi-LLM grading.

    Always includes the primary Ollama model (``OLLAMA_MODEL``). Optional extras
    come from non-empty ``GRADING_MODEL_2`` / ``GRADING_MODEL_3``. When those are
    unset, multimodal grading uses a single model and relies on
    ``MULTIMODAL_SAMPLES_PER_MODEL`` for semantic entropy and consensus.
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


_DEFAULT_HF_MAVERICK_FP8 = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"

_LLAMA_MODEL_DESCRIPTOR_TO_HF: dict[str, str] = {
    "llama-4-maverick-17b-128e-instruct:fp8": _DEFAULT_HF_MAVERICK_FP8,
    "Llama-4-Maverick-17B-128E-Instruct:fp8": _DEFAULT_HF_MAVERICK_FP8,
}


def _normalize_hf_grading_model_id(raw: str) -> str:
    key = (raw or "").strip()
    if not key:
        return ""
    if key in _LLAMA_MODEL_DESCRIPTOR_TO_HF:
        return _LLAMA_MODEL_DESCRIPTOR_TO_HF[key]
    lk = key.lower()
    if lk in _LLAMA_MODEL_DESCRIPTOR_TO_HF:
        return _LLAMA_MODEL_DESCRIPTOR_TO_HF[lk]
    return key


def multimodal_llm_backend_uses_huggingface(cfg: Config) -> bool:
    b = (getattr(cfg, "MULTIMODAL_LLM_BACKEND", "") or "").strip().lower()
    return b in ("huggingface", "hf")


def multimodal_llm_backend_uses_openai(cfg: Config) -> bool:
    b = (getattr(cfg, "MULTIMODAL_LLM_BACKEND", "") or "").strip().lower()
    return b == "openai"


def openai_multimodal_grading_model(cfg: Config) -> str:
    """Chat model id for multimodal grading / structure when backend is ``openai``."""
    m = (getattr(cfg, "OPENAI_MULTIMODAL_GRADING_MODEL", "") or "").strip()
    if m:
        return m
    m2 = (getattr(cfg, "OPENAI_TRIO_RAG_CHAT_MODEL", "") or "").strip()
    if m2:
        return m2
    return "gpt-5.4-nano"


def huggingface_grading_model_id(cfg: Config) -> str:
    """HF repo id for multimodal structure + grading when backend is Hugging Face."""
    rid = (getattr(cfg, "HUGGINGFACE_GRADING_MODEL_ID", "") or "").strip()
    if rid:
        return _normalize_hf_grading_model_id(rid)
    return _DEFAULT_HF_MAVERICK_FP8


def huggingface_json_client_from_config(
    cfg: Config, model_id: str | None = None
) -> ChatClient:
    """Local ``transformers`` chat client (lazy-load weights on first ``chat_json``)."""
    from .hf_local_chat import HuggingFaceJsonChatClient

    mid = _normalize_hf_grading_model_id((model_id or huggingface_grading_model_id(cfg)).strip())
    if not mid:
        raise ValueError(
            "HUGGINGFACE_GRADING_MODEL_ID is empty; set it or use the default Maverick FP8 repo."
        )
    return HuggingFaceJsonChatClient(cfg, mid)


def build_multimodal_grading_clients(cfg: Config) -> list[tuple[ChatClient, str]]:
    """
    Multimodal chunk grading: primary from ``MULTIMODAL_LLM_BACKEND`` (``openai``, Hugging Face,
    or Ollama), plus optional ``GRADING_MODEL_2`` / ``GRADING_MODEL_3`` (``ollama:`` / ``openai:``).
    """
    if multimodal_llm_backend_uses_huggingface(cfg):
        mid = huggingface_grading_model_id(cfg)
        primary = huggingface_json_client_from_config(cfg, mid)
        clients: list[tuple[ChatClient, str]] = [(primary, f"huggingface:{mid}")]
    elif multimodal_llm_backend_uses_openai(cfg):
        key = (cfg.OPENAI_API_KEY or "").strip()
        omid = openai_multimodal_grading_model(cfg)
        if key:
            clients = [(OpenAIJsonClient(key, omid), f"openai:{omid}")]
        else:
            _log.warning(
                "MULTIMODAL_LLM_BACKEND=openai but OPENAI_API_KEY is empty; "
                "falling back to primary Ollama client"
            )
            primary = primary_ollama_client(cfg)
            oll = (cfg.OLLAMA_MODEL or "llama3.2:3b").strip()
            clients = [(primary, f"ollama:{oll}")]
    else:
        primary = primary_ollama_client(cfg)
        om = (cfg.OLLAMA_MODEL or "llama3.2:3b").strip()
        clients = [(primary, f"ollama:{om}")]

    for spec in (cfg.GRADING_MODEL_2, cfg.GRADING_MODEL_3):
        parsed = _parse_model_spec(spec, cfg)
        if parsed:
            clients.append(parsed)

    return clients


def multimodal_structure_llm_trace_label(cfg: Config) -> str:
    """Human-readable model id for logs / ``_agentic_workflow`` (trio / QA segment)."""
    if multimodal_llm_backend_uses_huggingface(cfg):
        return huggingface_grading_model_id(cfg)
    if multimodal_llm_backend_uses_openai(cfg):
        return openai_multimodal_grading_model(cfg)
    m = (getattr(cfg, "MULTIMODAL_TRIO_CHUNKING_MODEL", "") or "").strip()
    if m:
        return m
    om = (getattr(cfg, "OLLAMA_MODEL", "") or "").strip()
    return om or "(structure_llm)"


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
            ],
            temperature=None,
        )
        out["_used_openai_arbitration"] = True
        return out
    except Exception:
        return local_result
