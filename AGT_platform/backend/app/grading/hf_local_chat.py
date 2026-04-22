"""
Local Hugging Face transformers inference for multimodal structure + grading.

Loads gated Llama 4 checkpoints (e.g. ``meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8``)
when ``MULTIMODAL_LLM_BACKEND=huggingface``. Requires ``pip install -r
requirements-huggingface.txt`` and a hub token with model access (``HF_TOKEN`` /
``HUGGINGFACE_HUB_TOKEN``).

RAG embeddings use :func:`app.grading.rag_embeddings.compute_submission_embedding`
(``RAG_EMBEDDING_BACKEND``, default SentenceTransformers); this module is chat/JSON only.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

from app.config import Config

_log = logging.getLogger(__name__)

_LOAD_LOCK = threading.Lock()
# model_id -> (model, tokenizer_or_processor, kind: "llama4" | "causal_lm")
_STACK_CACHE: dict[str, tuple[Any, Any, str]] = {}


def _hub_token(cfg: Config) -> str | None:
    t = (
        (getattr(cfg, "HUGGINGFACE_HUB_TOKEN", "") or "").strip()
        or (getattr(cfg, "HF_TOKEN", "") or "").strip()
        or (os.getenv("HUGGINGFACE_HUB_TOKEN", "") or "").strip()
        or (os.getenv("HF_TOKEN", "") or "").strip()
    )
    return t or None


def _trust_remote_code(cfg: Config) -> bool:
    if bool(getattr(cfg, "HUGGINGFACE_TRUST_REMOTE_CODE", False)):
        return True
    raw = (os.getenv("HUGGINGFACE_TRUST_REMOTE_CODE", "") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _max_new_tokens(cfg: Config) -> int:
    try:
        n = int(getattr(cfg, "HUGGINGFACE_MAX_NEW_TOKENS", 2048))
    except (TypeError, ValueError):
        n = 2048
    return max(64, min(n, 8192))


def _load_stack(model_id: str, cfg: Config) -> tuple[Any, Any, str]:
    with _LOAD_LOCK:
        if model_id in _STACK_CACHE:
            return _STACK_CACHE[model_id]
        try:
            import torch
        except ImportError as e:
            raise RuntimeError(
                "MULTIMODAL_LLM_BACKEND=huggingface requires PyTorch. "
                "Install: pip install -r requirements-huggingface.txt"
            ) from e

        token = _hub_token(cfg)
        trc = _trust_remote_code(cfg)
        mid_lower = model_id.lower()

        if "llama-4" in mid_lower or "llama4" in mid_lower.replace("_", ""):
            try:
                from transformers import AutoProcessor, Llama4ForConditionalGeneration

                processor = AutoProcessor.from_pretrained(
                    model_id,
                    token=token,
                    trust_remote_code=trc,
                )
                model = Llama4ForConditionalGeneration.from_pretrained(
                    model_id,
                    token=token,
                    trust_remote_code=trc,
                    device_map="auto",
                    torch_dtype="auto",
                )
                model.eval()
                _STACK_CACHE[model_id] = (model, processor, "llama4")
                _log.info("Loaded Llama 4 model from Hugging Face: %s", model_id)
                return _STACK_CACHE[model_id]
            except ImportError as e:
                raise RuntimeError(
                    "transformers version too old for Llama4ForConditionalGeneration. "
                    "Upgrade: pip install -U 'transformers>=4.51'"
                ) from e
            except Exception:
                _log.warning(
                    "Llama4ForConditionalGeneration load failed for %r; trying AutoModelForCausalLM",
                    model_id,
                    exc_info=True,
                )

        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=token,
            trust_remote_code=trc,
            use_fast=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=token,
            trust_remote_code=trc,
            device_map="auto",
            torch_dtype="auto",
        )
        model.eval()
        _STACK_CACHE[model_id] = (model, tokenizer, "causal_lm")
        _log.info("Loaded causal LM from Hugging Face: %s", model_id)
        return _STACK_CACHE[model_id]


def _apply_chat_template(processor: Any, messages: list[dict[str, Any]]) -> str:
    if hasattr(processor, "apply_chat_template"):
        try:
            return processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    tok = getattr(processor, "tokenizer", processor)
    if hasattr(tok, "apply_chat_template"):
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    parts = []
    for m in messages:
        role = str(m.get("role", ""))
        content = str(m.get("content", ""))
        parts.append(f"{role.upper()}:\n{content}\n")
    parts.append("ASSISTANT:\n")
    return "\n".join(parts)


class HuggingFaceJsonChatClient:
    """Implements :class:`app.grading.llm_router.ChatClient` using local ``transformers``."""

    def __init__(self, cfg: Config, model_id: str):
        self._cfg = cfg
        self.model_id = (model_id or "").strip()
        if not self.model_id:
            raise ValueError("HuggingFaceJsonChatClient requires a non-empty model_id")
        self._stack: tuple[Any, Any, str] | None = None

    def _stack_or_load(self) -> tuple[Any, Any, str]:
        if self._stack is None:
            self._stack = _load_stack(self.model_id, self._cfg)
        return self._stack

    def chat_json(
        self, messages: list[dict], *, temperature: float | None = None
    ) -> dict[str, Any]:
        from app.grading.llm_router import parse_llm_json_content

        import torch

        _log.warning(
            "HF chat_json: model_id=%r temperature=%r (first call may load weights)",
            self.model_id,
            temperature,
        )
        model, processor, _kind = self._stack_or_load()
        prompt = _apply_chat_template(processor, messages)
        tokenizer = getattr(processor, "tokenizer", processor)
        inputs = tokenizer(prompt, return_tensors="pt")
        dev = next(model.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()}

        temp = 0.3 if temperature is None else float(temperature)
        gen_kw: dict[str, Any] = {
            "max_new_tokens": _max_new_tokens(self._cfg),
        }
        if temp > 0:
            gen_kw["do_sample"] = True
            gen_kw["temperature"] = min(max(temp, 1e-5), 2.0)
        else:
            gen_kw["do_sample"] = False

        pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(
            tokenizer, "eos_token_id", None
        )
        if pad_id is not None and "attention_mask" in inputs:
            gen_kw["pad_token_id"] = pad_id
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None:
            gen_kw["eos_token_id"] = eos_id

        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kw)

        in_len = inputs["input_ids"].shape[1]
        new_tokens = out[0, in_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return parse_llm_json_content(text)
