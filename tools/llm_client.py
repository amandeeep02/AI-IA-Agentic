"""
llm_client.py — Unified LLM gateway for the AI code assistant.

Features:
  - Model registry with per-role defaults
  - Retry with exponential backoff on transient errors
  - Token usage tracking
  - Structured LLMResponse return type (no silent string returns)
  - Conversation (multi-turn) support
  - Lazy client init — fails fast at call time, not import time
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

load_dotenv()
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────

class Role:
    PLANNER   = "planner"
    CODEGEN   = "codegen"
    TESTGEN   = "testgen"
    ANALYZER  = "analyzer"
    CRITIC    = "critic"
    COMMIT    = "commit"
    DEFAULT   = "default"


_MODEL_MAP: dict[str, str] = {
    Role.PLANNER:  "gpt-4o-mini",
    Role.CODEGEN:  "gpt-4o",        # codegen benefits from the stronger model
    Role.TESTGEN:  "gpt-4o-mini",
    Role.ANALYZER: "gpt-4o",        # debugging needs deeper reasoning
    Role.CRITIC:   "gpt-4o-mini",
    Role.COMMIT:   "gpt-4o-mini",
    Role.DEFAULT:  "gpt-4o-mini",
}

_RETRY_DELAYS = (1.0, 3.0, 7.0)   # seconds between attempts


# ─────────────────────────────────────────────────────────────
# Return type
# ─────────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    content: str
    model: str
    role: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    attempts: int = 1
    error: str | None = None

    @property
    def success(self) -> bool:
        return bool(self.content) and self.error is None


# ─────────────────────────────────────────────────────────────
# Client (lazy singleton)
# ─────────────────────────────────────────────────────────────

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Add it to your .env file or environment."
            )
        _client = OpenAI(api_key=api_key)
        logger.debug("OpenAI client initialised.")
    return _client


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def call_llm(
    system_prompt: str,
    user_prompt: str,
    *,
    role: str = Role.DEFAULT,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    json_mode: bool = True,
    conversation_history: list[dict[str, str]] | None = None,
    model_override: str | None = None,
) -> LLMResponse:
    """
    Call the OpenAI chat completion API with retry logic.

    Args:
        system_prompt:        The system instruction.
        user_prompt:          The user message.
        role:                 One of Role.* — selects the default model.
        temperature:          Sampling temperature.
        max_tokens:           Maximum completion tokens.
        json_mode:            If True, requests JSON output mode.
        conversation_history: Optional prior turns for multi-turn flows.
                              Each dict must have "role" and "content" keys.
        model_override:       Explicitly override the model for this call.

    Returns:
        LLMResponse with .content, .success, token counts, and attempt count.
    """
    if not system_prompt or not user_prompt:
        raise ValueError("system_prompt and user_prompt must not be empty.")

    model = model_override or _MODEL_MAP.get(role, _MODEL_MAP[Role.DEFAULT])

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_prompt})

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    last_error = ""
    for attempt in range(1, len(_RETRY_DELAYS) + 2):
        try:
            logger.debug("LLM call | role=%s model=%s attempt=%d", role, model, attempt)
            response = _get_client().chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            usage = response.usage

            result = LLMResponse(
                content=content,
                model=model,
                role=role,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
                attempts=attempt,
            )
            logger.info(
                "LLM OK | role=%s model=%s tokens=%d attempt=%d",
                role, model, result.total_tokens, attempt,
            )
            return result

        except RateLimitError as exc:
            last_error = f"Rate limit: {exc}"
            logger.warning("Rate limit on attempt %d — backing off.", attempt)
        except APIConnectionError as exc:
            last_error = f"Connection error: {exc}"
            logger.warning("Connection error on attempt %d.", attempt)
        except APIStatusError as exc:
            last_error = f"API status {exc.status_code}: {exc.message}"
            # 4xx errors (except 429) are not retryable
            if exc.status_code != 429:
                logger.error("Non-retryable API error: %s", last_error)
                break
            logger.warning("429 on attempt %d — backing off.", attempt)

        if attempt <= len(_RETRY_DELAYS):
            time.sleep(_RETRY_DELAYS[attempt - 1])

    logger.error("LLM call failed after %d attempt(s): %s", attempt, last_error)
    return LLMResponse(
        content="",
        model=model,
        role=role,
        attempts=attempt,
        error=last_error,
    )