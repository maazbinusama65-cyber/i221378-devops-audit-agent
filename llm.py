"""Thin LLM client over the basic provider SDKs (no agent frameworks).

The agent only needs one operation: ``complete(messages) -> str``. Two backends
cover every common provider:

* ``anthropic``: the Anthropic SDK.
* anything OpenAI-compatible (OpenAI, Groq, Gemini, OpenRouter, Ollama, ...): the
  OpenAI SDK pointed at a ``base_url``.
"""

from __future__ import annotations

import os
import random
import re
import sys
import time
from dataclasses import dataclass
from typing import Any

from redact import redact, register_secret

Message = dict[str, str]


class LLMError(RuntimeError):
    """The model could not be reached or kept failing after retries."""


class ContextTooLarge(LLMError):
    """The request exceeded the model's context or tokens-per-minute budget; shrink and retry."""


@dataclass(frozen=True)
class Preset:
    key_env: str
    base_url: str | None
    model: str


PRESETS: dict[str, Preset] = {
    "groq": Preset("GROQ_API_KEY", "https://api.groq.com/openai/v1", "openai/gpt-oss-120b"),
    "openai": Preset("OPENAI_API_KEY", None, "gpt-4o-mini"),
    "anthropic": Preset("ANTHROPIC_API_KEY", None, "claude-sonnet-4-5"),
    "gemini": Preset("GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai/",
                     "gemini-2.0-flash"),
    "openrouter": Preset("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1", "openai/gpt-4o-mini"),
}
AUTODETECT_ORDER = ("groq", "openai", "anthropic", "gemini", "openrouter")


def resolve_config(env: dict[str, str] | None = None) -> dict[str, Any]:
    """Work out provider/model/key/base_url from environment variables.

    Precedence: explicit ``LLM_*`` variables, then the provider preset, then
    autodetection from whichever provider key is present.
    """
    env = dict(os.environ if env is None else env)
    provider = (env.get("LLM_PROVIDER") or "").strip().lower()
    if not provider:
        for candidate in AUTODETECT_ORDER:
            if env.get(PRESETS[candidate].key_env):
                provider = candidate
                break
        else:
            provider = "openai" if env.get("LLM_API_KEY") or env.get("LLM_BASE_URL") else ""
    if not provider:
        raise LLMError(
            "No LLM credentials found. Set LLM_PROVIDER plus the matching API key "
            "(GROQ_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, ...), or LLM_API_KEY with LLM_BASE_URL. "
            "See .env.example.")
    preset = PRESETS.get(provider, Preset("LLM_API_KEY", None, ""))
    api_key = env.get("LLM_API_KEY") or env.get(preset.key_env, "")
    base_url = env.get("LLM_BASE_URL") or preset.base_url
    model = env.get("LLM_MODEL") or preset.model
    if not model:
        raise LLMError(f"LLM_MODEL must be set for provider {provider!r}")
    if not api_key and not (base_url and ("localhost" in base_url or "127.0.0.1" in base_url)):
        raise LLMError(f"API key for provider {provider!r} not set (expected {preset.key_env} or LLM_API_KEY)")
    return {"provider": provider, "api_key": api_key or "not-needed", "base_url": base_url, "model": model}


class LLMClient:
    def __init__(self, provider: str, api_key: str, model: str, base_url: str | None = None,
                 temperature: float = 0.0, timeout: float = 90.0, max_retries: int = 6) -> None:
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.calls = 0
        self.output_cap: int | None = None
        register_secret(api_key)
        if provider == "anthropic":
            import anthropic  # imported lazily so the other backend needs no extra package

            self._client: Any = anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=0)
            self._retryable: tuple[type[BaseException], ...] = (
                anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError)
        else:
            import openai

            self._client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)
            self._retryable = (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "LLMClient":
        cfg = resolve_config(env)
        return cls(cfg["provider"], cfg["api_key"], cfg["model"], cfg["base_url"])

    # ------------------------------------------------------------------ public
    def complete(self, messages: list[Message], max_tokens: int = 1600) -> str:
        """Send a chat transcript (first item is the system prompt); return the reply text."""
        last_error: Exception | None = None
        json_mode = True
        if self.output_cap:
            max_tokens = min(max_tokens, self.output_cap)
        for attempt in range(self.max_retries + 1):
            try:
                self.calls += 1
                return self._once(messages, max_tokens, json_mode)
            except self._retryable as exc:
                last_error = exc
                lowered = str(exc).lower()
                if "output tokens" in lowered and "request too large" in lowered and max_tokens > 400:
                    limit = re.search(r"limit (\d+)", lowered)
                    max_tokens = max(300, int(limit.group(1)) // 2 if limit else max_tokens // 2)
                    self.output_cap = max_tokens  # remembered: later calls stay under the output limit
                    print(f"    LLM output-token limit hit; lowering max_tokens to {max_tokens}", file=sys.stderr)
                    continue
                if "request too large" in lowered or "reduce your message size" in lowered:
                    raise ContextTooLarge(f"request too large for the model: {redact(str(exc))[:200]}") from exc
                if "per day" in lowered or "(tpd)" in lowered or "daily" in lowered or "insufficient_quota" in lowered:
                    raise LLMError(f"LLM quota exhausted (not retryable): {redact(str(exc))[:240]}") from exc
                delay = self._backoff(exc, attempt)
                print(f"    LLM {type(exc).__name__}; retrying in {delay:.0f}s "
                      f"(attempt {attempt + 1}/{self.max_retries + 1}): {redact(str(exc))[:160]}", file=sys.stderr)
                time.sleep(delay)
            except Exception as exc:  # noqa: BLE001 - classify below
                status = getattr(exc, "status_code", None)
                text = str(exc).lower()
                if status == 413 or "context length" in text or "context_length" in text \
                        or "request too large" in text or "reduce your message size" in text:
                    raise ContextTooLarge(f"request too large for the model: {str(exc)[:200]}") from exc
                if status == 400 and json_mode and self.provider != "anthropic":
                    json_mode = False  # model/provider rejects response_format; retry without it
                    last_error = exc
                    continue
                raise LLMError(f"LLM request failed: {type(exc).__name__}: {str(exc)[:300]}") from exc
        raise LLMError(f"LLM unavailable after {self.max_retries + 1} attempts: "
                       f"{type(last_error).__name__}: {str(last_error)[:200]}")

    # ----------------------------------------------------------------- internal
    def _once(self, messages: list[Message], max_tokens: int, json_mode: bool) -> str:
        if self.provider == "anthropic":
            system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
            chat = [m for m in messages if m["role"] != "system"]
            response = self._client.messages.create(
                model=self.model, system=system, messages=chat, max_tokens=max_tokens,
                temperature=self.temperature)
            return "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages,
                                  "temperature": self.temperature, "max_tokens": max_tokens}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    @staticmethod
    def _backoff(exc: BaseException, attempt: int) -> float:
        headers = getattr(getattr(exc, "response", None), "headers", None) or {}
        retry_after = headers.get("retry-after") if hasattr(headers, "get") else None
        if retry_after:
            try:
                return min(float(retry_after) + 0.5, 60.0)
            except ValueError:
                pass
        message = str(exc)
        match = re.search(r"try again in (\d+(?:\.\d+)?)(ms|s|m)", message)
        if match:
            value = float(match.group(1)) * {"ms": 0.001, "s": 1, "m": 60}[match.group(2)]
            return min(value + 1.0, 60.0)
        return min(2.0 ** attempt + random.random(), 30.0)
