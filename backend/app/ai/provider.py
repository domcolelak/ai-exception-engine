"""Swappable AI provider abstraction.

Design rules enforced here:

* every call declares a Pydantic output model, so the rest of the code never
  parses free-form text;
* every call records model name, prompt version, latency and token usage;
* a failing provider degrades to ``None`` instead of breaking the request --
  the deterministic analytics are always the source of truth;
* an offline provider makes tests and demos work with no network and no keys.
"""
from __future__ import annotations

from types import UnionType

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar, Union, get_args, get_origin

import httpx
from pydantic import BaseModel, ValidationError

from app.core.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PHONE_RE = re.compile(r"(?<!\d)(\+?\d[\d\s\-()]{7,}\d)(?!\d)")
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")


def redact(value: Any) -> Any:
    """Strip obvious direct identifiers before anything leaves the process."""
    if isinstance(value, str):
        value = _EMAIL_RE.sub("[email]", value)
        value = _IBAN_RE.sub("[iban]", value)
        value = _PHONE_RE.sub("[phone]", value)
        return value
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


@dataclass
class AICallResult:
    """Envelope returned by every provider call."""

    output: BaseModel | None
    model: str
    prompt_version: str
    latency_ms: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.output is not None

    def as_log_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "prompt_version": self.prompt_version,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "error": self.error,
        }


class AIProvider(ABC):
    """Interface every provider implements."""

    name: str = "base"

    @abstractmethod
    def structured(
        self,
        *,
        system: str,
        evidence: dict[str, Any],
        output_model: type[T],
        prompt_version: str,
    ) -> AICallResult:
        """Return a validated ``output_model`` instance built from ``evidence``."""


class OfflineProvider(AIProvider):
    """Deterministic provider used for tests, CI and offline demos.

    It never invents numbers: it fills the declared output schema from the
    evidence dictionary it was given and labels itself in the output text.
    """

    name = "offline"

    def structured(
        self,
        *,
        system: str,
        evidence: dict[str, Any],
        output_model: type[T],
        prompt_version: str,
    ) -> AICallResult:
        started = time.perf_counter()
        payload = _fill_from_evidence(output_model, evidence)
        latency = int((time.perf_counter() - started) * 1000)
        try:
            output = output_model.model_validate(payload)
        except ValidationError as exc:  # pragma: no cover - schema mismatch guard
            return AICallResult(None, self.name, prompt_version, latency, error=str(exc))
        return AICallResult(output, "offline-deterministic", prompt_version, latency)


class OpenAICompatibleProvider(AIProvider):
    """Talks to any OpenAI style ``/chat/completions`` endpoint.

    Works with hosted gateways as well as local runtimes such as vLLM, Ollama
    or LM Studio -- only ``AI_BASE_URL`` and ``AI_MODEL`` change.
    """

    name = "openai_compatible"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def structured(
        self,
        *,
        system: str,
        evidence: dict[str, Any],
        output_model: type[T],
        prompt_version: str,
    ) -> AICallResult:
        schema = output_model.model_json_schema()
        safe_evidence = redact(evidence) if settings.ai_redact_pii else evidence
        instruction = (
            f"{system}\n\n"
            "You receive pre-computed evidence. Never invent, recompute or round "
            "numeric values -- reuse only the numbers you were given. Reply with a "
            "single JSON object matching this JSON Schema:\n"
            f"{json.dumps(schema)}"
        )
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": json.dumps(safe_evidence, default=str)},
        ]
        started = time.perf_counter()
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"},
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            usage = body.get("usage", {})
            output = output_model.model_validate_json(_strip_code_fence(content))
            latency = int((time.perf_counter() - started) * 1000)
            return AICallResult(
                output,
                self.model,
                prompt_version,
                latency,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
            )
        except Exception as exc:  # noqa: BLE001 - AI must never break analytics
            latency = int((time.perf_counter() - started) * 1000)
            logger.warning("AI call failed: %s", exc)
            return AICallResult(None, self.model, prompt_version, latency, error=str(exc))


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("`"):
        text = re.sub(r"^`{3}[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?`{3}$", "", text)
    return text.strip()


def _fill_from_evidence(model: type[BaseModel], evidence: dict[str, Any]) -> dict[str, Any]:
    """Build a schema-valid payload out of the supplied evidence only.

    Types are resolved by introspection rather than by matching the annotation
    as a string. String matching silently produced invalid payloads for any
    field that was a dict, a nested model, a list of models, or a Literal --
    the provider then reported a validation error and the caller saw "the AI
    layer is unavailable" when the real problem was here.
    """
    payload: dict[str, Any] = {}
    for name, field_info in model.model_fields.items():
        if name in evidence:
            payload[name] = evidence[name]
            continue
        payload[name] = _placeholder(field_info.annotation, name, evidence)
    return payload


def _placeholder(annotation: Any, name: str, evidence: dict[str, Any]) -> Any:
    """A schema-valid stand-in value for one field."""
    origin = get_origin(annotation)

    # Optional[X] / X | None: fill the first non-None member.
    if origin is Union or origin is UnionType:
        args = [a for a in get_args(annotation) if a is not type(None)]
        return _placeholder(args[0], name, evidence) if args else None

    if origin is Literal:
        options = get_args(annotation)
        return options[0] if options else _offline_sentence(name, evidence)

    if origin in (list, set, tuple):
        args = get_args(annotation)
        inner = args[0] if args else str
        if isinstance(inner, type) and issubclass(inner, BaseModel):
            return [_fill_from_evidence(inner, evidence)]
        if inner is str:
            return _offline_bullets(name)
        return []

    if origin is dict:
        return {}

    if isinstance(annotation, type):
        if issubclass(annotation, BaseModel):
            return _fill_from_evidence(annotation, evidence)
        if annotation is bool:
            return False
        if annotation in (int, float):
            return 0
        if annotation is dict:
            return {}
        if annotation is list:
            return []

    return _offline_sentence(name, evidence)


def _offline_sentence(field_name: str, evidence: dict[str, Any]) -> str:
    facts = ", ".join(
        f"{k}={v}" for k, v in list(evidence.items())[:4] if not isinstance(v, (dict, list))
    )
    label = field_name.replace("_", " ")
    return f"[offline] {label} derived from: {facts or 'supplied evidence'}"


def _offline_bullets(field_name: str) -> list[str]:
    label = field_name.replace("_", " ")
    return [f"[offline] {label} requires a configured AI provider"]


_provider: AIProvider | None = None


def get_provider() -> AIProvider:
    """Return the process-wide provider selected by ``AI_PROVIDER``."""
    global _provider
    if _provider is not None:
        return _provider
    if settings.ai_provider == "openai_compatible":
        _provider = OpenAICompatibleProvider(
            settings.ai_base_url,
            settings.ai_api_key,
            settings.ai_model,
            settings.ai_timeout_seconds,
        )
    else:
        _provider = OfflineProvider()
    return _provider


def set_provider(provider: AIProvider | None) -> None:
    """Test hook to swap the provider implementation."""
    global _provider
    _provider = provider
