"""Shared OpenAI LLM utilities for JSON-first responses with retries."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional, Tuple

from openai import OpenAI

from rag.config import LLM_MODEL, OPENAI_API_KEY

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def _usage_to_dict(usage: Any) -> Dict[str, int]:
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if isinstance(usage, dict):
        return {
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
        }
    # pydantic object from SDK
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def chat_json(
    system_prompt: str,
    user_prompt: str,
    model: str = LLM_MODEL,
    temperature: float = 0.1,
    max_completion_tokens: int = 2048,
    max_retries: int = 3,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Call chat completions with JSON mode and return parsed JSON + usage."""
    client = _get_client()

    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
                response_format={"type": "json_object"},
            )
            content = (resp.choices[0].message.content or "").strip()
            parsed = json.loads(content)
            return parsed, _usage_to_dict(resp.usage)
        except json.JSONDecodeError as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(min(2 ** attempt, 8))
                continue
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(min(2 ** attempt, 8))
                continue
            break

    error_payload = {
        "error": f"LLM call failed: {last_error}",
        "raw_response": "",
    }
    return error_payload, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
