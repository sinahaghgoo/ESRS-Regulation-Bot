"""LLM answer generation for RQ2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from openai import OpenAI

from rag.config.settings import RQ2_RESULTS_DIR, get_generation_model, get_openai_api_key
from rag.generation.prompts import build_answer_messages
from rag.types import GeneratedAnswer, GenerationSample, RetrievedEvidenceChunk


class AnswerGenerator:
    """Generate structured ESRS disclosure answers with strict JSON validation."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_completion_tokens: int = 4096,
        client: OpenAI | None = None,
        require_openai: bool = True,
    ) -> None:
        self.model = model or get_generation_model()
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.client = client
        if self.client is None:
            key = get_openai_api_key(required=require_openai)
            if key:
                self.client = OpenAI(api_key=key)
        if require_openai and self.client is None:
            raise RuntimeError("OpenAI client could not be initialised for RQ2 answer generation.")

    def _call_json(self, messages: list[dict[str, str]]) -> str:
        if self.client is None:
            raise RuntimeError("OpenAI client is not available.")
        request_kwargs = {
            "model": self.model,
            "reasoning": {"effort": "medium"},
            "input": messages,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "rq2_generated_answer",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "direct_answer": {
                                "type": "string",
                                "minLength": 1,
                            },
                            "compliance_level": {
                                "type": "string",
                                "enum": ["Low", "Medium", "High"],
                            },
                            "compliance_rationale": {
                                "type": "string",
                                "minLength": 1,
                            },
                            "key_findings": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "disclosure_gaps": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "cited_pages": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "direct_answer",
                            "compliance_level",
                            "compliance_rationale",
                            "key_findings",
                            "disclosure_gaps",
                            "cited_pages",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "max_output_tokens": self.max_completion_tokens,
            "timeout": 180,
        }
        if self.temperature is not None:
            request_kwargs["temperature"] = self.temperature
        response = self.client.responses.create(
            **request_kwargs,
        )
        return (response.output_text or "").strip()

    @staticmethod
    def _normalize_label(value: str) -> str:
        mapping = {"low": "Low", "medium": "Medium", "high": "High"}
        norm = mapping.get(str(value).strip().lower())
        if norm is None:
            raise ValueError(f"Invalid compliance_level: {value!r}")
        return norm

    @staticmethod
    def _normalize_str_list(value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    @classmethod
    def _parse_generated_answer(cls, raw: str) -> GeneratedAnswer:
        payload = json.loads(raw)
        direct_answer = str(payload.get("direct_answer", "")).strip()
        rationale = str(payload.get("compliance_rationale", "")).strip()
        if not direct_answer:
            raise ValueError("direct_answer is empty")
        if not rationale:
            raise ValueError("compliance_rationale is empty")

        return GeneratedAnswer(
            direct_answer=direct_answer,
            compliance_level=cls._normalize_label(payload.get("compliance_level", "")),
            compliance_rationale=rationale,
            key_findings=cls._normalize_str_list(payload.get("key_findings")),
            disclosure_gaps=cls._normalize_str_list(payload.get("disclosure_gaps")),
            cited_pages=cls._normalize_str_list(payload.get("cited_pages")),
            raw_response=raw,
            parse_error=None,
        )

    def generate(
        self,
        *,
        sample: GenerationSample,
        retrieved_chunks: list[RetrievedEvidenceChunk],
        esrs_context: str,
        condition: str | None = None,
    ) -> GeneratedAnswer:
        """Generate and validate a single structured answer."""
        messages = build_answer_messages(
            sample=sample,
            retrieved_chunks=retrieved_chunks,
            esrs_context=esrs_context,
            condition=condition,
        )
        raw = self._call_json(messages)
        try:
            return self._parse_generated_answer(raw)
        except Exception as exc:
            repair_messages = messages + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        "Your previous response did not match the required JSON schema. "
                        f"Error: {exc}. Return only valid JSON with the required keys."
                    ),
                },
            ]
            repaired_raw = self._call_json(repair_messages)
            try:
                answer = self._parse_generated_answer(repaired_raw)
                answer.raw_response = repaired_raw
                return answer
            except Exception as repair_exc:
                return GeneratedAnswer(
                    direct_answer="",
                    compliance_level="",
                    compliance_rationale="",
                    key_findings=[],
                    disclosure_gaps=[],
                    cited_pages=[],
                    raw_response=repaired_raw or raw,
                    parse_error=str(repair_exc),
                )


# ---------------------------------------------------------------------------
# Module-level convenience function with file-based caching
# ---------------------------------------------------------------------------

_MODULE_GENERATOR: AnswerGenerator | None = None


def _get_module_generator() -> AnswerGenerator:
    global _MODULE_GENERATOR  # noqa: PLW0603
    if _MODULE_GENERATOR is None:
        _MODULE_GENERATOR = AnswerGenerator(require_openai=True)
    return _MODULE_GENERATOR


def _cache_key_for_call(
    question: str, company: str, condition: str, esrs_context: str,
) -> str:
    blob = f"{question}|{company}|{condition}|{len(esrs_context)}"
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def generate_answer(
    *,
    question: str,
    company: str,
    retrieved_passages: list[str],
    esrs_context: str,
    condition: str,
    use_cache: bool = True,
    question_id: str = "",
    esrs_standard: str = "",
    esrs_disclosure_id: str = "",
    sample: GenerationSample | None = None,
) -> dict:
    """High-level generation helper for notebook use.

    Wraps ``AnswerGenerator`` with optional file-based caching.  Returns a
    plain dict with ``compliance_level`` and ``answer`` keys.
    """
    cache_dir = RQ2_RESULTS_DIR / "nb_cache" / condition
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{_cache_key_for_call(question, company, condition, esrs_context)}.json"

    if use_cache and cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if cached.get("compliance_level"):
                return cached
        except Exception:
            pass

    gen = _get_module_generator()

    if sample is None:
        sample = GenerationSample(
            question_id=question_id, question=question,
            esrs_standard=esrs_standard,
            esrs_disclosure_id=esrs_disclosure_id, company=company,
            ground_truth_answer="", fulfillment_level="",
            fulfillment_rationale="", evidences=[],
        )

    chunks = [
        RetrievedEvidenceChunk(
            chunk_id=f"p{i}", text=text, score=1.0, rank=i + 1,
            page_start=None, page_end=None, header_hierarchy=[],
        )
        for i, text in enumerate(retrieved_passages)
    ]

    answer = gen.generate(
        sample=sample,
        retrieved_chunks=chunks,
        esrs_context=esrs_context,
        condition=condition,
    )

    result = {
        "compliance_level": answer.compliance_level,
        "answer": answer.direct_answer,
        "rationale": answer.compliance_rationale,
        "key_findings": answer.key_findings,
        "disclosure_gaps": answer.disclosure_gaps,
        "cited_pages": answer.cited_pages,
        "raw_response": answer.raw_response,
        "parse_error": answer.parse_error,
    }

    if use_cache:
        cache_file.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result
