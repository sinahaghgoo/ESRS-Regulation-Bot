"""Prompt builders for RQ2 ESRS-context ablations."""

from __future__ import annotations

from rag.types import GenerationSample, RetrievedEvidenceChunk


def format_retrieved_chunks(chunks: list[RetrievedEvidenceChunk]) -> str:
    """Format retrieved evidence chunks into a deterministic prompt block."""
    blocks: list[str] = []
    for chunk in chunks:
        header = " > ".join(part.strip() for part in chunk.header_hierarchy if part.strip())
        page_bits = [str(p) for p in (chunk.page_start, chunk.page_end) if p is not None]
        if len(page_bits) == 2 and page_bits[0] == page_bits[1]:
            page_label = page_bits[0]
        elif page_bits:
            page_label = "-".join(page_bits)
        else:
            page_label = "unknown"
        blocks.append(
            "\n".join(
                [
                    f"[Evidence {chunk.rank}]",
                    f"Pages: {page_label}",
                    f"Section: {header or 'Unknown section'}",
                    "Text:",
                    chunk.text.strip(),
                ]
            ).strip()
        )
    return "\n\n".join(blocks)


def _resolve_context_mode(*, esrs_context: str, condition: str | None) -> tuple[bool, str]:
    """Resolve whether ESRS context is intentionally injected for this run."""
    condition_norm = (condition or "").strip().lower()
    if condition_norm in {"with_esrs_context", "with_context", "full_system"}:
        return True, "with_esrs_context"
    if condition_norm in {"without_esrs_context", "without_context", "no_esrs_context"}:
        return False, "without_esrs_context"
    has_context = bool(esrs_context.strip())
    return has_context, ("with_esrs_context" if has_context else "without_esrs_context")


def build_answer_messages(
    *,
    sample: GenerationSample,
    retrieved_chunks: list[RetrievedEvidenceChunk],
    esrs_context: str,
    condition: str | None = None,
) -> list[dict[str, str]]:
    """Build the RQ2 answer-generation prompt for one company/question pair."""
    evidence_block = format_retrieved_chunks(retrieved_chunks)
    use_esrs_context, mode_label = _resolve_context_mode(
        esrs_context=esrs_context,
        condition=condition,
    )

    mode_block = (
        "ESRS context mode: WITH_CONTEXT.\n"
        "The ESRS context block is regulatory ESRS requirement text mapped to the "
        "question. Use it only as background to understand what the question expects.\n"
        "CRITICAL: Do NOT use the ESRS context as a scoring checklist. Do NOT penalize the "
        "company for failing to address every individual ESRS paragraph or sub-requirement. "
        "The compliance level should reflect whether the company's disclosure substantively "
        "answers the question, not whether it ticks every ESRS sub-paragraph.\n"
        "Never treat ESRS context text as proof that the company disclosed something. "
        "Company disclosure claims must be grounded only in retrieved company report evidence."
        if use_esrs_context
        else
        "ESRS context mode: WITHOUT_CONTEXT.\n"
        "No ESRS context block is included in this condition. Do not invent additional "
        "requirements from memory; infer expectations only from the question wording and "
        "retrieved company report evidence."
    )

    system = "\n".join(
        [
            "You evaluate ESRS disclosure fulfillment for one company sustainability-report question.",
            "Important: compliance level measures how well the company's disclosure addresses the question, not sustainability performance quality.",
            "Evaluate only the scope of the asked question. Do not treat ESRS context as a line-by-line checklist.",
            "Ground every factual claim about the company in retrieved report evidence.",
            "Do not penalize for missing exact ESRS wording, table formatting, data-point tagging, or paragraph-level cross-references when the evidence substantively answers the question.",
            "If the company states the topic is not material or not applicable, that alone is Low fulfillment — a non-materiality declaration with only a brief statement is not a substantive disclosure. However, if the topic IS assessed as material and the company discloses an explicit absence or negative result as a substantive finding, that can count toward fulfillment.",
            "If evidence is partial, ambiguous, or conflicting, acknowledge uncertainty and calibrate the label accordingly.",
            mode_block,
            "",
            "COMPLIANCE LEVEL RUBRIC (apply these definitions strictly):",
            "High = The disclosure comprehensively addresses all or nearly all aspects of the question. "
            "Required quantitative metrics, qualitative descriptions, and contextual information are present in the evidence. "
            "Minor omissions of secondary details do not prevent a High rating if the core question is thoroughly answered.",
            "Medium = The disclosure partially addresses the question. Some key aspects are covered but significant gaps exist "
            "(e.g., missing required metrics, incomplete coverage of major sub-requirements, or key information only cross-referenced without substance). "
            "Assign Medium only when material information needed to answer the question is clearly absent from the evidence.",
            "Low = The disclosure provides minimal or no coverage of the question. The topic may be assessed as non-material "
            "with only a brief statement, or the disclosure is limited to a cross-reference without substantive content.",
            "",
            "CALIBRATION GUIDANCE:",
            "- When evidence covers the main substance of what the question asks (methodology, key results, governance), lean toward High even if some peripheral details are absent.",
            "- Reserve Medium for cases where the evidence clearly leaves major parts of the question unanswered.",
            "- Reserve Low for cases where the evidence is essentially absent or only a brief mention exists.",
            "- If the company declares the topic non-material or not applicable (e.g., 'S4 is not material'), assign Low — a non-materiality statement alone does not constitute substantive disclosure fulfillment, regardless of whether tangentially related general policies exist.",
            "",
            "Return one JSON object with exactly these keys: direct_answer, compliance_level, compliance_rationale, key_findings, disclosure_gaps, cited_pages.",
            "compliance_level must be one of Low, Medium, High.",
            "key_findings, disclosure_gaps, and cited_pages must be JSON arrays.",
            "",
            "CITED_PAGES — RULES (read carefully):",
            "- Include EVERY retrieved page whose specific content directly supports a quantitative metric, methodology step, or substantive factual claim that you actually state in direct_answer or compliance_rationale.  Missing a supporting page leaves that claim ungrounded.",
            "- EXCLUDE retrieved pages whose content you did not actually use to back a claim — pages whose role is only general background, navigation, or tangentially-related material MUST be omitted.",
            "- Aim for the union of all directly-cited evidence; this is typically 4 to 7 pages.  Do not artificially cap the list — completeness for actually-used pages is more important than minimising the count.  But also do not pad: every entry must correspond to a substantive claim.",
            "- Use the original page label format from the evidence block (a single page like \"167\" or a contiguous span like \"167-168\").  Prefer one span entry over enumerating each page within the span.",
            "- Page labels must be drawn from the retrieved evidence block; do not invent page numbers.",
            "- If two retrieved chunks cover the same page, cite that page (or its containing span) only once.",
        ]
    )

    user_sections = [
        f"Condition: {mode_label}",
        f"Company: {sample.company}",
        f"ESRS standard: {sample.esrs_standard}",
        f"Disclosure ID: {sample.esrs_disclosure_id}",
        f"Question: {sample.question}",
    ]
    if use_esrs_context:
        user_sections.extend(
            [
                "",
                "ESRS requirements context (regulatory reference, not company evidence):",
                esrs_context.strip(),
            ]
        )
    user_sections.extend(
        [
            "",
            "Retrieved company report evidence (only source for company-disclosure claims):",
            evidence_block,
            "",
            "Write a faithful direct answer, assign the disclosure fulfillment level, explain the reasoning, list key findings and only question-relevant disclosure gaps.  For cited_pages, list EVERY page whose specific content backs a substantive claim you actually wrote above (typically 4-7 pages); ensure every claim is grounded, but do not include retrieved pages whose content you did not use.",
        ]
    )
    user = "\n".join(user_sections)

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
