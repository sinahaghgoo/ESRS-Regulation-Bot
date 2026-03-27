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


def build_answer_messages(
    *,
    sample: GenerationSample,
    retrieved_chunks: list[RetrievedEvidenceChunk],
    esrs_context: str,
) -> list[dict[str, str]]:
    """Build the RQ2 answer-generation prompt for one company/question pair."""
    evidence_block = format_retrieved_chunks(retrieved_chunks)
    system = (
        "You evaluate ESRS disclosure fulfillment for a company's sustainability report. "
        "Important: the compliance level refers to disclosure fulfillment, not to whether "
        "the company's sustainability performance is good or bad. Evaluate fulfillment at "
        "the level of the specific research question, not as an exhaustive ESRS checklist "
        "unless the question explicitly asks for a full checklist. Ground every claim in the "
        "retrieved report evidence. Use the ESRS requirements block only as interpretive "
        "context for the question. Do not downgrade solely because the disclosure is not "
        "presented in a single table, does not use exact ESRS wording, or lacks data-point-"
        "level tagging, if the retrieved evidence still clearly and substantially answers "
        "the question. If the company clearly discloses that something is absent, not "
        "applicable, or negative, that can still support High disclosure fulfillment when "
        "it directly answers the question. If a required detail is genuinely missing from "
        "the retrieved evidence, say so. "
        "Use this rubric: High = the retrieved evidence clearly and substantially answers "
        "the question with the key process or outcome details needed for that question; "
        "Medium = the evidence answers the question only partially or with material "
        "ambiguity; Low = the evidence does not answer the question or only mentions it "
        "superficially. "
        "Return a JSON object with exactly these keys: "
        "`direct_answer`, `compliance_level`, `compliance_rationale`, "
        "`key_findings`, `disclosure_gaps`, `cited_pages`. "
        "`compliance_level` must be one of `Low`, `Medium`, `High`. "
        "`key_findings`, `disclosure_gaps`, and `cited_pages` must be JSON arrays."
    )
    user = (
        f"Question ID: {sample.question_id}\n"
        f"Company: {sample.company}\n"
        f"ESRS standard: {sample.esrs_standard}\n"
        f"Disclosure ID: {sample.esrs_disclosure_id}\n"
        f"Question: {sample.question}\n\n"
        "Injected ESRS requirements:\n"
        f"{esrs_context.strip()}\n\n"
        "Retrieved company report evidence:\n"
        f"{evidence_block}\n\n"
        "Write a faithful direct answer, assign the disclosure fulfillment level, explain "
        "the reasoning, list key findings and only question-relevant disclosure gaps, and "
        "cite only pages that appear in the retrieved evidence."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
