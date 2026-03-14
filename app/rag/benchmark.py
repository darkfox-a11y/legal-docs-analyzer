"""
Reusable benchmark helpers for comparing RAG performance across models.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "in", "is", "it", "its", "of", "on", "or", "that", "the", "their", "this",
    "to", "was", "were", "will", "with", "you", "your", "under", "into", "than",
    "then", "them", "they", "he", "she", "we", "our", "ours", "about", "what",
    "when", "where", "which", "who", "whom", "why", "how", "can", "could",
    "should", "would", "may", "might", "must", "shall", "do", "does", "did",
}


def load_json_file(file_path: str | Path) -> List[Dict]:
    """Load a JSON file containing a top-level list."""
    path = Path(file_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {path}, got {type(payload).__name__}")

    return payload


def load_questions(file_path: str | Path) -> List[Dict]:
    """Load the benchmark question set."""
    return load_json_file(file_path)


def load_corpus_manifest(file_path: str | Path) -> List[Dict]:
    """Load the normalized benchmark corpus manifest."""
    return load_json_file(file_path)


def normalize_text(text: str) -> str:
    """Normalize text for overlap-style metrics."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric terms."""
    return re.findall(r"[a-z0-9][a-z0-9\-']*", normalize_text(text))


def content_tokens(text: str) -> set[str]:
    """Return meaningful tokens with common stopwords removed."""
    return {token for token in tokenize(text) if token not in STOPWORDS and len(token) > 2}


def safe_ratio(numerator: float, denominator: float) -> float:
    """Return a bounded ratio even when the denominator is zero."""
    if denominator <= 0:
        return 0.0
    return max(0.0, min(1.0, numerator / denominator))


def safe_mean(values: Iterable[float]) -> float:
    """Average values, returning 0 for empty iterables."""
    items = list(values)
    return mean(items) if items else 0.0


def compute_keyword_coverage(text: str, expected_keywords: Sequence[str] | None) -> float:
    """Measure how many expected keywords appear in the text."""
    if not expected_keywords:
        return 0.0

    haystack = normalize_text(text)
    hits = sum(1 for keyword in expected_keywords if normalize_text(keyword) in haystack)
    return safe_ratio(hits, len(expected_keywords))


def compute_context_groundedness(answer: str, context_chunks: Sequence[str]) -> float:
    """Estimate how much of the answer is grounded in retrieved context."""
    answer_terms = content_tokens(answer)
    if not answer_terms:
        return 0.0

    context_terms: set[str] = set()
    for chunk in context_chunks:
        context_terms.update(content_tokens(chunk))

    return safe_ratio(len(answer_terms & context_terms), len(answer_terms))


def compute_question_alignment(answer: str, question: str) -> float:
    """Measure whether the answer addresses the key terms from the question."""
    question_terms = content_tokens(question)
    if not question_terms:
        return 0.0

    answer_terms = content_tokens(answer)
    return safe_ratio(len(question_terms & answer_terms), len(question_terms))


def compute_retrieval_keyword_hit_rate(
    retrieved_chunks: Sequence[Dict],
    expected_keywords: Sequence[str] | None,
) -> float:
    """Measure expected keyword coverage across retrieved chunks."""
    if not retrieved_chunks or not expected_keywords:
        return 0.0

    combined_context = " ".join(
        chunk.get("text_preview") or chunk.get("text", "")
        for chunk in retrieved_chunks
        if isinstance(chunk, dict)
    )
    return compute_keyword_coverage(combined_context, expected_keywords)


def normalized_judge_score(judge_scores: Dict | None) -> float:
    """Convert judge outputs into a single 0-5 score."""
    if not judge_scores:
        return 0.0

    numeric_scores = []
    for key, value in judge_scores.items():
        if not key.endswith("_score") or not isinstance(value, (int, float)):
            continue

        score = float(value)
        if key == "hallucination_risk_score":
            score = max(1.0, min(5.0, 6.0 - score))
        numeric_scores.append(score)

    return safe_mean(numeric_scores)


def score_benchmark_case(
    *,
    model_name: str,
    document_id: str,
    document_title: str,
    question: Dict,
    rag_response: Dict,
    latency_ms: float,
    judge_scores: Dict | None = None,
) -> Dict:
    """
    Build a normalized metric bundle for one model/document/question run.
    """
    sources = rag_response.get("sources", [])
    context_chunks = [source.get("text_preview") or source.get("text", "") for source in sources]
    answer = rag_response.get("answer", "")
    retrieval_scores = [
        source.get("relevance_score", source.get("score", 0.0))
        for source in sources
        if isinstance(source, dict)
    ]
    expected_keywords = question.get("expected_keywords", [])

    retrieval_metrics = {
        "num_chunks": len(sources),
        "avg_score": safe_mean(retrieval_scores),
        "max_score": max(retrieval_scores) if retrieval_scores else 0.0,
        "min_score": min(retrieval_scores) if retrieval_scores else 0.0,
        "keyword_hit_rate": compute_retrieval_keyword_hit_rate(sources, expected_keywords),
    }

    answer_metrics = {
        "word_count": len(answer.split()),
        "groundedness": compute_context_groundedness(answer, context_chunks),
        "question_alignment": compute_question_alignment(answer, question["prompt"]),
        "keyword_coverage": compute_keyword_coverage(answer, expected_keywords),
        "is_generic": "couldn't find" in normalize_text(answer) or "insufficient information" in normalize_text(answer),
    }

    judge_mean = normalized_judge_score(judge_scores)

    latency_score = max(0.0, 1.0 - min(latency_ms, 30000.0) / 30000.0)
    composite_score = safe_mean([
        retrieval_metrics["avg_score"],
        retrieval_metrics["keyword_hit_rate"],
        answer_metrics["groundedness"],
        answer_metrics["keyword_coverage"],
        answer_metrics["question_alignment"],
        judge_mean / 5.0 if judge_mean else 0.0,
        latency_score,
    ])

    return {
        "model_name": model_name,
        "document_id": document_id,
        "document_title": document_title,
        "question_id": question["id"],
        "question_category": question["category"],
        "question_prompt": question["prompt"],
        "latency_ms": round(latency_ms, 2),
        "confidence": rag_response.get("confidence", "unknown"),
        "retrieval": retrieval_metrics,
        "answer": answer_metrics,
        "judge": judge_scores or {},
        "composite_score": round(composite_score, 4),
    }


def aggregate_results(case_results: Sequence[Dict]) -> Dict:
    """Aggregate case-level results by model for easy comparison."""
    by_model: Dict[str, List[Dict]] = {}
    for result in case_results:
        by_model.setdefault(result["model_name"], []).append(result)

    models = []
    for model_name, results in by_model.items():
        models.append({
            "model_name": model_name,
            "cases": len(results),
            "avg_latency_ms": round(safe_mean(result["latency_ms"] for result in results), 2),
            "avg_composite_score": round(safe_mean(result["composite_score"] for result in results), 4),
            "avg_retrieval_score": round(safe_mean(result["retrieval"]["avg_score"] for result in results), 4),
            "avg_retrieval_keyword_hit_rate": round(
                safe_mean(result["retrieval"]["keyword_hit_rate"] for result in results), 4
            ),
            "avg_groundedness": round(safe_mean(result["answer"]["groundedness"] for result in results), 4),
            "avg_answer_keyword_coverage": round(
                safe_mean(result["answer"]["keyword_coverage"] for result in results), 4
            ),
            "avg_question_alignment": round(
                safe_mean(result["answer"]["question_alignment"] for result in results), 4
            ),
            "generic_answer_rate": round(
                safe_mean(1.0 if result["answer"]["is_generic"] else 0.0 for result in results), 4
            ),
            "avg_judge_score": round(
                safe_mean(
                    normalized_judge_score(result.get("judge", {}))
                    for result in results
                ),
                4,
            ),
        })

    return {
        "total_cases": len(case_results),
        "models": sorted(models, key=lambda item: item["avg_composite_score"], reverse=True),
        "results": list(case_results),
    }


def build_markdown_report(aggregate_payload: Dict) -> str:
    """Render a compact human-readable summary."""
    lines = [
        "# RAG Benchmark Summary",
        "",
        f"Total cases: {aggregate_payload['total_cases']}",
        "",
        "| Model | Cases | Composite | Retrieval | Groundedness | Keyword Coverage | Latency (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for model in aggregate_payload.get("models", []):
        lines.append(
            "| {model_name} | {cases} | {avg_composite_score:.4f} | {avg_retrieval_score:.4f} | "
            "{avg_groundedness:.4f} | {avg_answer_keyword_coverage:.4f} | {avg_latency_ms:.2f} |".format(
                **model
            )
        )

    return "\n".join(lines) + "\n"
