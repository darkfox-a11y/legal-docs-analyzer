from app.rag.benchmark import (
    aggregate_results,
    build_markdown_report,
    compute_context_groundedness,
    compute_keyword_coverage,
    score_benchmark_case,
)


def test_compute_keyword_coverage_hits_expected_terms():
    text = "The agreement may terminate service, suspend access, and cancel accounts for misuse."
    coverage = compute_keyword_coverage(text, ["terminate", "suspend", "cancel", "refund"])
    assert coverage == 0.75


def test_compute_context_groundedness_uses_context_terms():
    answer = "The service may suspend accounts for misuse and terminate repeat violators."
    context = [
        "We may suspend accounts for misuse.",
        "Repeat violators may have service terminated.",
    ]
    groundedness = compute_context_groundedness(answer, context)
    assert groundedness > 0.5


def test_score_and_aggregate_results_produce_model_summary():
    question = {
        "id": "termination",
        "category": "termination",
        "prompt": "How can access be terminated?",
        "expected_keywords": ["terminate", "suspend"],
    }
    rag_response = {
        "answer": "The provider may suspend or terminate access for violations.",
        "confidence": "high",
        "sources": [
            {
                "text_preview": "The provider may suspend or terminate access for violations.",
                "relevance_score": 0.88,
            }
        ],
    }

    case = score_benchmark_case(
        model_name="gemini-2.0-flash",
        document_id="doc_1",
        document_title="Sample Terms",
        question=question,
        rag_response=rag_response,
        latency_ms=1200,
        judge_scores={
            "groundedness_score": 5,
            "completeness_score": 4,
            "legal_reasoning_score": 4,
            "clarity_score": 5,
            "hallucination_risk_score": 1,
        },
    )

    aggregate_payload = aggregate_results([case])

    assert aggregate_payload["total_cases"] == 1
    assert aggregate_payload["models"][0]["model_name"] == "gemini-2.0-flash"
    assert aggregate_payload["models"][0]["avg_retrieval_score"] == 0.88
    assert "RAG Benchmark Summary" in build_markdown_report(aggregate_payload)
