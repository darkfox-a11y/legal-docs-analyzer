import pytest

from app.rag.chunking import smart_chunking
from app.rag.embeddings import compute_similarity
from app.rag.evaluation import create_evaluation_report, evaluate_rag_pipeline


def test_smart_chunking_creates_chunks_for_contract_text():
    text = """
    EMPLOYMENT AGREEMENT

    This agreement is made on January 1, 2024. The Employee will receive
    a salary of $150,000 per year. Payment is made bi-weekly.

    TERMINATION

    Either party may terminate with 30 days notice. Upon termination, all
    company property must be returned immediately.
    """

    chunks = smart_chunking(text, document_type="contract", chunk_size=200)

    assert chunks
    assert any("Payment is made bi-weekly." in chunk for chunk in chunks)
    assert any("Either party may terminate" in chunk for chunk in chunks)


def test_compute_similarity_scores_related_vectors_higher():
    related_a = [1.0, 0.0, 0.0]
    related_b = [0.9, 0.1, 0.0]
    unrelated = [0.0, 1.0, 0.0]

    similar_score = compute_similarity(related_a, related_b)
    different_score = compute_similarity(related_a, unrelated)

    assert similar_score > different_score
    assert similar_score > 0.9


def test_evaluate_rag_pipeline_with_expected_answer_uses_mocked_embeddings(monkeypatch):
    embedding_map = {
        "According to the contract, payment is due within 30 days from invoice date.": [1.0, 0.0],
        "Payment is due within 30 days of invoice.": [0.95, 0.05],
    }

    def fake_generate_single_embedding(text, model_name="default", normalize=True):
        return embedding_map[text]

    monkeypatch.setattr(
        "app.rag.embeddings.generate_single_embedding",
        fake_generate_single_embedding,
    )

    evaluation = evaluate_rag_pipeline(
        question="What is the payment schedule?",
        answer="According to the contract, payment is due within 30 days from invoice date.",
        retrieved_chunks=[
            {"text": "Payment is due within 30 days of invoice.", "score": 0.89},
            {"text": "Invoices are sent monthly.", "score": 0.72},
        ],
        confidence="high",
        expected_answer="Payment is due within 30 days of invoice.",
    )

    assert evaluation["retrieval"]["avg_score"] == pytest.approx(0.805)
    assert evaluation["answer"]["confidence_level"] == "high"
    assert evaluation["answer"]["matches_expected"] is True
    assert evaluation["overall_quality"] in {"good", "excellent"}


def test_create_evaluation_report_contains_key_sections():
    evaluation = evaluate_rag_pipeline(
        question="What is the payment schedule?",
        answer="According to the contract, payment is due within 30 days from invoice date.",
        retrieved_chunks=[
            {"text": "Payment is due within 30 days of invoice.", "score": 0.89},
            {"text": "Invoices are sent monthly.", "score": 0.72},
        ],
        confidence="high",
    )

    report = create_evaluation_report(evaluation)

    assert "RAG PIPELINE EVALUATION REPORT" in report
    assert "RETRIEVAL METRICS" in report
    assert "ANSWER METRICS" in report
