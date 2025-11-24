"""
RAG Evaluation System - Measure quality of retrieval and generation.
"""

from typing import List, Dict, Tuple
import logging
from app.rag.embeddings import compute_similarity

logger = logging.getLogger(__name__)


def evaluate_retrieval(
    query: str,
    retrieved_chunks: List[Dict],
    ground_truth_chunks: List[str] = None
) -> Dict:
    """
    Evaluate quality of retrieved chunks.
    
    Metrics:
    - Relevance scores
    - Score distribution
    - Coverage (if ground truth provided)
    
    Args:
        query: The search query
        retrieved_chunks: List of dicts with 'text' and 'score'
        ground_truth_chunks: Optional list of expected chunks
    
    Returns:
        Evaluation metrics
    """
    if not retrieved_chunks:
        return {
            "num_chunks": 0,
            "error": "No chunks retrieved"
        }
    
    scores = [chunk['score'] for chunk in retrieved_chunks]
    
    metrics = {
        "num_chunks": len(retrieved_chunks),
        "avg_score": sum(scores) / len(scores),
        "max_score": max(scores),
        "min_score": min(scores),
        "score_range": max(scores) - min(scores),
        "high_quality_chunks": sum(1 for s in scores if s > 0.7),  # Score > 0.7
        "medium_quality_chunks": sum(1 for s in scores if 0.5 < s <= 0.7),
        "low_quality_chunks": sum(1 for s in scores if s <= 0.5),
    }
    
    # Add coverage if ground truth provided
    if ground_truth_chunks:
        # Check how many ground truth chunks were retrieved
        retrieved_texts = [chunk['text'] for chunk in retrieved_chunks]
        coverage = sum(
            1 for gt in ground_truth_chunks 
            if any(gt in rt or rt in gt for rt in retrieved_texts)
        ) / len(ground_truth_chunks)
        metrics['ground_truth_coverage'] = coverage
    
    return metrics


def evaluate_answer_quality(
    question: str,
    answer: str,
    context_chunks: List[str],
    expected_answer: str = None
) -> Dict:
    """
    Evaluate quality of generated answer.
    
    Metrics:
    - Answer length
    - Context usage
    - Confidence indicators
    - Similarity to expected (if provided)
    
    Args:
        question: The question asked
        answer: Generated answer
        context_chunks: Context used to generate answer
        expected_answer: Optional expected answer
    
    Returns:
        Quality metrics
    """
    answer_lower = answer.lower()
    
    # Basic metrics
    metrics = {
        "answer_length": len(answer),
        "answer_word_count": len(answer.split()),
        "question_length": len(question),
    }
    
    # Check for confidence indicators
    high_confidence = [
        'explicitly states', 'clearly indicates', 'directly mentions',
        'according to', 'states that', 'specifies that'
    ]
    medium_confidence = [
        'suggests', 'indicates', 'implies', 'appears to',
        'based on', 'from the information', 'we can infer'
    ]
    low_confidence = [
        'unclear', 'not specified', 'cannot determine',
        'may or may not', 'insufficient information', 'does not mention'
    ]
    
    metrics['confidence_level'] = 'medium'  # default
    if any(phrase in answer_lower for phrase in high_confidence):
        metrics['confidence_level'] = 'high'
    elif any(phrase in answer_lower for phrase in low_confidence):
        metrics['confidence_level'] = 'low'
    elif any(phrase in answer_lower for phrase in medium_confidence):
        metrics['confidence_level'] = 'medium'
    
    # Context usage
    context_text = ' '.join(context_chunks).lower()
    
    # Count how many context words appear in answer
    answer_words = set(answer_lower.split())
    context_words = set(context_text.split())
    common_words = answer_words & context_words
    
    if len(answer_words) > 0:
        metrics['context_word_overlap'] = len(common_words) / len(answer_words)
    else:
        metrics['context_word_overlap'] = 0
    
    # Check if answer is generic/unhelpful
    generic_phrases = [
        'i cannot answer', 'no information', 'not found',
        'unable to determine', 'please rephrase'
    ]
    metrics['is_generic'] = any(phrase in answer_lower for phrase in generic_phrases)
    
    # Compare to expected answer if provided
    if expected_answer:
        from app.rag.embeddings import generate_single_embedding
        
        answer_emb = generate_single_embedding(answer)
        expected_emb = generate_single_embedding(expected_answer)
        
        similarity = compute_similarity(answer_emb, expected_emb)
        metrics['similarity_to_expected'] = similarity
        metrics['matches_expected'] = similarity > 0.75
    
    return metrics


def evaluate_rag_pipeline(
    question: str,
    answer: str,
    retrieved_chunks: List[Dict],
    confidence: str,
    expected_answer: str = None
) -> Dict:
    """
    Comprehensive evaluation of entire RAG pipeline.
    
    Args:
        question: User question
        answer: Generated answer
        retrieved_chunks: Retrieved context chunks
        confidence: Confidence level from QA system
        expected_answer: Optional expected answer for comparison
    
    Returns:
        Complete evaluation metrics
    """
    logger.info(f"📊 Evaluating RAG pipeline for question: {question[:50]}...")
    
    # Evaluate retrieval
    retrieval_metrics = evaluate_retrieval(question, retrieved_chunks)
    
    # Evaluate answer
    context_texts = [chunk['text'] for chunk in retrieved_chunks]
    answer_metrics = evaluate_answer_quality(
        question, 
        answer, 
        context_texts,
        expected_answer
    )
    
    # Overall assessment
    overall = {
        "question": question[:100],
        "retrieval": retrieval_metrics,
        "answer": answer_metrics,
        "confidence": confidence,
        "overall_quality": assess_overall_quality(
            retrieval_metrics, 
            answer_metrics, 
            confidence
        )
    }
    
    logger.info(f"✅ Evaluation complete - Overall quality: {overall['overall_quality']}")
    
    return overall


def assess_overall_quality(
    retrieval_metrics: Dict,
    answer_metrics: Dict,
    confidence: str
) -> str:
    """
    Assess overall quality based on all metrics.
    
    Returns: "excellent", "good", "fair", or "poor"
    """
    score = 0
    
    # Retrieval quality (max 40 points)
    if retrieval_metrics.get('avg_score', 0) > 0.8:
        score += 20
    elif retrieval_metrics.get('avg_score', 0) > 0.6:
        score += 10
    
    if retrieval_metrics.get('high_quality_chunks', 0) >= 2:
        score += 20
    elif retrieval_metrics.get('high_quality_chunks', 0) >= 1:
        score += 10
    
    # Answer quality (max 40 points)
    if answer_metrics.get('answer_length', 0) > 100:
        score += 10
    
    if not answer_metrics.get('is_generic', True):
        score += 15
    
    if answer_metrics.get('context_word_overlap', 0) > 0.3:
        score += 15
    
    # Confidence (max 20 points)
    if confidence == 'high':
        score += 20
    elif confidence == 'medium':
        score += 10
    
    # Determine overall quality
    if score >= 80:
        return "excellent"
    elif score >= 60:
        return "good"
    elif score >= 40:
        return "fair"
    else:
        return "poor"


def create_evaluation_report(evaluation: Dict) -> str:
    """
    Create human-readable evaluation report.
    
    Args:
        evaluation: Output from evaluate_rag_pipeline
    
    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 70)
    report.append("RAG PIPELINE EVALUATION REPORT")
    report.append("=" * 70)
    
    report.append(f"\n📝 Question: {evaluation['question']}")
    report.append(f"🎯 Overall Quality: {evaluation['overall_quality'].upper()}")
    report.append(f"🔍 Confidence: {evaluation['confidence']}")
    
    report.append("\n--- RETRIEVAL METRICS ---")
    retrieval = evaluation['retrieval']
    report.append(f"Chunks Retrieved: {retrieval['num_chunks']}")
    report.append(f"Average Score: {retrieval.get('avg_score', 0):.3f}")
    report.append(f"High Quality Chunks: {retrieval.get('high_quality_chunks', 0)}")
    report.append(f"Medium Quality Chunks: {retrieval.get('medium_quality_chunks', 0)}")
    report.append(f"Low Quality Chunks: {retrieval.get('low_quality_chunks', 0)}")
    
    report.append("\n--- ANSWER METRICS ---")
    answer = evaluation['answer']
    report.append(f"Answer Length: {answer['answer_length']} chars ({answer['answer_word_count']} words)")
    report.append(f"Confidence Level: {answer['confidence_level']}")
    report.append(f"Context Usage: {answer.get('context_word_overlap', 0):.1%}")
    report.append(f"Generic Answer: {'Yes' if answer.get('is_generic') else 'No'}")
    
    if 'similarity_to_expected' in answer:
        report.append(f"Similarity to Expected: {answer['similarity_to_expected']:.3f}")
    
    report.append("\n" + "=" * 70)
    
    return "\n".join(report)


# Test when run directly
if __name__ == "__main__":
    print("\n🧪 Testing Evaluation System\n")
    
    # Mock data
    test_question = "What are the payment terms?"
    test_answer = """Based on Section 4.2 of the contract, payment terms are net 30 days. 
    The client must pay within 30 days from invoice date. Late payments incur a 1.5% monthly fee."""
    
    test_chunks = [
        {"text": "Section 4.2: Payment Terms. Payment is due within 30 days.", "score": 0.89},
        {"text": "Late payments will incur a fee of 1.5% per month.", "score": 0.82},
        {"text": "All payments must be made in US Dollars.", "score": 0.65},
    ]
    
    # Run evaluation
    evaluation = evaluate_rag_pipeline(
        question=test_question,
        answer=test_answer,
        retrieved_chunks=test_chunks,
        confidence="high"
    )
    
    # Print report
    report = create_evaluation_report(evaluation)
    print(report)