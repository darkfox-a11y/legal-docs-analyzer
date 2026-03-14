"""
Question Answering with Gemini AI.

This is where the magic happens! 🪄
1. User asks a question
2. We find relevant chunks (vector search)
3. We send chunks + question to Gemini
4. Gemini generates intelligent answer!
"""

import logging
from typing import Dict

import google.generativeai as genai

from app.config import settings
from app.rag.vector_store import search_similar_chunks
from app.rag.evaluation import evaluate_rag_pipeline

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"


# Configure Gemini API once at import time.
genai.configure(api_key=settings.gemini_api_key)


def get_gemini_model(model_name: str = DEFAULT_GEMINI_MODEL) -> genai.GenerativeModel:
    """Create a Gemini model client for the requested model."""
    return genai.GenerativeModel(model_name)


def classify_question_type(query: str) -> str:
    """Classify a question so we can use a tighter prompt."""
    normalized = query.lower()
    if any(
        phrase in normalized
        for phrase in [
            "main legal issue",
            "legal issue",
            "procedural posture",
            "holding",
            "disposition",
            "relief",
            "remand",
            "analysis",
            "analyze",
            "implication",
        ]
    ):
        return "analytical"
    if any(
        phrase in normalized
        for phrase in ["parties involved", "orders", "decisions", "extract", "case name", "caption"]
    ):
        return "extraction"
    return "factual"


def normalize_document_type(document_type: str | None) -> str:
    """Normalize document type labels into stable answer modes."""
    normalized = (document_type or "general").lower().strip()
    if normalized in {"legal", "judicial", "court", "judgment", "order"}:
        return "judicial"
    if normalized in {"contract", "policy", "report"}:
        return normalized
    return "general"


def build_prompt(
    query: str,
    context: str,
    question_type: str,
    detail_level: str,
    document_type: str,
) -> str:
    """Build a concise prompt tuned to question type."""
    brevity_instruction = {
        "brief": "Keep the answer to 1-2 short sentences.",
        "detailed": "Keep the answer to 2-3 concise sentences.",
        "comprehensive": "Keep the answer to one short paragraph with only the most relevant details.",
    }.get(detail_level, "Keep the answer concise.")

    if document_type == "judicial":
        return f"""You are analyzing a court decision or judicial order.

Answer using only the provided excerpts.
Prefer exact case-specific information over general legal explanation.
If the excerpts show the caption, procedural posture, holding, disposition, or relief, state those directly.
If the excerpts are incomplete, say what is supported by the text and do not use outside knowledge.
{brevity_instruction}

Question:
{query}

Relevant excerpts:
{context}

Answer:"""

    if question_type == "analytical":
        return f"""You are a legal document analysis expert.

Answer the question using only the provided excerpts.
Give a brief analysis grounded in the text, and be direct when the context is clear.
Only express uncertainty when the excerpts are genuinely ambiguous.
{brevity_instruction}

Question:
{query}

Relevant excerpts:
{context}

Answer:"""

    return f"""You are a legal document analysis expert.

Answer the question directly using only the provided excerpts.
Be confident when the excerpts support a clear answer.
If possible, cite the relevant excerpt number.
Only express uncertainty when the text is genuinely ambiguous.
{brevity_instruction}

Question:
{query}

Relevant excerpts:
{context}

Answer:"""


def infer_confidence(answer: str, question_type: str, search_results: list[dict]) -> str:
    """Infer a confidence label from answer language and retrieval quality."""
    answer_lower = answer.lower()
    uncertainty_markers = [
        "cannot determine",
        "unclear",
        "insufficient information",
        "not possible to determine",
        "not enough information",
    ]
    explicit_markers = [
        "explicitly states",
        "clearly indicates",
        "according to",
        "directly mentions",
        "excerpt",
    ]
    inference_markers = [
        "infer",
        "suggest",
        "might be",
        "could be",
        "possibly",
        "likely",
        "reasonably conclude",
    ]

    if any(marker in answer_lower for marker in uncertainty_markers):
        return "low"
    if any(marker in answer_lower for marker in explicit_markers):
        return "high"
    if any(marker in answer_lower for marker in inference_markers):
        return "medium"

    top_score = search_results[0]["score"] if search_results else 0.0
    if question_type in {"factual", "extraction"} and top_score >= 0.75:
        return "high"
    return "medium"


def answer_query(
    query: str,  
    document_id: int = None, 
    top_k: int = 5,
    detail_level: str = "detailed",
    model_name: str = DEFAULT_GEMINI_MODEL,
    collection_name: str = None,
    document_type: str | None = None,
) -> dict:
    """
    Answer a query using RAG with intelligent reasoning and inference
    """
    prompt_document_type = normalize_document_type(document_type)
    effective_top_k = max(top_k, 5) if prompt_document_type == "judicial" else top_k

    # Search for relevant chunks (use imported function)
    search_results = search_similar_chunks(  
        query=query,
        top_k=effective_top_k,
        document_id=document_id,
        collection_name=collection_name,
    )
    
    if not search_results:
        return {
            "answer": "I couldn't find any relevant information in the document to answer your question. Please try rephrasing or asking about different aspects of the document.",
            "context": [],
            "sources": [],
            "confidence": "none",
            "detail_level": detail_level,
            "model_name": model_name,
            "document_type": prompt_document_type,
            "top_k_used": effective_top_k,
        }
    
    # Build context
    context_parts = []
    for i, result in enumerate(search_results, 1):
        context_parts.append(f"[Excerpt {i}]:\n{result['text']}\n")
    
    context = "\n".join(context_parts)
    
    question_type = classify_question_type(query)
    prompt = build_prompt(query, context, question_type, detail_level, prompt_document_type)
    
    try:
        gemini_model = get_gemini_model(model_name)
        
        # Configure for better reasoning
        generation_config = genai.types.GenerationConfig(
            temperature=0.1,
            max_output_tokens=300,
            top_p=0.8,
            top_k=20
        )
        
        response = gemini_model.generate_content(
            prompt,
            generation_config=generation_config
        )
        answer = response.text
        
        confidence = infer_confidence(answer, question_type, search_results)
            
    except Exception as e:
        logger.error(f"❌ Error generating answer: {e}")
        return {
            "answer": f"Error generating answer: {str(e)}",
            "context": context_parts,
            "sources": search_results,
            "confidence": "error",
            "detail_level": detail_level,
            "model_name": model_name,
            "document_type": prompt_document_type,
            "top_k_used": effective_top_k,
        }
    
    # Prepare result
    result = {
        "answer": answer,
        "context": context_parts,
        "sources": [
            {
                "chunk_index": src["chunk_index"],
                "text_preview": src["text"][:200] + "..." if len(src["text"]) > 200 else src["text"],
                "relevance_score": round(src["score"], 4)
            }
            for src in search_results
        ],
        "confidence": confidence,
        "detail_level": detail_level,
        "model_name": model_name,
        "document_type": prompt_document_type,
        "top_k_used": effective_top_k,
    }
    
    # Add evaluation metrics (optional, can be toggled)
    try:
        evaluation = evaluate_rag_pipeline(
            question=query,
            answer=answer,
            retrieved_chunks=search_results,
            confidence=confidence
        )
        result["evaluation"] = {
            "overall_quality": evaluation["overall_quality"],
            "retrieval_quality": evaluation["retrieval"]["avg_score"],
            "num_high_quality_chunks": evaluation["retrieval"]["high_quality_chunks"]
        }
        logger.info(f"📊 Answer quality: {evaluation['overall_quality']}")
    except Exception as e:
        logger.warning(f"Failed to evaluate: {e}")
    
    return result

def summarize_document(
    document_id: int,
    max_chunks: int = 10,
    model_name: str = DEFAULT_GEMINI_MODEL,
    collection_name: str = None,
) -> Dict:
    """
    Generate a summary of the entire document.
    
    Args:
        document_id: ID of document to summarize
        max_chunks: Maximum chunks to use for summary
        
    Returns:
        Dictionary with summary and key points
    """
    logger.info(f"📄 Summarizing document {document_id}")
    
    # Get representative chunks (we'll improve this later)
    # For now, just get top chunks from a generic query
    chunks = search_similar_chunks(
        query="main points key information important details",
        document_id=document_id,
        top_k=max_chunks,
        collection_name=collection_name,
    )
    
    if not chunks:
        return {
            "document_id": document_id,
            "summary": "Unable to generate summary - no content found.",
            "key_points": []
        }
    
    # Build context
    context = "\n\n".join([chunk['text'] for chunk in chunks])
    
    # Build prompt
    prompt = f"""You are analyzing a legal document. Based on the following excerpts, provide:

1. A concise summary (2-3 sentences)
2. Key points (bullet points)

Excerpts:

{context}

Please provide:
1. Summary:
2. Key Points:"""
    
    # Get response from Gemini
    try:
        response = get_gemini_model(model_name).generate_content(prompt)
        result = response.text
        
        return {
            "document_id": document_id,
            "summary": result,
            "chunks_analyzed": len(chunks),
            "model_name": model_name,
        }
        
    except Exception as e:
        logger.error(f"❌ Gemini error: {e}")
        return {
            "document_id": document_id,
            "summary": "Error generating summary.",
            "error": str(e)
        }


# Test when run directly
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 Testing QA Module")
    print("="*60 + "\n")
    
    # This assumes you have data in Qdrant from previous tests
    print("Note: This test requires data in Qdrant!")
    print("Run vector_store.py test first to populate test data.\n")
    
    # Test question
    test_question = "What are the payment terms?"
    
    print(f"Question: '{test_question}'\n")
    print("Generating answer with Gemini...\n")
    
    result = answer_query(
        question=test_question,
        document_id=999,  # Test document from vector_store.py
        top_k=3
    )
    
    print("="*60)
    print("RESULT:")
    print("="*60)
    print(f"\nQuestion: {result['question']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nConfidence: {result.get('confidence', 'N/A')}")
    print(f"\nSources used: {len(result['sources'])}")
    
    if result['sources']:
        print("\nRelevant excerpts:")
        for i, source in enumerate(result['sources'], 1):
            print(f"\n  {i}. (Score: {source['score']:.2f})")
            print(f"     {source['text'][:100]}...")
    
    print("\n" + "="*60)
    print("✅ QA Module test complete!")
    print("="*60 + "\n")
