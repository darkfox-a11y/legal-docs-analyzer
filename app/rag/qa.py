"""
Question Answering with Gemini AI.

This is where the magic happens! 🪄
1. User asks a question
2. We find relevant chunks (vector search)
3. We send chunks + question to Gemini
4. Gemini generates intelligent answer!
"""

import google.generativeai as genai
from typing import List, Dict
import logging

from app.config import settings
from app.rag.vector_store import search_similar_chunks
from app.rag.evaluation import evaluate_rag_pipeline

logger = logging.getLogger(__name__)

# Configure Gemini API
print("🤖 Configuring Gemini API...")
genai.configure(api_key=settings.gemini_api_key)
model = genai.GenerativeModel('gemini-2.0-flash')
print("✅ Gemini API ready!")


def answer_query(
    query: str,  
    document_id: int = None, 
    top_k: int = 5,
    detail_level: str = "detailed"
) -> dict:
    """
    Answer a query using RAG with intelligent reasoning and inference
    """
    # Search for relevant chunks (use imported function)
    search_results = search_similar_chunks(  
        query=query,
        top_k=top_k,
        document_id=document_id
    )
    
    if not search_results:
        return {
            "answer": "I couldn't find any relevant information in the document to answer your question. Please try rephrasing or asking about different aspects of the document.",
            "context": [],
            "sources": [],
            "confidence": "none",
            "detail_level": detail_level
        }
    
    # Build context
    context_parts = []
    for i, result in enumerate(search_results, 1):
        context_parts.append(f"[Excerpt {i}]:\n{result['text']}\n")
    
    context = "\n".join(context_parts)
    
    # Enhanced prompt with reasoning capabilities
    prompt = f"""You are an expert legal document analyzer with strong analytical and reasoning abilities. Your task is to provide comprehensive, intelligent answers based on document excerpts.

**USER QUESTION:**
{query}

**RELEVANT DOCUMENT EXCERPTS:**
{context}

**YOUR TASK:**
Provide a detailed, helpful answer following these guidelines:

**1. DIRECT INFORMATION:**
   - If the answer is explicitly stated in the excerpts, provide it directly
   - Quote specific phrases using quotation marks

**2. INFERENCE & REASONING:**
   - If the exact answer isn't stated, use logical reasoning based on the available information
   - Make reasonable inferences from the context provided
   - Explain your reasoning process clearly
   - Use phrases like "Based on the information provided..." or "From the context, we can infer..."

**3. COMPARATIVE ANALYSIS:**
   - If the question asks about alternatives or comparisons, analyze what IS stated
   - Draw logical conclusions from the available data
   - Provide context for understanding the implications

**4. HANDLING INCOMPLETE INFORMATION:**
   - If information is partially available, explain what you know and what's reasonable to conclude
   - Don't just say "I can't answer" - instead, provide what insights you CAN offer
   - Suggest what additional information would be needed for a complete answer

**5. RESPONSE STRUCTURE:**
   - Start with the most direct answer you can provide
   - Follow with supporting details and reasoning
   - If making inferences, clearly indicate this
   - End with any relevant caveats or additional context

**EXAMPLE APPROACH:**
Instead of: "The excerpts don't mention X, so I can't answer."
Better: "While the excerpts don't explicitly state X, based on the information provided about Y and Z, we can reasonably infer that... [explanation]. This suggests that..."

**IMPORTANT RULES:**
✓ Use logical reasoning and inference when direct answers aren't available
✓ Be helpful and provide actionable insights
✓ Always distinguish between explicit statements and inferences
✓ Base all reasoning on information in the excerpts
✓ Be professional and thorough
✗ Don't make wild guesses unconnected to the document
✗ Don't refuse to answer if you can provide useful context
✗ Don't invent facts not supported by the excerpts

**YOUR DETAILED ANSWER:**"""
    
    try:
        # Create model instance here
        gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Configure for better reasoning
        generation_config = genai.types.GenerationConfig(
            temperature=0.4,
            max_output_tokens=1500,
            top_p=0.95,
            top_k=40
        )
        
        response = gemini_model.generate_content(
            prompt,
            generation_config=generation_config
        )
        answer = response.text
        
        # Determine confidence level
        answer_lower = answer.lower()
        if any(phrase in answer_lower for phrase in [
            "explicitly states", "clearly indicates", "according to", "directly mentions"
        ]):
            confidence = "high"
        elif any(phrase in answer_lower for phrase in [
            "infer", "suggest", "might be", "could be", "possibly", "likely", "reasonably conclude"
        ]):
            confidence = "medium"
        elif any(phrase in answer_lower for phrase in [
            "cannot determine", "unclear", "insufficient information"
        ]):
            confidence = "low"
        else:
            confidence = "medium"
            
    except Exception as e:
        logger.error(f"❌ Error generating answer: {e}")
        return {
            "answer": f"Error generating answer: {str(e)}",
            "context": context_parts,
            "sources": search_results,
            "confidence": "error",
            "detail_level": detail_level
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
        "detail_level": detail_level
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

def summarize_document(document_id: int, max_chunks: int = 10) -> Dict:
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
        top_k=max_chunks
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
        response = model.generate_content(prompt)
        result = response.text
        
        return {
            "document_id": document_id,
            "summary": result,
            "chunks_analyzed": len(chunks)
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