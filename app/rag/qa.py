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

logger = logging.getLogger(__name__)

# Configure Gemini API
print("🤖 Configuring Gemini API...")
genai.configure(api_key=settings.gemini_api_key)
model = genai.GenerativeModel('gemini-2.0-flash')
print("✅ Gemini API ready!")


def answer_question(
    question: str,
    document_id: int = None,
    top_k: int = 3
) -> Dict:
    """
    Answer a question using RAG (Retrieval-Augmented Generation).
    
    How it works:
    1. Search for relevant chunks (vector search)
    2. Build prompt with question + chunks
    3. Send to Gemini
    4. Return answer + sources
    
    Args:
        question: User's question
        document_id: Optional - search specific document only
        top_k: Number of chunks to retrieve (default 3)
        
    Returns:
        Dictionary with answer, sources, and metadata
        
    Example:
        result = answer_question(
            question="What are the payment terms?",
            document_id=1,
            top_k=3
        )
        
        # Returns:
        {
            "question": "What are the payment terms?",
            "answer": "Payment is due within 30 days...",
            "sources": [
                {"text": "Payment is due...", "score": 0.89},
                {"text": "Late fees apply...", "score": 0.76}
            ]
        }
    """
    logger.info(f"🔍 Answering question: '{question}'")
    
    # Step 1: Find relevant chunks using vector search
    logger.info(f"📚 Searching for relevant chunks (top {top_k})...")
    relevant_chunks = search_similar_chunks(
        query=question,
        document_id=document_id,
        top_k=top_k
    )
    # 📞 CALLS vector_store.py!
    
    if not relevant_chunks:
        logger.warning("⚠️ No relevant chunks found")
        return {
            "question": question,
            "answer": "I couldn't find any relevant information in the document to answer this question.",
            "sources": [],
            "confidence": "low"
        }
    
    logger.info(f"✅ Found {len(relevant_chunks)} relevant chunks")
    
    # Step 2: Build context from chunks
    context = "\n\n".join([
        f"[Excerpt {i+1}]:\n{chunk['text']}"
        for i, chunk in enumerate(relevant_chunks)
    ])
    
    # Step 3: Build prompt for Gemini
    prompt = f"""You are a helpful AI assistant analyzing legal documents.

Question: {question}

Relevant excerpts from the document:

{context}

Instructions:
- Answer the question based ONLY on the provided excerpts
- Be concise and accurate
- If the excerpts don't contain enough information, say so
- Cite which excerpt(s) you used (e.g., "According to Excerpt 1...")
- Use professional language appropriate for legal documents

Answer:"""
    
    logger.info("🤖 Sending to Gemini...")
    
    # Step 4: Get answer from Gemini
    try:
        response = model.generate_content(prompt)
        answer = response.text
        logger.info("✅ Gemini response received")
        
    except Exception as e:
        logger.error(f"❌ Gemini error: {e}")
        return {
            "question": question,
            "answer": "I encountered an error while generating the answer. Please try again.",
            "sources": relevant_chunks,
            "error": str(e)
        }
    
    # Step 5: Return result with sources
    return {
        "question": question,
        "answer": answer,
        "sources": relevant_chunks,
        "confidence": "high" if relevant_chunks[0]['score'] > 0.7 else "medium"
    }


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
    
    result = answer_question(
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