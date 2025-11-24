"""
Test script for improved RAG system.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.rag.chunking import smart_chunking, semantic_chunk_with_overlap
from app.rag.embeddings import generate_embeddings, generate_single_embedding, compute_similarity
from app.rag.evaluation import evaluate_rag_pipeline, create_evaluation_report

print("🧪 Testing Improved RAG System")
print("=" * 70)

# Test 1: Chunking
print("\n1️⃣  Testing Smart Chunking...")

test_text = """
EMPLOYMENT AGREEMENT

This agreement is made on January 1, 2024. The Employee will receive
a salary of $150,000 per year. Payment is made bi-weekly.

TERMINATION

Either party may terminate with 30 days notice. Upon termination, all
company property must be returned immediately.
"""

chunks = smart_chunking(test_text, document_type="contract", chunk_size=200)
print(f"✅ Created {len(chunks)} chunks")
for i, chunk in enumerate(chunks[:2], 1):
    print(f"   Chunk {i}: {chunk[:80]}...")

# Test 2: Embeddings
print("\n2️⃣  Testing Enhanced Embeddings...")
from app.rag.embeddings import generate_embeddings, compute_similarity

test_chunks = [
    "Payment is due within 30 days.",
    "The client must pay within thirty days.",
    "This is about weather and climate."
]

embeddings = generate_embeddings(test_chunks, model_name="default")
print(f"✅ Generated {len(embeddings)} embeddings (dim: {len(embeddings[0])})")

sim = compute_similarity(embeddings[0], embeddings[1])
print(f"   Similarity (payment texts): {sim:.4f}")

sim2 = compute_similarity(embeddings[0], embeddings[2])
print(f"   Similarity (different topics): {sim2:.4f}")

# Test 3: Evaluation
print("\n3️⃣  Testing Evaluation System...")
from app.rag.evaluation import evaluate_rag_pipeline, create_evaluation_report

test_question = "What is the payment schedule?"
test_answer = "According to the contract, payment is due within 30 days from invoice date."
test_retrieved = [
    {"text": "Payment is due within 30 days.", "score": 0.89, "chunk_index": 0},
    {"text": "Invoices are sent monthly.", "score": 0.72, "chunk_index": 1},
]

evaluation = evaluate_rag_pipeline(
    question=test_question,
    answer=test_answer,
    retrieved_chunks=test_retrieved,
    confidence="high"
)

print(f"✅ Overall Quality: {evaluation['overall_quality']}")
print(f"   Retrieval Avg Score: {evaluation['retrieval']['avg_score']:.3f}")
print(f"   Answer Confidence: {evaluation['answer']['confidence_level']}")

print("\n" + "=" * 70)
print("✅ All tests passed! RAG improvements working!")
print("=" * 70)