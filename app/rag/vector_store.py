"""
Qdrant vector store integration.

What this does:
1. Connects to Qdrant Cloud (vector database)
2. Stores document chunks with their embeddings
3. Searches for similar chunks (semantic search!)
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Tuple
import logging
from uuid import uuid4

from app.config import settings
from app.rag.embeddings import get_embedding_dimension, generate_embedding

logger = logging.getLogger(__name__)

# Initialize Qdrant client
print("Connecting to Qdrant Cloud...")
client = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key,
)
print("✅ Connected to Qdrant!")


def create_collection_if_not_exists(collection_name: str = None):
    """
    Create Qdrant collection if it doesn't exist.
    """
    if collection_name is None:
        collection_name = settings.qdrant_collection_name
    
    # Check if collection exists
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    
    if collection_name in collection_names:
        print(f"✅ Collection '{collection_name}' already exists")
        return
    
    # Import PayloadSchema classes
    from qdrant_client.models import PayloadSchemaType
    
    # Create new collection
    print(f"🔨 Creating collection '{collection_name}'...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=get_embedding_dimension(),
            distance=Distance.COSINE
        )
    )
    
    # Create index for document_id (required for filtering!)
    print(f"🔨 Creating index for document_id...")
    client.create_payload_index(
        collection_name=collection_name,
        field_name="document_id",
        field_schema=PayloadSchemaType.INTEGER
    )
    
    print(f"✅ Collection '{collection_name}' created with index!")


def store_document_chunks(
    document_id: int,
    chunks: List[str],
    embeddings: List[List[float]],
    collection_name: str = None
) -> int:
    """
    Store document chunks with their embeddings in Qdrant.
    
    How it works:
    1. Takes chunks of text and their embeddings
    2. Stores each chunk with metadata (document_id, chunk_index)
    3. Returns number of chunks stored
    
    Args:
        document_id: ID of the document these chunks belong to
        chunks: List of text chunks
        embeddings: List of embeddings (one per chunk)
        collection_name: Name of collection (default from settings)
        
    Returns:
        Number of chunks stored
        
    Example:
        chunks = ["Chunk 1 text", "Chunk 2 text"]
        embeddings = [[0.1, 0.2, ...], [0.3, 0.4, ...]]
        count = store_document_chunks(1, chunks, embeddings)
        # Stores 2 chunks in Qdrant
    """
    if collection_name is None:
        collection_name = settings.qdrant_collection_name
    
    # Ensure collection exists
    create_collection_if_not_exists(collection_name)
    
    logger.info(f"📝 Storing {len(chunks)} chunks for document {document_id}")
    
    # Prepare points (Qdrant's term for vectors with metadata)
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point = PointStruct(
            id=str(uuid4()),  # Unique ID for this chunk
            vector=embedding,  # The 384 numbers
            payload={  # Metadata we can filter/search by
                "document_id": document_id,
                "chunk_index": i,
                "text": chunk,
                "chunk_length": len(chunk)
            }
        )
        points.append(point)
    
    # Upload to Qdrant (batch upload - fast!)
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    
    logger.info(f"✅ Stored {len(points)} chunks in Qdrant")
    
    return len(points)


def search_similar_chunks(
    query: str,
    document_id: int = None,
    top_k: int = 5,
    collection_name: str = None
) -> List[Dict]:
    """
    Search for chunks similar to the query.
    
    This is the MAGIC of RAG! 🪄
    
    How it works:
    1. Convert query to embedding
    2. Find chunks with similar embeddings (cosine similarity)
    3. Return most relevant chunks
    
    Args:
        query: User's question
        document_id: Optional - search only in specific document
        top_k: How many results to return
        collection_name: Name of collection (default from settings)
        
    Returns:
        List of dictionaries with chunk text and similarity score
        
    Example:
        query = "What are the payment terms?"
        results = search_similar_chunks(query, document_id=1, top_k=3)
        
        # Returns top 3 most relevant chunks:
        [
            {"text": "Payment is due in 30 days...", "score": 0.89},
            {"text": "Late fees apply after...", "score": 0.76},
            {"text": "Contact us for payment...", "score": 0.65}
        ]
    """
    if collection_name is None:
        collection_name = settings.qdrant_collection_name
    
    logger.info(f"🔍 Searching for: '{query}'")
    
    # Convert query to embedding
    query_embedding = generate_embedding(query)
    
    # Build filter if document_id provided
    query_filter = None
    if document_id is not None:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id)
                )
            ]
        )
    
    # Search in Qdrant
    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        query_filter=query_filter,
        limit=top_k
    )
    
    # Format results
    results = []
    for hit in search_results:
        results.append({
            "text": hit.payload["text"],
            "score": hit.score,  # Similarity score (0-1, higher = more similar)
            "document_id": hit.payload["document_id"],
            "chunk_index": hit.payload["chunk_index"]
        })
    
    logger.info(f"✅ Found {len(results)} relevant chunks")
    
    return results


def delete_document_chunks(document_id: int, collection_name: str = None):
    """
    Delete all chunks for a specific document.
    
    Used when user deletes a document.
    
    Args:
        document_id: ID of document to delete chunks for
        collection_name: Name of collection (default from settings)
    """
    if collection_name is None:
        collection_name = settings.qdrant_collection_name
    
    logger.info(f"🗑️  Deleting chunks for document {document_id}")
    
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    
    # Delete all points where document_id matches
    client.delete(
        collection_name=collection_name,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id)
                )
            ]
        )
    )
    
    logger.info(f"✅ Deleted chunks for document {document_id}")


# Test when run directly
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 Testing Qdrant Integration")
    print("="*60 + "\n")
    
    # Test 1: Create collection
    print("TEST 1: Create collection\n")
    create_collection_if_not_exists()
    
    print("\n" + "="*60 + "\n")
    
    # Test 2: Store and search
    print("TEST 2: Store and search chunks\n")
    
    from app.rag.embeddings import generate_embeddings
    
    # Sample chunks
    test_chunks = [
        "Payment is due within 30 days of invoice date.",
        "Late fees of 1.5% per month apply after 45 days.",
        "This agreement can be terminated with 60 days written notice.",
        "All disputes will be resolved through binding arbitration.",
        "The contract is governed by the laws of California."
    ]
    
    # Generate embeddings
    print("Generating embeddings for test chunks...")
    test_embeddings = generate_embeddings(test_chunks)
    
    # Store in Qdrant
    print("Storing chunks in Qdrant...")
    count = store_document_chunks(
        document_id=999,  # Test document ID
        chunks=test_chunks,
        embeddings=test_embeddings
    )
    print(f"✅ Stored {count} chunks")
    
    print("\n" + "="*60 + "\n")
    
    # Test 3: Search
    print("TEST 3: Semantic search\n")
    
    query = "What are the payment terms?"
    print(f"Query: '{query}'\n")
    
    results = search_similar_chunks(query, document_id=999, top_k=3)
    
    print("Search results:")
    for i, result in enumerate(results, 1):
        print(f"\n  Result {i}:")
        print(f"  Text: {result['text'][:60]}...")
        print(f"  Score: {result['score']:.4f}")
    
    print("\n" + "="*60)
    
    # Cleanup
    #print("\n🧹 Cleaning up test data...")
    #delete_document_chunks(999)
    print("✅ Test complete!")
    print("="*60 + "\n")