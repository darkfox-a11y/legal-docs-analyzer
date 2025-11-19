"""
Embeddings generation with smart model caching.

Model downloads once, then cached forever!
Location: ~/.cache/torch/sentence_transformers/
"""

from sentence_transformers import SentenceTransformer
from typing import List
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = 'all-MiniLM-L6-v2'
CACHE_DIR = Path(os.getenv(
    'SENTENCE_TRANSFORMERS_HOME',
    str(Path.home() / '.cache' / 'torch' / 'sentence_transformers')
))


def check_model_cached() -> bool:
    """
    Check if model is already downloaded and cached.
    
    Returns:
        True if model exists in cache, False otherwise
    """
    # Check common cache locations
    model_path = CACHE_DIR / f"sentence-transformers_{MODEL_NAME}"
    alternative_path = CACHE_DIR / MODEL_NAME
    
    return model_path.exists() or alternative_path.exists()


def load_embedding_model() -> SentenceTransformer:
    """
    Load embedding model with smart caching.
    
    - First run: Downloads ~80MB, saves to cache
    - All future runs: Loads from cache instantly!
    
    Returns:
        Loaded SentenceTransformer model
    """
    is_cached = check_model_cached()
    
    if is_cached:
        print(f"✅ Loading embedding model from cache...")
        print(f"   Cache location: {CACHE_DIR}")
    else:
        print(f"📥 Downloading embedding model (first time only, ~80MB)...")
        print(f"   Model: {MODEL_NAME}")
        print(f"   Saving to: {CACHE_DIR}")
        print(f"   Future runs will load from cache instantly!")
    
    # Load model (downloads if not cached)
    model = SentenceTransformer(MODEL_NAME)
    
    if not is_cached:
        print(f"✅ Model downloaded and cached!")
        print(f"   All future runs will be instant!")
    else:
        print(f"✅ Model loaded from cache!")
    
    print()
    return model


# Load model ONCE when module is imported
# This happens only when the Python process starts
print("="*60)
print("🧠 Initializing Embeddings Module")
print("="*60)
model = load_embedding_model()
print("="*60)
print("✅ Embeddings module ready!")
print("="*60)
print()


def generate_embedding(text: str) -> List[float]:
    """
    Convert single text to embedding vector.
    
    The model is already loaded in memory, so this is FAST!
    
    Args:
        text: Text to convert (sentence, paragraph, or chunk)
        
    Returns:
        List of 384 floats representing the meaning
        
    Example:
        >>> embedding = generate_embedding("Payment is due in 30 days")
        >>> len(embedding)
        384
    """
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Convert multiple texts to embeddings (BATCH - much faster!).
    
    Processes all texts in parallel using the pre-loaded model.
    
    Args:
        texts: List of texts to convert
        
    Returns:
        List of embeddings (each is 384 floats)
        
    Example:
        >>> texts = ["Text 1", "Text 2", "Text 3"]
        >>> embeddings = generate_embeddings(texts)
        >>> len(embeddings)
        3
        >>> len(embeddings[0])
        384
    """
    logger.info(f"🔄 Generating embeddings for {len(texts)} texts...")
    
    # Batch encode (parallel processing - fast!)
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=len(texts) > 10  # Show progress for large batches
    )
    
    logger.info(f"✅ Generated {len(embeddings)} embeddings")
    
    return embeddings.tolist()


def get_embedding_dimension() -> int:
    """
    Get dimension of embedding vectors.
    
    Returns:
        384 (for all-MiniLM-L6-v2)
    """
    return model.get_sentence_embedding_dimension()


# Testing
"""if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 Testing Embeddings Module")
    print("="*60 + "\n")
    
    # Test 1: Single embedding
    print("TEST 1: Single text embedding\n")
    test_text = "Payment terms are net 30 days."
    embedding = generate_embedding(test_text)
    
    print(f"Text: '{test_text}'")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 10 values: {[f'{x:.4f}' for x in embedding[:10]]}")
    
    print("\n" + "="*60 + "\n")
    
    # Test 2: Batch embeddings
    print("TEST 2: Batch embeddings (multiple texts)\n")
    test_texts = [
        "Payment is due within 30 days.",
        "Late fees apply after 45 days.",
        "Termination requires 60 days notice."
    ]
    
    embeddings = generate_embeddings(test_texts)
    
    print(f"Number of texts: {len(test_texts)}")
    print(f"Number of embeddings generated: {len(embeddings)}")
    print(f"Embedding dimension: {get_embedding_dimension()}")
    
    print("\nSample embeddings:")
    for i, (text, emb) in enumerate(zip(test_texts, embeddings), 1):
        print(f"\n  Text {i}: '{text}'")
        print(f"  First 5 values: {[f'{x:.4f}' for x in emb[:5]]}")
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("✅ Model is cached and ready for production!")
    print("="*60 + "\n")"""