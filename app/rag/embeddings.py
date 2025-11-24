"""
Enhanced embedding generation with multiple model options and caching.
"""

from sentence_transformers import SentenceTransformer
from typing import List, Optional
import numpy as np
import logging
import hashlib
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# Cache directory for embeddings
CACHE_DIR = Path("embeddings_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Available embedding models
EMBEDDING_MODELS = {
    "default": "all-MiniLM-L6-v2",  # Fast, good quality, 384 dims
    "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",  # 384 dims, multi-language
    "legal": "nlpaueb/legal-bert-base-uncased",  # Legal domain
    "high_quality": "all-mpnet-base-v2",  # Best quality, slower, 768 dims
}

# Global model instance (lazy loaded)
_embedding_model = None
_current_model_name = None


def get_embedding_model(model_name: str = "default") -> SentenceTransformer:
    """
    Get or create embedding model instance (singleton pattern).
    Models are cached after first load.
    """
    global _embedding_model, _current_model_name
    
    # Get actual model name
    actual_model = EMBEDDING_MODELS.get(model_name, EMBEDDING_MODELS["default"])
    
    # Return cached model if same
    if _embedding_model is not None and _current_model_name == actual_model:
        return _embedding_model
    
    # Load new model
    logger.info(f"📥 Loading embedding model: {actual_model}")
    _embedding_model = SentenceTransformer(actual_model)
    _current_model_name = actual_model
    logger.info(f"✅ Model loaded (dimension: {_embedding_model.get_sentence_embedding_dimension()})")
    
    return _embedding_model


def generate_embeddings(
    texts: List[str],
    model_name: str = "default",
    batch_size: int = 32,
    show_progress: bool = True,
    normalize: bool = True,
    use_cache: bool = False
) -> List[List[float]]:
    """
    Generate embeddings for a list of texts with advanced features.
    
    Args:
        texts: List of text strings
        model_name: Model to use ("default", "multilingual", "legal", "high_quality")
        batch_size: Batch size for processing
        show_progress: Show progress bar
        normalize: Normalize embeddings to unit length
        use_cache: Use cached embeddings if available
    
    Returns:
        List of embedding vectors
    """
    if not texts:
        logger.warning("Empty text list provided")
        return []
    
    logger.info(f"🧠 Generating embeddings for {len(texts)} texts using '{model_name}' model")
    
    # Check cache if enabled
    if use_cache:
        cached = try_load_from_cache(texts, model_name)
        if cached is not None:
            logger.info("✅ Loaded embeddings from cache")
            return cached
    
    # Get model
    model = get_embedding_model(model_name)
    
    # Generate embeddings
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=normalize
    )
    
    # Convert to list of lists
    embeddings_list = embeddings.tolist()
    
    # Save to cache if enabled
    if use_cache:
        save_to_cache(texts, model_name, embeddings_list)
    
    logger.info(f"✅ Generated {len(embeddings_list)} embeddings (dimension: {len(embeddings_list[0])})")
    
    return embeddings_list


def generate_single_embedding(
    text: str,
    model_name: str = "default",
    normalize: bool = True
) -> List[float]:
    """
    Generate embedding for a single text (optimized for queries).
    
    Args:
        text: Single text string
        model_name: Model to use
        normalize: Normalize embedding
    
    Returns:
        Single embedding vector
    """
    if not text or not text.strip():
        logger.warning("Empty text provided")
        return []
    
    model = get_embedding_model(model_name)
    
    embedding = model.encode(
        [text],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=normalize
    )[0]
    
    return embedding.tolist()


def get_embedding_dimension(model_name: str = "default") -> int:
    """Get the dimension of embeddings for a given model."""
    model = get_embedding_model(model_name)
    return model.get_sentence_embedding_dimension()


def compute_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Returns:
        Similarity score (0-1, higher is more similar)
    """
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    
    # Cosine similarity
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    return float(similarity)


def try_load_from_cache(texts: List[str], model_name: str) -> Optional[List[List[float]]]:
    """Try to load embeddings from cache."""
    # Create hash of texts + model
    content_hash = hashlib.md5(
        (json.dumps(texts) + model_name).encode()
    ).hexdigest()
    
    cache_file = CACHE_DIR / f"{content_hash}.json"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            return cached_data['embeddings']
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
    
    return None


def save_to_cache(texts: List[str], model_name: str, embeddings: List[List[float]]):
    """Save embeddings to cache."""
    # Create hash
    content_hash = hashlib.md5(
        (json.dumps(texts) + model_name).encode()
    ).hexdigest()
    
    cache_file = CACHE_DIR / f"{content_hash}.json"
    
    try:
        with open(cache_file, 'w') as f:
            json.dump({
                'texts_count': len(texts),
                'model': model_name,
                'embeddings': embeddings
            }, f)
        logger.info(f"💾 Saved embeddings to cache: {cache_file.name}")
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")


# Backward compatibility
def generate_embedding(text: str) -> List[float]:
    """Legacy function - use generate_single_embedding instead"""
    logger.warning("generate_embedding is deprecated, use generate_single_embedding")
    return generate_single_embedding(text)


# Test when run directly
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 Testing Enhanced Embeddings")
    print("="*70 + "\n")
    
    test_texts = [
        "The payment terms are net 30 days.",
        "Payment must be made within thirty days.",
        "This is a completely different topic about weather.",
    ]
    
    print("1️⃣  Generating embeddings...")
    embeddings = generate_embeddings(test_texts, model_name="default")
    print(f"✅ Generated {len(embeddings)} embeddings")
    print(f"   Dimension: {len(embeddings[0])}")
    
    print("\n2️⃣  Computing similarities...")
    sim_1_2 = compute_similarity(embeddings[0], embeddings[1])
    sim_1_3 = compute_similarity(embeddings[0], embeddings[2])
    
    print(f"   Similarity (text 1 vs text 2 - similar): {sim_1_2:.4f}")
    print(f"   Similarity (text 1 vs text 3 - different): {sim_1_3:.4f}")
    
    print("\n3️⃣  Available models:")
    for name, model_id in EMBEDDING_MODELS.items():
        print(f"   {name}: {model_id}")
    
    print("\n" + "="*70)
    print("✅ Embedding tests complete!")
    print("="*70 + "\n")