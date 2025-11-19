"""
Simple text chunking - breaks documents into smaller pieces.

Why chunking?
- Documents are too large to send to AI all at once
- We split them into small chunks (paragraphs)
- Find relevant chunks for user's question
- Send only relevant chunks to AI (faster, cheaper!)
"""

from typing import List


def simple_chunk_by_sentences(text: str, sentences_per_chunk: int = 5) -> List[str]:
    """
    Split text into chunks by sentences (BEST for legal documents).
    
    How it works:
    1. Split text into sentences (at periods)
    2. Group sentences together (5 sentences = 1 chunk)
    3. Return list of chunks
    
    Example:
        text = "Sentence 1. Sentence 2. Sentence 3. Sentence 4. Sentence 5. Sentence 6."
        chunks = simple_chunk_by_sentences(text, sentences_per_chunk=2)
        
        Result:
        [
            "Sentence 1. Sentence 2.",
            "Sentence 3. Sentence 4.",
            "Sentence 5. Sentence 6."
        ]
    
    Args:
        text: Full document text
        sentences_per_chunk: How many sentences per chunk (default 5 = ~1 paragraph)
        
    Returns:
        List of text chunks
    """
    # Step 1: Split text into sentences
    # Replace ". " with ".|" then split on "|"
    # This keeps the period with the sentence
    sentences = text.replace('. ', '.|').split('|')
    
    # Step 2: Group sentences into chunks
    chunks = []
    
    # Loop through sentences, taking sentences_per_chunk at a time
    for i in range(0, len(sentences), sentences_per_chunk):
        # Take the next sentences_per_chunk sentences
        chunk_sentences = sentences[i:i + sentences_per_chunk]
        
        # Join them back into one string
        chunk = " ".join(chunk_sentences)
        
        # Only add if not empty
        if chunk.strip():
            chunks.append(chunk.strip())
    
    return chunks


def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    """
    Simple chunking by character count (backup method).
    
    How it works:
    1. If text is short (< chunk_size), return as-is
    2. Otherwise, split every chunk_size characters
    
    Example:
        text = "A" * 3000  # 3000 A's
        chunks = chunk_text(text, chunk_size=1000)
        
        Result: ["AAA..." (1000 chars), "AAA..." (1000 chars), "AAA..." (1000 chars)]
    
    Args:
        text: Full document text
        chunk_size: Max characters per chunk (default 1000)
        
    Returns:
        List of text chunks
    """
    # If text is already small, no need to chunk
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    
    # Split text every chunk_size characters
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    
    return chunks


# Test the functions
if __name__ == "__main__":
    print("=== Testing Chunking Module ===\n")
    
    # Sample legal text
    sample_text = """
    This Agreement is entered into on January 1, 2024. 
    The parties agree to the following terms and conditions. 
    Payment shall be made within thirty days of invoice date. 
    Late payments will incur a fee of 1.5% per month. 
    Either party may terminate this agreement with 60 days written notice. 
    All disputes shall be resolved through binding arbitration. 
    This agreement is governed by the laws of California.
    """
    
    print("Original text:")
    print(sample_text)
    print("\n" + "="*60 + "\n")
    
    # Test 1: Chunk by sentences (2 sentences per chunk)
    print("TEST 1: Chunking by sentences (2 sentences per chunk)\n")
    sentence_chunks = simple_chunk_by_sentences(sample_text, sentences_per_chunk=2)
    
    for i, chunk in enumerate(sentence_chunks, 1):
        print(f"Chunk {i}:")
        print(f"  {chunk}")
        print()
    
    print("="*60 + "\n")
    
    # Test 2: Chunk by characters (100 chars per chunk)
    print("TEST 2: Chunking by characters (100 chars per chunk)\n")
    char_chunks = chunk_text(sample_text, chunk_size=100)
    
    for i, chunk in enumerate(char_chunks, 1):
        print(f"Chunk {i} ({len(chunk)} chars):")
        print(f"  {chunk[:50]}...")  # Show first 50 chars
        print()
    
    print("="*60)
    print(f"\n✅ Created {len(sentence_chunks)} sentence chunks")
    print(f"✅ Created {len(char_chunks)} character chunks")