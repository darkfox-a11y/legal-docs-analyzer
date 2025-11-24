"""
Advanced document chunking strategies for optimal RAG performance.
"""

import re
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


def semantic_chunk_with_overlap(
    text: str,
    chunk_size: int = 500,
    overlap_size: int = 100,
    min_chunk_size: int = 50
) -> List[str]:
    """
    Advanced chunking with overlap for better context retention.
    
    Benefits:
    - Overlap prevents losing context at chunk boundaries
    - Semantic splitting on sentences (not random cuts)
    - Configurable sizes for different document types
    
    Args:
        text: Input text to chunk
        chunk_size: Target characters per chunk
        overlap_size: Overlap between chunks (prevents context loss)
        min_chunk_size: Minimum chunk size to keep
    
    Returns:
        List of text chunks with overlap
    """
    if not text or len(text.strip()) == 0:
        logger.warning("Empty text provided for chunking")
        return []
    
    # Split into sentences first
    sentences = split_into_sentences(text)
    
    if not sentences:
        logger.warning("No sentences found in text")
        return []
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        # If adding this sentence exceeds chunk_size
        if current_length + sentence_length > chunk_size and current_chunk:
            # Save current chunk
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= min_chunk_size:
                chunks.append(chunk_text)
            
            # Create overlap: keep last few sentences
            overlap_sentences = []
            overlap_length = 0
            for s in reversed(current_chunk):
                if overlap_length + len(s) <= overlap_size:
                    overlap_sentences.insert(0, s)
                    overlap_length += len(s)
                else:
                    break
            
            # Start new chunk with overlap
            current_chunk = overlap_sentences + [sentence]
            current_length = overlap_length + sentence_length
        else:
            # Add to current chunk
            current_chunk.append(sentence)
            current_length += sentence_length
    
    # Add final chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        if len(chunk_text) >= min_chunk_size:
            chunks.append(chunk_text)
    
    logger.info(f"✅ Created {len(chunks)} semantic chunks with overlap (avg size: {sum(len(c) for c in chunks) // len(chunks) if chunks else 0} chars)")
    return chunks


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences intelligently.
    Handles abbreviations, numbers, legal citations, etc.
    """
    # Protect common abbreviations
    abbreviations = {
        'Dr.': 'Dr<DOT>',
        'Mr.': 'Mr<DOT>',
        'Mrs.': 'Mrs<DOT>',
        'Ms.': 'Ms<DOT>',
        'Sr.': 'Sr<DOT>',
        'Jr.': 'Jr<DOT>',
        'Inc.': 'Inc<DOT>',
        'Ltd.': 'Ltd<DOT>',
        'Corp.': 'Corp<DOT>',
        'Co.': 'Co<DOT>',
        'etc.': 'etc<DOT>',
        'vs.': 'vs<DOT>',
        'e.g.': 'eg<DOT>',
        'i.e.': 'ie<DOT>',
        'Ph.D.': 'PhD<DOT>',
        'U.S.': 'US<DOT>',
        'U.K.': 'UK<DOT>',
        'No.': 'No<DOT>',
        'Vol.': 'Vol<DOT>',
        'Sec.': 'Sec<DOT>',
        'Art.': 'Art<DOT>',
        'Fig.': 'Fig<DOT>',
        'Ref.': 'Ref<DOT>',
        'et al.': 'etal<DOT>',
    }
    
    # Replace abbreviations
    for abbr, placeholder in abbreviations.items():
        text = text.replace(abbr, placeholder)
    
    # Split on sentence boundaries
    # Matches: . ! ? followed by space and capital letter or number
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text)
    
    # Restore abbreviations and clean
    sentences = [
        s.replace('<DOT>', '.').strip()
        for s in sentences
        if s.strip()
    ]
    
    return sentences


def hierarchical_chunking(
    text: str,
    max_chunk_size: int = 1000
) -> List[Tuple[str, str]]:
    """
    Chunk by document structure (sections, paragraphs).
    Best for structured legal documents.
    
    Returns: List of (section_title, content) tuples
    """
    # Common section patterns in legal documents
    section_patterns = [
        r'^#+\s+(.+)$',  # Markdown headers
        r'^(\d+\.?\s+[A-Z][A-Za-z\s]+)$',  # "1. SECTION NAME" or "1 SECTION NAME"
        r'^([A-Z][A-Z\s]{3,}):?$',  # ALL CAPS headers
        r'^(Article\s+[IVX\d]+[:\.]?\s*.*)$',  # Article I, Article 1
        r'^(Section\s+\d+[:\.]?\s*.*)$',  # Section 1
        r'^(WHEREAS.*)$',  # Contract clauses
        r'^(NOW THEREFORE.*)$',
        r'^(SCHEDULE\s+[A-Z0-9]+.*)$',  # Schedules/Appendices
        r'^(EXHIBIT\s+[A-Z0-9]+.*)$',
        r'^(APPENDIX\s+[A-Z0-9]+.*)$',
    ]
    
    chunks = []
    current_section = "Preamble"
    current_content = []
    
    paragraphs = text.split('\n\n')
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # Check if this is a section header
        is_header = False
        for pattern in section_patterns:
            match = re.match(pattern, para, re.MULTILINE | re.IGNORECASE)
            if match:
                # Save previous section
                if current_content:
                    content = '\n\n'.join(current_content)
                    if len(content) > 50:  # Minimum content length
                        # Split large sections
                        if len(content) > max_chunk_size:
                            sub_chunks = semantic_chunk_with_overlap(
                                content, 
                                chunk_size=max_chunk_size,
                                overlap_size=100
                            )
                            for i, sub_chunk in enumerate(sub_chunks):
                                chunks.append((f"{current_section} (Part {i+1})", sub_chunk))
                        else:
                            chunks.append((current_section, content))
                
                # Start new section
                current_section = para[:100]  # Limit header length
                current_content = []
                is_header = True
                break
        
        if not is_header:
            current_content.append(para)
    
    # Add final section
    if current_content:
        content = '\n\n'.join(current_content)
        if len(content) > 50:
            if len(content) > max_chunk_size:
                sub_chunks = semantic_chunk_with_overlap(
                    content, 
                    chunk_size=max_chunk_size,
                    overlap_size=100
                )
                for i, sub_chunk in enumerate(sub_chunks):
                    chunks.append((f"{current_section} (Part {i+1})", sub_chunk))
            else:
                chunks.append((current_section, content))
    
    logger.info(f"✅ Created {len(chunks)} hierarchical chunks from document structure")
    return chunks


def smart_chunking(
    text: str,
    document_type: str = "general",
    chunk_size: int = 500,
    overlap_size: int = 100
) -> List[str]:
    """
    Automatically choose best chunking strategy based on document type.
    
    Document Types:
    - "contract": Legal contracts (uses hierarchical)
    - "legal": Legal documents (uses hierarchical)
    - "report": Reports/articles (uses semantic with overlap)
    - "general": Default (uses semantic with overlap)
    
    Args:
        text: Document text
        document_type: Type of document
        chunk_size: Target size for chunks
        overlap_size: Overlap between chunks
    
    Returns:
        Optimally chunked text
    """
    logger.info(f"📄 Smart chunking for document type: '{document_type}' ({len(text)} chars)")
    
    if not text or len(text.strip()) == 0:
        logger.warning("Empty text provided")
        return []
    
    # Normalize document type
    doc_type = document_type.lower().strip()
    
    # Check if document has clear structure
    has_structure = bool(re.search(
        r'(^#+\s+|^\d+\.\s+[A-Z]|^[A-Z][A-Z\s]{3,}:|^Article\s+|^Section\s+|^WHEREAS)',
        text,
        re.MULTILINE
    ))
    
    if doc_type in ["contract", "legal", "agreement"] or has_structure:
        # Use hierarchical chunking for structured legal documents
        logger.info("Using hierarchical chunking (structured document)")
        hierarchical = hierarchical_chunking(text, max_chunk_size=chunk_size * 2)
        
        # Flatten: just return content, prepend section name
        chunks = []
        for section, content in hierarchical:
            chunk_text = f"[{section}]\n{content}"
            chunks.append(chunk_text)
        
        return chunks
    
    else:
        # Use semantic chunking with overlap for other documents
        logger.info(f"Using semantic chunking with overlap (chunk_size={chunk_size}, overlap={overlap_size})")
        return semantic_chunk_with_overlap(
            text,
            chunk_size=chunk_size,
            overlap_size=overlap_size
        )


# Backward compatibility - keep old function name
def simple_chunk_by_sentences(text: str, sentences_per_chunk: int = 5) -> List[str]:
    """
    Legacy simple chunking - DEPRECATED
    Use smart_chunking() or semantic_chunk_with_overlap() instead
    """
    logger.warning("simple_chunk_by_sentences is deprecated, use smart_chunking instead")
    return semantic_chunk_with_overlap(text, chunk_size=500, overlap_size=100)


# Test when run directly
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 Testing Advanced Chunking Strategies")
    print("="*70 + "\n")
    
    test_contract = """
    EMPLOYMENT AGREEMENT
    
    This Employment Agreement ("Agreement") is entered into as of January 1, 2024.
    
    1. POSITION AND DUTIES
    
    The Employee shall serve as Senior Software Engineer. The Employee agrees to 
    perform all duties assigned. The Employee shall report to the CTO.
    
    2. COMPENSATION
    
    The Company shall pay Employee a base salary of $150,000 per year. Salary 
    shall be paid bi-weekly. The Employee is eligible for annual bonuses.
    
    3. BENEFITS
    
    The Employee shall be entitled to health insurance. The Company provides 
    dental and vision coverage. Employee receives 15 days PTO per year.
    
    4. TERMINATION
    
    Either party may terminate this Agreement with 30 days notice. Upon termination,
    all company property must be returned. Final payment will be made within 14 days.
    """
    
    print("\n1️⃣  SEMANTIC CHUNKING WITH OVERLAP")
    print("-" * 70)
    semantic = semantic_chunk_with_overlap(
        test_contract, 
        chunk_size=200, 
        overlap_size=50
    )
    for i, chunk in enumerate(semantic, 1):
        print(f"\nChunk {i} ({len(chunk)} chars):")
        print(chunk[:150] + "..." if len(chunk) > 150 else chunk)
    
    print("\n\n2️⃣  HIERARCHICAL CHUNKING")
    print("-" * 70)
    hierarchical = hierarchical_chunking(test_contract)
    for section, content in hierarchical:
        print(f"\n📋 Section: {section}")
        print(f"Content ({len(content)} chars): {content[:100]}...")
    
    print("\n\n3️⃣  SMART CHUNKING (contract type)")
    print("-" * 70)
    smart = smart_chunking(test_contract, document_type="contract", chunk_size=300)
    for i, chunk in enumerate(smart, 1):
        print(f"\nChunk {i} ({len(chunk)} chars):")
        print(chunk[:150] + "..." if len(chunk) > 150 else chunk)
    
    print("\n\n" + "="*70)
    print("✅ Chunking tests complete!")
    print("="*70 + "\n")