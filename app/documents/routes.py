"""
Document management API routes.
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from typing import List
import os
import uuid
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

from app.database import get_db
from app.auth.dependencies import get_current_user
from app.auth.models import User
from app.documents import models, schemas, processing

router = APIRouter()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/upload", response_model=schemas.DocumentUpload, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload a document (PDF or DOCX)."""
    
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided"
        )
    
    is_valid, file_type = processing.validate_file_type(file.filename)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type. Supported: PDF, DOCX, DOC"
        )
    
    unique_filename = f"{uuid.uuid4()}.{file_type}"
    file_path = UPLOAD_DIR / unique_filename
    
    contents = await file.read()
    
    with open(file_path, "wb") as f:
        f.write(contents)
    
    document = models.Document(
        user_id=current_user.id,
        filename=unique_filename,
        original_filename=file.filename,
        file_type=file_type,
        file_size=len(contents),
        status="processing"
    )
    
    db.add(document)
    db.commit()
    db.refresh(document)
    
    # Extract text
    try:
        extracted_text, page_count, error = processing.process_document(str(file_path), file_type)
        
        if error:
            document.status = "error"
            document.error_message = error
        else:
            document.extracted_text = extracted_text
            document.page_count = page_count
            document.status = "ready"
        
        db.commit()
        db.refresh(document)
    except Exception as e:
        document.status = "error"
        document.error_message = str(e)
        db.commit()
    
    # RAG Processing (NEW!)
    try:
        from app.rag.chunking import simple_chunk_by_sentences
        from app.rag.embeddings import generate_embeddings
        from app.rag.vector_store import store_document_chunks
        
        # Only process if text extraction succeeded
        if document.status == "ready" and document.extracted_text:
            logger.info(f"🧩 Processing document {document.id} for RAG...")
            
            # Step 1: Chunk the text
            chunks = simple_chunk_by_sentences(
                document.extracted_text,
                sentences_per_chunk=5
            )
            logger.info(f"✅ Created {len(chunks)} chunks")
            
            # Step 2: Generate embeddings
            embeddings = generate_embeddings(chunks)
            logger.info(f"✅ Generated {len(embeddings)} embeddings")
            
            # Step 3: Store in Qdrant
            count = store_document_chunks(
                document_id=document.id,
                chunks=chunks,
                embeddings=embeddings
            )
            logger.info(f"✅ Stored {count} chunks in Qdrant")
            
    except Exception as e:
        logger.error(f"❌ RAG processing error: {e}")
        # Don't fail the upload if RAG fails
    
    return document


@router.get("/", response_model=List[schemas.DocumentList])
async def list_documents(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all documents for the current user."""
    documents = db.query(models.Document).filter(
        models.Document.user_id == current_user.id
    ).order_by(models.Document.created_at.desc()).all()
    
    return documents


@router.get("/{document_id}", response_model=schemas.DocumentResponse)
async def get_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get full details of a specific document."""
    document = db.query(models.Document).filter(
        models.Document.id == document_id,
        models.Document.user_id == current_user.id
    ).first()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    return document


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a document."""
    document = db.query(models.Document).filter(
        models.Document.id == document_id,
        models.Document.user_id == current_user.id
    ).first()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    file_path = UPLOAD_DIR / document.filename
    if file_path.exists():
        os.remove(file_path)
    
    db.delete(document)
    db.commit()
    
    return None

@router.post("/ask")
async def ask_question(
    question: str,
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Ask a question about a document using RAG.
    
    This uses:
    1. Vector search to find relevant chunks
    2. Gemini AI to generate intelligent answers
    
    Args:
        question: User's question (e.g., "What are the payment terms?")
        document_id: ID of document to search in
        
    Returns:
        {
            "question": "What are the payment terms?",
            "answer": "Payment is due within 30 days...",
            "sources": [...],
            "confidence": "high"
        }
    """
    from app.rag.qa import answer_query
    
    # Check if document exists and belongs to user
    document = db.query(models.Document).filter(
        models.Document.id == document_id,
        models.Document.user_id == current_user.id
    ).first()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found or you don't have access to it"
        )
    
    # Check if document is ready
    if document.status != "ready":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Document is not ready yet. Status: {document.status}"
        )
    
    # Get answer using RAG
    try:
        result = answer_query(
            query=question,
            document_id=document_id,
            top_k=3  # Return top 3 most relevant chunks
        )
        return result
        
    except Exception as e:
        logger.error(f"❌ Error answering question: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating answer: {str(e)}"
        )