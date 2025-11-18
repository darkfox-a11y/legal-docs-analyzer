"""
Pydantic schemas for document management.
These define the structure of data for API requests and responses.
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class DocumentUpload(BaseModel):
    """
    Schema for IMMEDIATE response after file upload.
    
    Returns basic info quickly (< 1 second) before processing starts.
    Status will always be "processing" at this stage.
    
    Use case: POST /documents/upload
    
    Example response:
    {
        "id": 1,
        "filename": "a1b2c3d4-e5f6-7890.pdf",  # UUID we generated
        "original_filename": "my-contract.pdf",  # What user uploaded
        "file_type": "pdf",
        "file_size": 524288,  # bytes
        "status": "processing",  # Still processing!
        "created_at": "2024-11-18T19:30:00Z"
    }
    """
    id: int  # Document ID (1, 2, 3...)
    filename: str  # Our generated unique filename (UUID)
    original_filename: str  # User's original filename
    file_type: str  # File extension (pdf, docx, doc)
    file_size: int  # Size in bytes
    status: str  # Always "processing" on upload
    created_at: datetime  # When uploaded
    
    class Config:
        from_attributes = True  # Allows conversion from SQLAlchemy model


class DocumentResponse(BaseModel):
    """
    Schema for FULL document details after processing.
    
    Returns complete information including processing results.
    Status could be "ready", "processing", or "error".
    
    Use case: GET /documents/{id}
    
    Example response:
    {
        "id": 1,
        "user_id": 42,  # Who owns this
        "filename": "a1b2c3d4-e5f6-7890.pdf",
        "original_filename": "my-contract.pdf",
        "file_type": "pdf",
        "file_size": 524288,
        "page_count": 15,  # Now we know! (after processing)
        "status": "ready",  # Processing complete
        "error_message": null,  # No errors
        "created_at": "2024-11-18T19:30:00Z",
        "updated_at": "2024-11-18T19:30:45Z"  # When processing finished
    }
    """
    id: int  # Document ID
    user_id: int  # Owner's user ID
    filename: str  # Stored filename (UUID)
    original_filename: str  # User's original filename
    file_type: str  # File extension
    file_size: int  # Size in bytes
    page_count: Optional[int]  # Number of pages (None if still processing or failed)
    status: str  # "processing", "ready", or "error"
    error_message: Optional[str]  # Error details if processing failed (None if success)
    created_at: datetime  # When uploaded
    updated_at: Optional[datetime]  # When last updated (None if never updated)
    
    class Config:
        from_attributes = True


class DocumentList(BaseModel):
    """
    Schema for LISTING multiple documents efficiently.
    
    Returns only essential fields to keep response small and fast.
    Good for showing many documents at once.
    
    Use case: GET /documents (list all user's documents)
    
    Example response (array):
    [
        {
            "id": 1,
            "original_filename": "contract.pdf",
            "file_type": "pdf",
            "file_size": 524288,
            "page_count": 15,
            "status": "ready",
            "created_at": "2024-11-18T19:30:00Z"
        },
        {
            "id": 2,
            "original_filename": "agreement.docx",
            "file_type": "docx",
            "file_size": 102400,
            "page_count": null,  # Still processing
            "status": "processing",
            "created_at": "2024-11-18T19:35:00Z"
        }
    ]
    """
    id: int  # Document ID
    original_filename: str  # User's filename (not internal UUID)
    file_type: str  # File extension
    file_size: int  # Size in bytes
    page_count: Optional[int]  # Number of pages (None if processing)
    status: str  # Current status
    created_at: datetime  # When uploaded
    
    # Note: Excluded fields for efficiency:
    # - user_id (all documents belong to requesting user)
    # - filename (internal detail, not needed)
    # - error_message (too detailed for list view)
    # - updated_at (created_at is enough)
    
    class Config:
        from_attributes = True


"""
SUMMARY OF DIFFERENCES:

┌──────────────────┬────────────────┬──────────────────┬──────────────┐
│ Field            │ DocumentUpload │ DocumentResponse │ DocumentList │
├──────────────────┼────────────────┼──────────────────┼──────────────┤
│ When returned    │ Immediately    │ On request       │ List all     │
│ Processing done  │ No (starting)  │ Maybe            │ Varies       │
│ id               │ ✅             │ ✅               │ ✅           │
│ user_id          │ ❌             │ ✅               │ ❌           │
│ filename         │ ✅ (UUID)      │ ✅ (UUID)        │ ❌           │
│ original_name    │ ✅             │ ✅               │ ✅           │
│ file_type        │ ✅             │ ✅               │ ✅           │
│ file_size        │ ✅             │ ✅               │ ✅           │
│ page_count       │ ❌ (unknown)   │ ✅               │ ✅           │
│ status           │ ✅ (always     │ ✅ (current)     │ ✅           │
│                  │  "processing") │                  │              │
│ error_message    │ ❌             │ ✅               │ ❌           │
│ created_at       │ ✅             │ ✅               │ ✅           │
│ updated_at       │ ❌             │ ✅               │ ❌           │
└──────────────────┴────────────────┴──────────────────┴──────────────┘

TYPICAL FLOW:
1. Upload file → Returns DocumentUpload (status: "processing")
2. [Background: Extract text, count pages, create embeddings]
3. Get details → Returns DocumentResponse (status: "ready", page_count: 15)
4. List all → Returns array of DocumentList (efficient summary)
"""