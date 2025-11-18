"""
Database models for document management.
"""

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, func
from sqlalchemy.orm import relationship
from app.database import Base


class Document(Base):
    """Document model for storing document metadata."""
    
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)  # pdf, docx, etc.
    file_size = Column(Integer, nullable=False)  # in bytes
    page_count = Column(Integer, nullable=True)
    extracted_text = Column(Text, nullable=True)  # Full text content
    status = Column(String(50), default="processing")  # processing, ready, error
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationship to user
    # user = relationship("User", back_populates="documents")
    
    def __repr__(self):
        return f"<Document(id={self.id}, filename='{self.filename}', user_id={self.user_id})>"