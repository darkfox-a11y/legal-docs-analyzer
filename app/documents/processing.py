"""
Document processing utilities.
Extract text from PDF and DOCX files.

This module handles:
1. PDF text extraction (digital PDFs with actual text)
2. DOCX text extraction (Microsoft Word documents)
3. File type validation
4. Error handling for corrupt or unsupported files

Note: Scanned PDFs (images of documents) are NOT yet supported.
OCR functionality will be added in Phase 2.
"""

import fitz  # PyMuPDF - library for reading PDF files
from docx import Document as DocxDocument  # python-docx for Word files
from typing import Tuple, Optional
import logging

# Create logger for this module (helps with debugging)
logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_path: str) -> Tuple[str, int]:
    """
    Extract text from a DIGITAL PDF file.
    
    This works for PDFs that have actual text (created from Word, Google Docs, etc.)
    It does NOT work for scanned PDFs (which are just images of pages).
    
    How it works:
    1. Opens the PDF file
    2. Loops through each page
    3. Extracts text from each page
    4. Combines all pages into one text string
    5. Returns the text and page count
    
    Args:
        file_path: Full path to the PDF file (e.g., "/uploads/contract.pdf")
        
    Returns:
        Tuple of (extracted_text, page_count)
        Example: ("This is the contract text...", 15)
        
    Raises:
        Exception: If PDF cannot be opened or read
        
    Example usage:
        text, pages = extract_text_from_pdf("/uploads/contract.pdf")
        # text = "Full contract text here..."
        # pages = 15
    """
    try:
        # Step 1: Open the PDF file using PyMuPDF (fitz)
        doc = fitz.open(file_path)
        
        # Step 2: Prepare storage for text from each page
        text_parts = []  # Will hold ["page 1 text", "page 2 text", ...]
        
        # Step 3: Count how many pages the PDF has
        page_count = len(doc)  # e.g., 15 pages
        
        # Step 4: Loop through each page and extract text
        for page_num in range(page_count):  # 0, 1, 2, ... 14
            # Get the page object
            page = doc[page_num]
            
            # Extract all text from this page
            # Returns a string with all text on the page
            text = page.get_text()
            
            # Add this page's text to our collection
            text_parts.append(text)
        
        # Step 5: Close the PDF file (cleanup, good practice)
        doc.close()
        
        # Step 6: Combine all pages into one big text string
        # Join with "\n\n" (two line breaks) between pages for readability
        # Example: "Page 1 text\n\nPage 2 text\n\nPage 3 text"
        full_text = "\n\n".join(text_parts)
        
        # Step 7: Log success (helpful for debugging)
        logger.info(f"Successfully extracted {len(full_text)} characters from {page_count} pages")
        
        # Step 8: Return both the text and page count
        return full_text, page_count
        
    except Exception as e:
        # If anything goes wrong (corrupt PDF, wrong format, etc.)
        # Log the error for debugging
        logger.error(f"Error extracting text from PDF: {e}")
        
        # Raise a new exception with helpful error message
        raise Exception(f"Failed to extract text from PDF: {str(e)}")


def extract_text_from_docx(file_path: str) -> Tuple[str, int]:
    """
    Extract text from a DOCX file (Microsoft Word document).
    
    How it works:
    1. Opens the DOCX file
    2. Loops through each paragraph
    3. Combines paragraphs into one text string
    4. Estimates page count (DOCX doesn't have explicit pages!)
    5. Returns text and estimated page count
    
    Important: DOCX files don't have "pages" like PDFs!
    They're just flowing text. We ESTIMATE pages based on word count.
    Assumption: ~500 words per page (standard document formatting)
    
    Args:
        file_path: Full path to the DOCX file
        
    Returns:
        Tuple of (extracted_text, estimated_page_count)
        Example: ("Document text here...", 3)
        
    Raises:
        Exception: If DOCX cannot be opened or read
        
    Example usage:
        text, pages = extract_text_from_docx("/uploads/agreement.docx")
        # text = "Full agreement text..."
        # pages = 3 (estimated based on word count)
    """
    try:
        # Step 1: Open the DOCX file
        doc = DocxDocument(file_path)
        
        # Step 2: Prepare storage for text from each paragraph
        text_parts = []  # Will hold ["paragraph 1", "paragraph 2", ...]
        
        # Step 3: Loop through all paragraphs in the document
        # A paragraph = each time user presses Enter in Word
        for paragraph in doc.paragraphs:
            # Skip empty paragraphs (just whitespace)
            # .strip() removes spaces/newlines, then check if anything left
            if paragraph.text.strip():
                # Add non-empty paragraph to our collection
                text_parts.append(paragraph.text)
        
        # Step 4: Combine all paragraphs into one big text string
        # Join with "\n\n" (two line breaks) for readability
        full_text = "\n\n".join(text_parts)
        
        # Step 5: Estimate page count (DOCX doesn't track pages!)
        # Count words by splitting text on spaces
        word_count = len(full_text.split())
        
        # Assumption: 500 words = 1 page (standard double-spaced document)
        # Use integer division: // (e.g., 1200 words // 500 = 2 pages)
        # max(1, ...) ensures at least 1 page even for short documents
        estimated_pages = max(1, word_count // 500)
        
        # Step 6: Log success with word count and estimated pages
        logger.info(f"Extracted {len(full_text)} characters ({word_count} words), estimated ~{estimated_pages} pages")
        
        # Step 7: Return text and ESTIMATED page count
        return full_text, estimated_pages
        
    except Exception as e:
        # If anything goes wrong (corrupt file, not a real DOCX, etc.)
        logger.error(f"Error extracting text from DOCX: {e}")
        raise Exception(f"Failed to extract text from DOCX: {str(e)}")


def process_document(file_path: str, file_type: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Main function to process ANY document type.
    
    This is the function you call - it figures out which extraction
    method to use based on file type.
    
    How it works:
    1. Check file type (pdf, docx, or doc)
    2. Call appropriate extraction function
    3. Return results OR error message
    
    Args:
        file_path: Full path to the document
        file_type: Type of file - "pdf", "docx", or "doc"
        
    Returns:
        Tuple of (extracted_text, page_count, error_message)
        
        Success case:
            ("Full document text...", 10, None)
            - Has text ✅
            - Has page count ✅
            - No error (None) ✅
        
        Failure case:
            (None, None, "Failed to extract text: corrupt file")
            - No text ❌
            - No page count ❌
            - Has error message ✅
    
    Example usage:
        # Success:
        text, pages, error = process_document("/uploads/contract.pdf", "pdf")
        if error is None:
            print(f"Success! {pages} pages")
        
        # Failure:
        text, pages, error = process_document("/uploads/corrupt.pdf", "pdf")
        if error:
            print(f"Failed: {error}")
    """
    try:
        # Check file type and route to appropriate function
        
        # Case 1: PDF files
        if file_type.lower() == "pdf":
            # Extract text using PDF extraction function
            text, pages = extract_text_from_pdf(file_path)
            # Return success: text, pages, no error (None)
            return text, pages, None
        
        # Case 2: Word documents (both .docx and .doc)
        # .lower() makes it case-insensitive: "DOCX", "docx", "Docx" all work
        elif file_type.lower() in ["docx", "doc"]:
            # Extract text using DOCX extraction function
            text, pages = extract_text_from_docx(file_path)
            # Return success: text, pages, no error (None)
            return text, pages, None
        
        # Case 3: Unsupported file type (like .txt, .jpg, .xlsx)
        else:
            # Create helpful error message
            error = f"Unsupported file type: {file_type}. Supported types: pdf, docx, doc"
            logger.error(error)
            # Return failure: no text, no pages, error message
            return None, None, error
    
    except Exception as e:
        # Catch ANY error that happened during processing
        # This could be: corrupt file, wrong format, file not found, etc.
        error = f"Processing failed: {str(e)}"
        logger.error(error)
        # Return failure: no text, no pages, error message
        return None, None, error


def validate_file_type(filename: str) -> Tuple[bool, str]:
    """
    Check if a filename has a supported file extension.
    
    Args:
        filename: Name of file (e.g., "contract.pdf", "agreement.DOCX")
        
    Returns:
        Tuple of (is_valid, file_type)
        
        Examples:
            validate_file_type("contract.pdf")     → (True, "pdf")
            validate_file_type("agreement.DOCX")   → (True, "docx")
            validate_file_type("image.jpg")        → (False, "jpg")
            validate_file_type("noextension")      → (False, "")
    """
    # Step 1: Define which file types we support
    allowed_extensions = ["pdf", "docx", "doc"]
    
    # Step 2: Check if filename has an extension
    if "." not in filename:
        return False, ""  # Invalid, no extension
    
    # Step 3: Extract the file extension
    # .rsplit(".", 1) splits from RIGHT, max 1 split
    # Examples:
    #   "contract.pdf".rsplit(".", 1)         → ["contract", "pdf"]
    #   "my.old.contract.pdf".rsplit(".", 1)  → ["my.old.contract", "pdf"]
    file_type = filename.rsplit(".", 1)[1].lower()
    
    # Step 4: Check if extension is in our allowed list
    if file_type in allowed_extensions:
        return True, file_type  # Valid!
    else:
        return False, file_type  # Invalid, but return what it was