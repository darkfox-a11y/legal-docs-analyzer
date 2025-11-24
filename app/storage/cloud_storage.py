"""
Google Cloud Storage helper with proper authentication.
"""

import logging
import os
import json
from google.cloud import storage
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "legal-doc-analyzer-files-neerad")
PROJECT_ID = os.getenv("GCP_PROJECT_ID")

def get_storage_client():
    """Get authenticated storage client."""
    
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    # First try: Use key file path
    if credentials_path:
        logger.info(f"🔑 Using service account from {credentials_path}")
        if os.path.exists(credentials_path):
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            client = storage.Client(credentials=credentials, project=PROJECT_ID)
            return client
        else:
            logger.error(f"❌ Key file not found: {credentials_path}")
    
    # Second try: Use JSON from env var (for Railway)
    gcs_key_json = os.getenv("GCS_KEY_JSON")
    if gcs_key_json:
        logger.info("🔑 Using service account from GCS_KEY_JSON")
        credentials_dict = json.loads(gcs_key_json)
        credentials = service_account.Credentials.from_service_account_info(credentials_dict)
        client = storage.Client(credentials=credentials, project=PROJECT_ID)
        return client
    
    # Last resort: Application default credentials
    logger.warning("⚠️ Using application default credentials (not recommended)")
    client = storage.Client(project=PROJECT_ID)
    return client


def upload_to_cloud(local_path: str, cloud_name: str) -> str:
    """
    Upload file to Google Cloud Storage.
    
    Args:
        local_path: Path to local file (e.g., "uploads/doc123.pdf")
        cloud_name: Name in GCS (e.g., "uploads/user1/doc123.pdf")
    
    Returns:
        Public URL to access the file
    """
    try:
        client = get_storage_client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(cloud_name)
        
        # Upload file
        blob.upload_from_filename(local_path)
        
        logger.info(f"✅ Uploaded to GCS: gs://{BUCKET_NAME}/{cloud_name}")
        
        # Return public URL
        return f"https://storage.googleapis.com/{BUCKET_NAME}/{cloud_name}"
        
    except Exception as e:
        logger.error(f"❌ GCS upload failed: {e}")
        # Fallback to local storage
        return f"local://{local_path}"


def delete_from_cloud(cloud_name: str) -> bool:
    """Delete file from Google Cloud Storage."""
    try:
        client = get_storage_client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(cloud_name)
        
        blob.delete()
        
        logger.info(f"✅ Deleted from GCS: gs://{BUCKET_NAME}/{cloud_name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ GCS delete failed: {e}")
        return False


def download_from_cloud(cloud_name: str, local_path: str) -> str:
    """Download file from Google Cloud Storage."""
    try:
        client = get_storage_client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(cloud_name)
        
        blob.download_to_filename(local_path)
        
        logger.info(f"✅ Downloaded from GCS: gs://{BUCKET_NAME}/{cloud_name}")
        return local_path
        
    except Exception as e:
        logger.error(f"❌ GCS download failed: {e}")
        raise