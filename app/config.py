from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str
    gemini_api_key:str
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection_name: str = "legal_documents"

    environment : str = "development"  # or "production"
    debug: bool = True
    app_name: str = "Legal Document Analyzer"
    app_version: str = "1.0.0"
    gcp_project_id: str = ""
    gcs_bucket_name: str = ""
    google_application_credentials: str = ""

    embeddings_model: str = "gemini-1.5-flash-embedding-002"

    model_config: SettingsConfigDict = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False
    )

settings = Settings()




