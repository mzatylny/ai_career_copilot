from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "AI Career Copilot"
    environment: str = Field(default="development", alias="ENVIRONMENT")
    api_prefix: str = "/api"

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    embedding_dimensions: int = Field(default=1536, alias="EMBEDDING_DIMENSIONS")

    chroma_path: str = Field(default="./chroma_db", alias="CHROMA_PATH")
    collection_name: str = Field(default="career_documents", alias="CHROMA_COLLECTION")

    max_upload_mb: int = Field(default=12, alias="MAX_UPLOAD_MB")
    max_context_chunks: int = Field(default=5, alias="MAX_CONTEXT_CHUNKS")
    max_pdf_pages: int = Field(default=250, alias="MAX_PDF_PAGES")
    embedding_batch_size: int = Field(default=64, alias="EMBEDDING_BATCH_SIZE")

    # Useful for tests, demos and screenshots when no paid API key is available.
    mock_llm: bool = Field(default=False, alias="AI_COPILOT_MOCK_LLM")
    mock_embeddings: bool = Field(default=False, alias="AI_COPILOT_MOCK_EMBEDDINGS")

    cors_origins_raw: str = Field(default="http://localhost:3000,http://localhost:5173", alias="CORS_ORIGINS")

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins_raw.split(",") if origin.strip()]

    @property
    def cors_allows_credentials(self) -> bool:
        return "*" not in self.cors_origins

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def should_use_mock_ai(self) -> bool:
        return self.mock_llm or not self.openai_api_key

    @property
    def should_use_mock_embeddings(self) -> bool:
        return self.mock_embeddings or not self.openai_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()
