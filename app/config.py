from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "AI Career Copilot"
    environment: str = Field(default="development", alias="ENVIRONMENT")
    api_prefix: str = "/api"

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    api_access_key: SecretStr | None = Field(default=None, alias="AI_COPILOT_API_KEY")
    tenant_api_keys_raw: SecretStr = Field(
        default_factory=lambda: SecretStr(""), alias="AI_COPILOT_TENANT_KEYS"
    )
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    embedding_dimensions: int = Field(default=1536, ge=32, le=4096, alias="EMBEDDING_DIMENSIONS")

    chroma_path: str = Field(default="./chroma_db", alias="CHROMA_PATH")
    collection_name: str = Field(default="career_documents", alias="CHROMA_COLLECTION")
    session_database_path: str = Field(default="./data/sessions.db", alias="SESSION_DATABASE_PATH")
    object_storage_path: str = Field(default="./data/objects", alias="OBJECT_STORAGE_PATH")

    max_upload_mb: int = Field(default=12, ge=1, le=100, alias="MAX_UPLOAD_MB")
    max_context_chunks: int = Field(default=5, ge=1, le=20, alias="MAX_CONTEXT_CHUNKS")
    max_pdf_pages: int = Field(default=250, ge=1, le=2_000, alias="MAX_PDF_PAGES")
    max_document_characters: int = Field(
        default=2_000_000,
        ge=10_000,
        le=20_000_000,
        alias="MAX_DOCUMENT_CHARACTERS",
    )
    max_document_chunks: int = Field(
        default=2_000,
        ge=10,
        le=20_000,
        alias="MAX_DOCUMENT_CHUNKS",
    )
    embedding_batch_size: int = Field(default=64, ge=1, le=256, alias="EMBEDDING_BATCH_SIZE")
    embedding_cache_size: int = Field(default=512, ge=0, le=10_000, alias="EMBEDDING_CACHE_SIZE")
    requests_per_minute: int = Field(default=60, ge=1, le=10_000, alias="REQUESTS_PER_MINUTE")
    openai_timeout_seconds: float = Field(default=30, ge=1, le=120, alias="OPENAI_TIMEOUT_SECONDS")
    openai_max_retries: int = Field(default=2, ge=0, le=5, alias="OPENAI_MAX_RETRIES")
    llm_max_output_tokens: int = Field(
        default=1_800, ge=128, le=8_000, alias="LLM_MAX_OUTPUT_TOKENS"
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    expose_docs: bool = Field(default=True, alias="EXPOSE_DOCS")
    otel_exporter_otlp_endpoint: str | None = Field(
        default=None, alias="OTEL_EXPORTER_OTLP_ENDPOINT"
    )

    # Useful for tests, demos and screenshots when no paid API key is available.
    mock_llm: bool = Field(default=False, alias="AI_COPILOT_MOCK_LLM")
    mock_embeddings: bool = Field(default=False, alias="AI_COPILOT_MOCK_EMBEDDINGS")

    cors_origins_raw: str = Field(
        default="http://localhost:3000,http://localhost:5173", alias="CORS_ORIGINS"
    )

    @field_validator("environment")
    @classmethod
    def normalize_environment(cls, value: str) -> str:
        return value.strip().lower()

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins_raw.split(",") if origin.strip()]

    @property
    def cors_allows_credentials(self) -> bool:
        return "*" not in self.cors_origins

    @property
    def tenant_api_keys(self) -> dict[str, str]:
        """Parse comma-separated `tenant:key` pairs from the secret environment value."""
        entries: dict[str, str] = {}
        seen_keys: set[str] = set()
        for raw_entry in self.tenant_api_keys_raw.get_secret_value().split(","):
            if not raw_entry.strip():
                continue
            tenant, separator, key = raw_entry.partition(":")
            tenant = tenant.strip()
            key = key.strip()
            if not separator or not tenant or not key:
                raise ValueError("AI_COPILOT_TENANT_KEYS must contain tenant:key pairs")
            if not tenant.replace("-", "").replace("_", "").isalnum():
                raise ValueError(
                    "Tenant identifiers may contain letters, numbers, hyphens and underscores"
                )
            if tenant in entries:
                raise ValueError(f"Duplicate tenant identifier: {tenant}")
            if key in seen_keys:
                raise ValueError("An API key cannot be assigned to multiple tenants")
            entries[tenant] = key
            seen_keys.add(key)
        return entries

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
