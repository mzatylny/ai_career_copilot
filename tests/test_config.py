import pytest
from pydantic import ValidationError

from app.config import Settings


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("MAX_UPLOAD_MB", 0),
        ("MAX_CONTEXT_CHUNKS", 21),
        ("MAX_PDF_PAGES", 0),
        ("MAX_DOCUMENT_CHARACTERS", 9_999),
        ("MAX_DOCUMENT_CHUNKS", 9),
        ("EMBEDDING_BATCH_SIZE", 257),
        ("EMBEDDING_DIMENSIONS", 16),
    ],
)
def test_resource_limits_are_validated(field, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


def test_api_key_is_redacted_from_settings_repr():
    settings = Settings(
        _env_file=None,
        AI_COPILOT_API_KEY="super-secret",
        AI_COPILOT_TENANT_KEYS="alpha:another-secret",
    )

    assert "super-secret" not in repr(settings)
    assert "another-secret" not in repr(settings)


def test_tenant_keys_are_parsed():
    settings = Settings(
        _env_file=None,
        AI_COPILOT_TENANT_KEYS="alpha:key-one,beta:key-two",
    )

    assert settings.tenant_api_keys == {"alpha": "key-one", "beta": "key-two"}


def test_invalid_tenant_key_configuration_is_rejected_when_used():
    settings = Settings(_env_file=None, AI_COPILOT_TENANT_KEYS="missing-separator")

    with pytest.raises(ValueError):
        _ = settings.tenant_api_keys


@pytest.mark.parametrize(
    "configured",
    [
        "alpha:key-one,alpha:key-two",
        "alpha:shared-key,beta:shared-key",
    ],
)
def test_duplicate_tenant_or_key_configuration_is_rejected(configured):
    settings = Settings(_env_file=None, AI_COPILOT_TENANT_KEYS=configured)

    with pytest.raises(ValueError):
        _ = settings.tenant_api_keys
