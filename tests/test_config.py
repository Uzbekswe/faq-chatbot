from __future__ import annotations

import pytest
from pydantic import ValidationError

from faq_chatbot.config import Settings


def test_settings_reject_invalid_database_and_retrieval_values():
    with pytest.raises(ValidationError, match="DATABASE_URL must use sqlite"):
        Settings(database_url="postgresql://localhost/chat")
    with pytest.raises(ValidationError):
        Settings(top_k=0, similarity_threshold=2)


def test_secret_values_are_masked():
    settings = Settings(openai_api_key="private-test-value")
    assert "private-test-value" not in repr(settings)
    assert settings.missing_configuration == []
