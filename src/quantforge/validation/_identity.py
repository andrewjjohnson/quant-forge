"""Shared syntax guards for validation provenance identifiers."""

from quantforge.validation.errors import ValidationPlanError


def validated_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationPlanError(f"{label} must be a non-empty string")
    return value


def validated_hash(value: object, label: str) -> str:
    text = validated_text(value, label)
    if (
        len(text) != 64
        or text != text.lower()
        or any(character not in "0123456789abcdef" for character in text)
    ):
        raise ValidationPlanError(f"{label} must be a lowercase SHA-256 value")
    return text
