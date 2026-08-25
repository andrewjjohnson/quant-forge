"""Stable validation-plan serialization and cache-compatibility checks."""

import json
from typing import cast

from quantforge.configuration import PrimitiveMapping
from quantforge.data.identity import canonical_json_bytes
from quantforge.validation.errors import ValidationPlanIdentityError
from quantforge.validation.models import ValidationPlan


def serialize_validation_plan(plan: ValidationPlan) -> bytes:
    """Serialize one validation manifest using QuantForge canonical JSON."""
    return canonical_json_bytes(plan.to_manifest())


def validate_validation_plan_manifest(
    plan: ValidationPlan, content: bytes
) -> PrimitiveMapping:
    """Reject non-canonical, corrupted, or cross-plan cached manifests."""
    try:
        decoded = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValidationPlanIdentityError(
            "validation plan manifest is not valid JSON"
        ) from error
    if not isinstance(decoded, dict):
        raise ValidationPlanIdentityError(
            "validation plan manifest must contain a JSON object"
        )
    primitive = cast(PrimitiveMapping, decoded)
    if canonical_json_bytes(primitive) != content:
        raise ValidationPlanIdentityError(
            "validation plan manifest does not use canonical serialization"
        )
    expected = plan.to_manifest()
    if primitive.get("plan_id") != plan.plan_id or primitive != expected:
        raise ValidationPlanIdentityError(
            "validation plan manifest does not match the fixed plan identity"
        )
    return primitive
