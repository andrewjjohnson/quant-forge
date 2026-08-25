"""Domain errors for leakage-safe validation plans."""


class ValidationPlanError(ValueError):
    """Base error for invalid validation definitions or membership."""


class ValidationPlanIdentityError(ValidationPlanError):
    """A serialized validation artifact does not match its declared identity."""
