"""Exception hierarchy for the drscreen package.

Every error raised deliberately by drscreen derives from ``DrScreenError`` so
that entry points can separate expected, user-actionable failures (bad config,
bad input image) from programming errors, which should propagate with a full
traceback.
"""


class DrScreenError(Exception):
    """Base class for all errors raised deliberately by drscreen."""


class ConfigError(DrScreenError):
    """A configuration file is missing, malformed or fails validation."""


class DataError(DrScreenError):
    """A dataset, manifest or image file is missing, unreadable or malformed."""


class FundusNotFoundError(DataError):
    """An image does not contain a detectable circular fundus region."""
