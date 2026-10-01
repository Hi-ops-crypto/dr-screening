import logging

import pytest

from drscreen.logging_setup import configure_logging


@pytest.mark.parametrize(
    ("name", "expected"), [("debug", logging.DEBUG), ("WARNING", logging.WARNING)]
)
def test_sets_root_level_case_insensitively(name: str, expected: int) -> None:
    configure_logging(name)

    assert logging.getLogger().level == expected


def test_unknown_level_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown log level"):
        configure_logging("verbose")
