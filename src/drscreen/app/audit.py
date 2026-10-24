"""Append-only CSV record of screening outcomes.

Each row holds a UTC timestamp, the quality outcome and the grade (empty when
the image was rejected). Images and anything that could identify a person are
never written.
"""

import csv
import threading
from datetime import UTC, datetime
from pathlib import Path

COLUMNS = ("timestamp", "quality", "grade")


class AuditLog:
    """Thread-safe appender for the audit CSV."""

    def __init__(self, path: Path) -> None:
        """Create the log; the file and its directory are created on first write.

        Args:
            path: CSV file to append to.
        """
        self.path = path
        self._lock = threading.Lock()

    def record(self, quality: str, grade: int | None) -> None:
        """Append one screening outcome.

        Args:
            quality: Quality gate label.
            grade: Predicted ICDR grade, or ``None`` if the image was rejected.
        """
        row = (
            datetime.now(UTC).isoformat(timespec="seconds"),
            quality,
            "" if grade is None else grade,
        )
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            is_new = not self.path.is_file()
            with self.path.open("a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                if is_new:
                    writer.writerow(COLUMNS)
                writer.writerow(row)
