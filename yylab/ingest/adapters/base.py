from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ..records import Doc


class Adapter:
    name = "base"
    extensions: set[str] = set()
    default_domain = "other"

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() in self.extensions

    def extract(self, path: Path) -> Iterator[Doc]:
        raise NotImplementedError

    def estimated_records(self, path: Path) -> int | None:
        """Return a cheap record-count estimate when the container provides one."""

        return None

    def _reset_issues(self) -> None:
        self._extraction_issues: list[tuple[str, str, str]] = []

    def _record_issue(self, locator: str, reason_code: str, detail: str) -> None:
        if not hasattr(self, "_extraction_issues"):
            self._reset_issues()
        self._extraction_issues.append((locator, reason_code, detail[:2000]))

    def drain_issues(self) -> list[tuple[str, str, str]]:
        issues = list(getattr(self, "_extraction_issues", ()))
        self._extraction_issues = []
        return issues

    def _read_text(self, path: Path) -> str:
        text = path.read_text("utf-8", errors="ignore").removeprefix("\ufeff")
        return text.replace("\r\n", "\n").replace("\r", "\n")
