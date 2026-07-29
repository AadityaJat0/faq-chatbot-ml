"""Owner-side access to the `knowledge_suggestions` table.

This module is intentionally separate from `history_store.py`. That module
serves the anonymous public chat; this one serves a single authenticated
maintainer. Keeping them apart makes the trust boundary visible in the file
layout: nothing the public app imports can change a suggestion's status.

Approving a suggestion here records a REVIEW DECISION. It does not retrain
the model and it does not publish anything to GitHub. ResolveBot trains only
from `faq_data.json`, so publication remains a deliberate commit.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

TABLE_NAME = "knowledge_suggestions"

PENDING = "pending"
APPROVED = "approved"
REJECTED = "rejected"
VALID_STATUSES = (PENDING, APPROVED, REJECTED)

# The schema was created by hand, so tolerate the most likely column spellings
# instead of hard-failing on a name mismatch. First match wins.
QUESTION_FIELDS = ("question", "asked_question", "original_question")
ANSWER_FIELDS = ("suggested_answer", "answer", "suggestion")
CREATED_FIELDS = ("created_at", "inserted_at", "createdAt")
ID_FIELDS = ("id", "suggestion_id", "uuid")


class SuggestionStoreError(RuntimeError):
    """Raised when the review store cannot complete an operation."""


@dataclass(frozen=True)
class Suggestion:
    """One row, normalised so the interface never touches raw column names."""

    identifier: str
    question: str
    suggested_answer: str
    status: str
    created_at: str
    reviewed_at: str | None = None
    reviewer_note: str | None = None

    @property
    def is_pending(self) -> bool:
        return self.status == PENDING


def _first_present(row: dict[str, Any], candidates: Sequence[str], default: str = "") -> str:
    for name in candidates:
        value = row.get(name)
        if value not in (None, ""):
            return str(value)
    return default


def normalize_row(row: dict[str, Any]) -> Suggestion:
    """Turn a raw Supabase row into a Suggestion regardless of column naming."""
    return Suggestion(
        identifier=_first_present(row, ID_FIELDS),
        question=_first_present(row, QUESTION_FIELDS),
        suggested_answer=_first_present(row, ANSWER_FIELDS),
        status=str(row.get("status") or PENDING),
        created_at=_first_present(row, CREATED_FIELDS),
        reviewed_at=row.get("reviewed_at"),
        reviewer_note=row.get("reviewer_note"),
    )


def validate_status(status: str) -> str:
    """Guard the only field the owner interface is allowed to write."""
    if status not in VALID_STATUSES:
        raise ValueError(
            f"Unsupported status {status!r}; expected one of {', '.join(VALID_STATUSES)}"
        )
    return status


def review_payload(status: str, note: str | None = None, now: datetime | None = None) -> dict[str, Any]:
    """Build the update body for a review decision.

    Kept as a pure function so the approve/reject logic is testable without a
    database connection.
    """
    validate_status(status)
    stamp = (now or datetime.now(timezone.utc)).isoformat()
    payload: dict[str, Any] = {"status": status, "reviewed_at": stamp}
    cleaned = (note or "").strip()
    if cleaned:
        payload["reviewer_note"] = cleaned[:500]
    return payload


class SuggestionReviewStore:
    """Thin wrapper over the Supabase table used by the owner review page."""

    def __init__(
        self,
        url: str,
        service_role_key: str,
        client_factory: Callable[[str, str], Any] | None = None,
    ) -> None:
        if not url or not service_role_key:
            raise SuggestionStoreError("Supabase credentials are not configured.")
        if client_factory is None:
            try:
                from supabase import create_client
            except ImportError as error:  # pragma: no cover - import guard
                raise SuggestionStoreError(
                    "The supabase package is not installed."
                ) from error
            client_factory = create_client
        self._client = client_factory(url, service_role_key)

    # -- reads ------------------------------------------------------------
    def list_suggestions(
        self,
        status: str | None = None,
        limit: int = 25,
        offset: int = 0,
    ) -> list[Suggestion]:
        """Newest first, paginated so a long queue stays usable."""
        try:
            query = self._client.table(TABLE_NAME).select("*")
            if status:
                query = query.eq("status", validate_status(status))
            response = (
                query.order("created_at", desc=True)
                .range(offset, offset + max(limit, 1) - 1)
                .execute()
            )
        except ValueError:
            raise
        except Exception as error:  # noqa: BLE001
            raise SuggestionStoreError("Could not load suggestions.") from error

        rows = getattr(response, "data", None) or []
        return [normalize_row(row) for row in rows]

    def count_by_status(self) -> dict[str, int]:
        """Small queue sizes, so a plain count per status is cheap enough."""
        counts: dict[str, int] = {}
        for status in VALID_STATUSES:
            try:
                response = (
                    self._client.table(TABLE_NAME)
                    .select("status", count="exact")
                    .eq("status", status)
                    .limit(1)
                    .execute()
                )
                counts[status] = int(getattr(response, "count", 0) or 0)
            except Exception:  # noqa: BLE001 - a missing count must not break the page
                counts[status] = 0
        return counts

    # -- writes -----------------------------------------------------------
    def set_status(
        self,
        identifier: str,
        status: str,
        note: str | None = None,
        id_column: str = "id",
    ) -> Suggestion:
        """Record an approve/reject decision. Nothing else is ever written."""
        if not identifier:
            raise SuggestionStoreError("This suggestion has no identifier to update.")
        payload = review_payload(status, note)
        try:
            response = (
                self._client.table(TABLE_NAME)
                .update(payload)
                .eq(id_column, identifier)
                .execute()
            )
        except Exception as error:  # noqa: BLE001
            # `reviewed_at` / `reviewer_note` may not exist yet if the
            # migration has not been applied. Retry with status alone so the
            # buttons still work on an un-migrated deployment.
            try:
                response = (
                    self._client.table(TABLE_NAME)
                    .update({"status": payload["status"]})
                    .eq(id_column, identifier)
                    .execute()
                )
            except Exception:  # noqa: BLE001
                raise SuggestionStoreError("Could not update this suggestion.") from error

        rows = getattr(response, "data", None) or []
        if not rows:
            raise SuggestionStoreError("No suggestion was updated; it may have been removed.")
        return normalize_row(rows[0])
