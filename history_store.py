"""Persistent, anonymous conversation storage for ResolveBot.

The browser keeps an opaque random access token in a cookie.  This module only
stores a SHA-256 hash of that token, so a database export cannot be used to
reconstruct a visitor's browser token.

All calls are made from the Streamlit server with a Supabase service-role key.
That key must stay in Streamlit Secrets and must never be sent to the browser.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Mapping, Sequence

from supabase import Client, create_client


MAX_MESSAGE_LENGTH = 4_000


class HistoryStoreError(RuntimeError):
    """Raised when a persistent-history operation cannot be completed."""


class SupabaseHistoryStore:
    """Store ResolveBot conversations in a Supabase Postgres database."""

    def __init__(self, url: str, service_role_key: str) -> None:
        self._client: Client = create_client(url, service_role_key)

    @staticmethod
    def token_hash(access_token: str) -> str:
        """Return the non-reversible database lookup value for a browser token."""
        return sha256(access_token.encode("utf-8")).hexdigest()

    def load_messages(self, access_token: str) -> list[dict[str, str]]:
        """Return one browser's messages, ordered from oldest to newest."""
        conversation_id = self._get_or_create_conversation_id(access_token)
        try:
            response = (
                self._client.table("messages")
                .select("id, role, content, prediction_badge, created_at")
                .eq("conversation_id", conversation_id)
                .order("created_at")
                .execute()
            )
        except Exception as exc:  # Supabase exposes several transport exception types.
            raise HistoryStoreError("ResolveBot could not load this chat history.") from exc

        rows = response.data or []
        messages = []
        for row in rows:
            message = {
                "id": str(row["id"]),
                "role": str(row["role"]),
                "content": str(row["content"]),
                "created_at": str(row["created_at"]),
            }
            if row.get("prediction_badge"):
                message["badge"] = str(row["prediction_badge"])
            messages.append(message)
        return messages

    def save_messages(
        self, access_token: str, messages: Sequence[Mapping[str, Any]]
    ) -> None:
        """Persist messages immediately. UUID message IDs make retries idempotent."""
        if not messages:
            return

        conversation_id = self._get_or_create_conversation_id(access_token)
        rows = [self._message_row(conversation_id, message) for message in messages]
        try:
            # Saving one row at a time keeps optional fields (such as a badge)
            # attached to the correct assistant message. Bulk upserts infer their
            # columns from the first row, which is normally a user message.
            for row in rows:
                self._client.table("messages").upsert(row).execute()
        except Exception as exc:
            raise HistoryStoreError("ResolveBot could not save this chat right now.") from exc

    def save_feedback(
        self, access_token: str, question: str, suggested_answer: str
    ) -> None:
        """Save a user suggestion for review without changing the trained model."""
        if not question.strip() or not suggested_answer.strip():
            raise ValueError("A feedback question and answer are both required.")
        if len(question) > MAX_MESSAGE_LENGTH or len(suggested_answer) > MAX_MESSAGE_LENGTH:
            raise ValueError("Feedback is too long to save.")

        conversation_id = self._get_or_create_conversation_id(access_token)
        try:
            (
                self._client.table("knowledge_suggestions")
                .insert(
                    {
                        "conversation_id": conversation_id,
                        "question": question,
                        "suggested_answer": suggested_answer,
                    }
                )
                .execute()
            )
        except Exception as exc:
            raise HistoryStoreError("ResolveBot could not save that suggestion.") from exc

    def delete_history(self, access_token: str) -> None:
        """Permanently delete the conversation associated with one browser token."""
        conversation_id = self._find_conversation_id(access_token)
        if conversation_id is None:
            return

        try:
            (
                self._client.table("conversations")
                .delete()
                .eq("id", conversation_id)
                .execute()
            )
        except Exception as exc:
            raise HistoryStoreError("ResolveBot could not delete this chat.") from exc

    def _get_or_create_conversation_id(self, access_token: str) -> str:
        existing = self._find_conversation_id(access_token)
        if existing is not None:
            return existing

        token_hash = self.token_hash(access_token)
        try:
            response = (
                self._client.table("conversations")
                .insert({"visitor_token_hash": token_hash})
                .select("id")
                .execute()
            )
            rows = response.data or []
            if rows:
                return str(rows[0]["id"])
        except Exception:
            # A second request from the same browser can create the row first.
            # Re-checking converts that harmless race into a normal read.
            existing = self._find_conversation_id(access_token)
            if existing is not None:
                return existing
            raise HistoryStoreError("ResolveBot could not start a saved chat.")

        raise HistoryStoreError("ResolveBot could not start a saved chat.")

    def _find_conversation_id(self, access_token: str) -> str | None:
        try:
            response = (
                self._client.table("conversations")
                .select("id")
                .eq("visitor_token_hash", self.token_hash(access_token))
                .limit(1)
                .execute()
            )
        except Exception as exc:
            raise HistoryStoreError("ResolveBot could not access saved chats.") from exc

        rows = response.data or []
        return str(rows[0]["id"]) if rows else None

    @staticmethod
    def _message_row(conversation_id: str, message: Mapping[str, Any]) -> dict[str, str]:
        required_fields = ("id", "role", "content", "created_at")
        missing = [field for field in required_fields if not message.get(field)]
        if missing:
            raise ValueError(f"Message is missing required fields: {', '.join(missing)}")

        role = str(message["role"])
        content = str(message["content"])
        if role not in {"user", "assistant"}:
            raise ValueError("Message role must be 'user' or 'assistant'.")
        if len(content) > MAX_MESSAGE_LENGTH:
            raise ValueError("Message is too long to save.")

        row = {
            "id": str(message["id"]),
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            # PostgREST uses the fields from a bulk request consistently across
            # every row. Include this nullable field for user messages too, so
            # an assistant badge in the same save is not dropped.
            "prediction_badge": None,
            "created_at": str(message["created_at"]),
        }
        if message.get("badge"):
            badge = str(message["badge"])
            if len(badge) > MAX_MESSAGE_LENGTH:
                raise ValueError("Message indicator is too long to save.")
            row["prediction_badge"] = badge
        return row
