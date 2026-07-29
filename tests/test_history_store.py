from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from history_store import HistoryStoreError, SupabaseHistoryStore


class Response:
    def __init__(self, data):
        self.data = data


class HistoryStoreTests(unittest.TestCase):
    def build_store(self):
        self.client = MagicMock()
        self.conversations = MagicMock()
        self.messages = MagicMock()
        self.suggestions = MagicMock()
        tables = {
            "conversations": self.conversations,
            "messages": self.messages,
            "knowledge_suggestions": self.suggestions,
        }
        self.client.table.side_effect = tables.__getitem__
        self.addCleanup(patch.stopall)
        patch("history_store.create_client", return_value=self.client).start()
        return SupabaseHistoryStore("https://example.supabase.co", "service-key")

    def configure_existing_conversation(self, conversation_id="conversation-id"):
        (
            self.conversations.select.return_value.eq.return_value.limit.return_value.execute.return_value
        ) = Response([{"id": conversation_id}])

    def test_token_hash_is_deterministic_and_not_the_raw_token(self):
        token = "browser-secret-token"
        token_hash = SupabaseHistoryStore.token_hash(token)

        self.assertEqual(token_hash, SupabaseHistoryStore.token_hash(token))
        self.assertNotEqual(token, token_hash)
        self.assertEqual(len(token_hash), 64)

    def test_save_messages_uses_upsert_for_safe_retry(self):
        store = self.build_store()
        self.configure_existing_conversation()
        message = {
            "id": "c0de5b94-6030-4faa-b9e7-11ab776db4f5",
            "role": "user",
            "content": "How do I reset my password?",
            "created_at": "2026-07-29T00:00:00+00:00",
        }

        store.save_messages("browser-secret-token", [message])

        self.messages.upsert.assert_called_once_with(
            [
                {
                    "id": message["id"],
                    "conversation_id": "conversation-id",
                    "role": "user",
                    "content": message["content"],
                    "created_at": message["created_at"],
                }
            ]
        )

    def test_load_messages_returns_only_the_matching_conversation(self):
        store = self.build_store()
        self.configure_existing_conversation("conversation-123")
        (
            self.messages.select.return_value.eq.return_value.order.return_value.execute.return_value
        ) = Response(
            [
                {
                    "id": "c0de5b94-6030-4faa-b9e7-11ab776db4f5",
                    "role": "assistant",
                    "content": "Go to Settings > System > Software Update.",
                    "created_at": "2026-07-29T00:00:01+00:00",
                }
            ]
        )

        messages = store.load_messages("browser-secret-token")

        self.assertEqual(messages[0]["role"], "assistant")
        self.assertEqual(messages[0]["content"], "Go to Settings > System > Software Update.")
        self.messages.select.return_value.eq.assert_called_once_with(
            "conversation_id", "conversation-123"
        )

    def test_invalid_message_is_rejected_before_database_write(self):
        store = self.build_store()
        self.configure_existing_conversation()

        with self.assertRaises(ValueError):
            store.save_messages(
                "browser-secret-token",
                [
                    {
                        "id": "c0de5b94-6030-4faa-b9e7-11ab776db4f5",
                        "role": "system",
                        "content": "invalid role",
                        "created_at": "2026-07-29T00:00:00+00:00",
                    }
                ],
            )

        self.messages.upsert.assert_not_called()

    def test_database_errors_are_exposed_as_safe_application_errors(self):
        store = self.build_store()
        self.configure_existing_conversation()
        self.messages.upsert.return_value.execute.side_effect = RuntimeError("connection failed")

        with self.assertRaises(HistoryStoreError):
            store.save_messages(
                "browser-secret-token",
                [
                    {
                        "id": "c0de5b94-6030-4faa-b9e7-11ab776db4f5",
                        "role": "user",
                        "content": "Hello",
                        "created_at": "2026-07-29T00:00:00+00:00",
                    }
                ],
            )


if __name__ == "__main__":
    unittest.main()
