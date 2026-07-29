"""Tests for the semantic domain gate and the owner review store.

These run without a network connection, without Streamlit, and without the
embedding model: the decision logic is a pure function of (intent, confidence,
margin, similarity), and the review store accepts an injected fake client.

Run:  python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_gate import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    OFF_TOPIC,
    OUT_OF_SCOPE_INTENT,
    SUGGESTION,
    VERIFIED,
    LexicalDomainGate,
    decide_scope,
    in_scope_questions,
)
from suggestion_store import (  # noqa: E402
    APPROVED,
    PENDING,
    REJECTED,
    SuggestionReviewStore,
    SuggestionStoreError,
    normalize_row,
    review_payload,
    validate_status,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LOW = DEFAULT_THRESHOLDS["hard_out_of_scope"]
BAR = DEFAULT_THRESHOLDS["suggestion_min_similarity"]


class DecisionLogicTests(unittest.TestCase):
    """The bug this project fixed lives entirely in these branches."""

    def test_clearly_unrelated_question_is_off_topic_not_a_suggestion(self):
        # "swimming?" — every intent scores near zero, `greeting` wins by
        # accident, and the old code opened the suggestion flow.
        decision = decide_scope(
            intent="greeting", confidence=0.08, margin=0.00, similarity=0.05
        )
        self.assertEqual(decision.outcome, OFF_TOPIC)
        self.assertFalse(decision.allows_suggestion)

    def test_unrelated_question_never_reaches_the_suggestion_queue(self):
        for similarity in (0.0, 0.05, 0.1, LOW - 0.01, BAR - 0.01):
            with self.subTest(similarity=similarity):
                decision = decide_scope(
                    intent="greeting", confidence=0.09, margin=0.01, similarity=similarity
                )
                self.assertFalse(decision.allows_suggestion)
                self.assertEqual(decision.outcome, OFF_TOPIC)

    def test_plausible_in_scope_unknown_question_allows_a_suggestion(self):
        decision = decide_scope(
            intent="battery_life", confidence=0.09, margin=0.01, similarity=0.55
        )
        self.assertEqual(decision.outcome, SUGGESTION)
        self.assertTrue(decision.allows_suggestion)

    def test_confident_known_question_returns_the_verified_answer(self):
        decision = decide_scope(
            intent="battery_life", confidence=0.31, margin=0.18, similarity=0.82
        )
        self.assertEqual(decision.outcome, VERIFIED)
        self.assertFalse(decision.allows_suggestion)

    def test_classifier_out_of_scope_still_redirects_regardless_of_confidence(self):
        decision = decide_scope(
            intent=OUT_OF_SCOPE_INTENT, confidence=0.13, margin=0.05, similarity=0.40
        )
        self.assertEqual(decision.outcome, OFF_TOPIC)

    def test_hard_floor_overrides_a_confident_classifier(self):
        decision = decide_scope(
            intent="greeting", confidence=0.90, margin=0.80, similarity=0.02
        )
        self.assertEqual(decision.outcome, OFF_TOPIC)
        self.assertIn("far outside", decision.reason)

    def test_threshold_boundaries_are_inclusive_on_the_permissive_side(self):
        at_bar = decide_scope(
            intent="battery_life", confidence=0.05, margin=0.01, similarity=BAR
        )
        just_below = decide_scope(
            intent="battery_life", confidence=0.05, margin=0.01, similarity=BAR - 0.001
        )
        self.assertEqual(at_bar.outcome, SUGGESTION)
        self.assertEqual(just_below.outcome, OFF_TOPIC)

    def test_custom_thresholds_are_honoured(self):
        decision = decide_scope(
            intent="battery_life",
            confidence=0.05,
            margin=0.01,
            similarity=0.30,
            thresholds={"hard_out_of_scope": 0.05, "suggestion_min_similarity": 0.25},
        )
        self.assertEqual(decision.outcome, SUGGESTION)


class ReferenceSetTests(unittest.TestCase):
    def test_out_of_scope_rows_are_excluded_from_the_reference_set(self):
        rows = [
            {"question": "battery drains", "intent": "battery_life"},
            {"question": "how do I cook pasta?", "intent": OUT_OF_SCOPE_INTENT},
        ]
        self.assertEqual(in_scope_questions(rows), ["battery drains"])

    def test_shipped_dataset_has_in_scope_questions_and_out_of_scope_coverage(self):
        with (REPO_ROOT / "faq_data.json").open() as handle:
            rows = json.load(handle)
        intents = {row["intent"] for row in rows}
        out_of_scope = [r for r in rows if r["intent"] == OUT_OF_SCOPE_INTENT]
        self.assertIn(OUT_OF_SCOPE_INTENT, intents)
        self.assertGreater(len(in_scope_questions(rows)), 40)
        # Diversified beyond the original ten examples.
        self.assertGreaterEqual(len(out_of_scope), 25)


class LexicalFallbackTests(unittest.TestCase):
    """The fallback must fail strict, never permissive."""

    def setUp(self):
        with (REPO_ROOT / "faq_data.json").open() as handle:
            rows = json.load(handle)
        self.gate = LexicalDomainGate(in_scope_questions(rows))

    def test_unrelated_question_scores_below_the_suggestion_bar(self):
        for question in ("swimming?", "how do I cook pasta?", "who won the cricket match?"):
            with self.subTest(question=question):
                self.assertLess(self.gate.similarity(question), BAR)

    def test_device_question_stays_eligible_for_a_suggestion(self):
        self.assertGreaterEqual(
            self.gate.similarity("my phone gets very hot while charging"), BAR
        )

    def test_fallback_reports_itself_as_available_so_the_app_never_crashes(self):
        self.assertTrue(self.gate.available)
        self.assertEqual(self.gate.mode, "lexical-fallback")

    def test_empty_question_scores_zero(self):
        self.assertEqual(self.gate.similarity("   "), 0.0)


class FakeQuery:
    """Minimal stand-in for the supabase-py fluent query builder."""

    def __init__(self, table):
        self._table = table
        self._filters = {}
        self._payload = None
        self._is_update = False

    def select(self, *_args, **_kwargs):
        return self

    def update(self, payload):
        self._payload = payload
        self._is_update = True
        return self

    def eq(self, column, value):
        self._filters[column] = value
        return self

    def order(self, *_args, **_kwargs):
        return self

    def range(self, start, end):
        self._range = (start, end)
        return self

    def limit(self, _n):
        return self

    def execute(self):
        rows = self._table.rows
        if self._is_update:
            updated = []
            for row in rows:
                if all(str(row.get(k)) == str(v) for k, v in self._filters.items()):
                    row.update(self._payload)
                    updated.append(row)
            self._table.updates.append(dict(self._payload))
            return type("Response", (), {"data": updated, "count": len(updated)})()
        matched = [
            row
            for row in rows
            if all(str(row.get(k)) == str(v) for k, v in self._filters.items())
        ]
        return type("Response", (), {"data": matched, "count": len(matched)})()


class FakeClient:
    def __init__(self, rows):
        self.rows = rows
        self.updates = []

    def table(self, _name):
        return FakeQuery(self)


class ReviewStoreTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {
                "id": "s1",
                "question": "how do I turn on quick sharing?",
                "suggested_answer": "Swipe down and tap Quick Share.",
                "status": PENDING,
                "created_at": "2026-07-29T10:00:00Z",
            },
            {
                "id": "s2",
                "question": "battery calibration?",
                "suggested_answer": "Drain and recharge fully.",
                "status": PENDING,
                "created_at": "2026-07-28T10:00:00Z",
            },
        ]
        self.client = FakeClient(self.rows)
        self.store = SuggestionReviewStore(
            "https://example.supabase.co",
            "service-role-key",
            client_factory=lambda url, key: self.client,
        )

    def test_approve_updates_only_the_targeted_row(self):
        updated = self.store.set_status("s1", APPROVED)
        self.assertEqual(updated.status, APPROVED)
        self.assertEqual(self.rows[0]["status"], APPROVED)
        self.assertEqual(self.rows[1]["status"], PENDING)

    def test_reject_records_a_reviewed_timestamp(self):
        self.store.set_status("s2", REJECTED)
        self.assertEqual(self.rows[1]["status"], REJECTED)
        self.assertIn("reviewed_at", self.rows[1])

    def test_reviewer_note_is_stored_when_supplied(self):
        self.store.set_status("s1", APPROVED, note="  correct and safe  ")
        self.assertEqual(self.rows[0]["reviewer_note"], "correct and safe")

    def test_blank_note_is_not_written(self):
        payload = review_payload(APPROVED, "   ")
        self.assertNotIn("reviewer_note", payload)

    def test_only_the_three_review_statuses_are_accepted(self):
        for status in (APPROVED, REJECTED, PENDING):
            self.assertEqual(validate_status(status), status)
        for bad in ("published", "deleted", "APPROVED", "", "approved; drop table"):
            with self.subTest(status=bad):
                with self.assertRaises(ValueError):
                    validate_status(bad)

    def test_review_payload_never_touches_the_answer_or_question(self):
        payload = review_payload(APPROVED, "fine")
        self.assertEqual(set(payload) - {"status", "reviewed_at", "reviewer_note"}, set())

    def test_review_payload_uses_an_iso_timestamp(self):
        payload = review_payload(REJECTED, None, now=datetime(2026, 7, 30, tzinfo=timezone.utc))
        self.assertEqual(payload["reviewed_at"], "2026-07-30T00:00:00+00:00")

    def test_missing_identifier_is_rejected(self):
        with self.assertRaises(SuggestionStoreError):
            self.store.set_status("", APPROVED)

    def test_listing_filters_by_status(self):
        self.store.set_status("s1", APPROVED)
        pending = self.store.list_suggestions(status=PENDING)
        self.assertEqual([s.identifier for s in pending], ["s2"])

    def test_rows_are_normalised_across_column_spellings(self):
        suggestion = normalize_row(
            {
                "suggestion_id": "x9",
                "asked_question": "why is my sim not detected",
                "answer": "Reseat the SIM tray.",
                "status": PENDING,
                "inserted_at": "2026-07-30T09:00:00Z",
            }
        )
        self.assertEqual(suggestion.identifier, "x9")
        self.assertEqual(suggestion.question, "why is my sim not detected")
        self.assertEqual(suggestion.suggested_answer, "Reseat the SIM tray.")
        self.assertTrue(suggestion.is_pending)


class BadgePersistenceShapeTests(unittest.TestCase):
    """Badges are stored per message, so the decision must carry what they show."""

    def test_decision_exposes_every_value_the_badge_renders(self):
        decision = decide_scope(
            intent="battery_life", confidence=0.31, margin=0.18, similarity=0.82
        )
        for attribute in ("intent", "confidence", "margin", "similarity", "outcome"):
            self.assertTrue(hasattr(decision, attribute))

    def test_each_outcome_maps_to_exactly_one_badge_colour(self):
        outcomes = {
            decide_scope("battery_life", 0.31, 0.18, 0.82).outcome,
            decide_scope("out_of_scope", 0.13, 0.05, 0.40).outcome,
            decide_scope("battery_life", 0.09, 0.01, 0.55).outcome,
        }
        self.assertEqual(outcomes, {VERIFIED, OFF_TOPIC, SUGGESTION})


if __name__ == "__main__":
    unittest.main(verbosity=2)
