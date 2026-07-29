"""Semantic domain gate for ResolveBot.

WHY THIS MODULE EXISTS
----------------------
The TF-IDF + Logistic Regression classifier answers one question well:
"which of my known intents is this question closest to?"

It cannot answer the question that actually matters before we invite a
visitor to teach us something: "does this question belong to phone support
at all?"

Those are different questions, and conflating them caused a real bug. The
word "swimming?" shares no vocabulary with any training example, so every
intent scored near-zero, the top score happened to be `greeting`, and the
low-confidence branch fired. ResolveBot then asked a visitor to supply a
*phone-support answer* for a swimming question. Low classifier confidence
means "I am unsure which intent", not "this is in my domain".

The fix is a second, independent signal. We embed the visitor's question
with a pretrained sentence-embedding model and compare it, by cosine
similarity, against the verified in-scope FAQ questions. Embeddings capture
meaning rather than shared words, so "swimming?" lands far away from every
phone-support question even though it shares no vocabulary with them -- which
is exactly the case simple keyword or TF-IDF overlap cannot catch.

Adding more `out_of_scope` training rows also helps and we do that too, but
it can only ever cover topics we thought of in advance. The semantic gate
generalises to unrelated topics nobody listed.

DESIGN NOTES
------------
* The embedding reference set deliberately EXCLUDES `out_of_scope` rows.
  We measure closeness to what ResolveBot genuinely knows about, not
  closeness to a list of things it refuses.
* Thresholds live in one place, are documented below, and can be overridden
  by `scope_thresholds.json` produced by `calibrate_thresholds.py`.
* If the embedding model cannot be loaded (no network on first run, install
  problem, low memory), we fall back to a deliberately STRICT lexical gate
  rather than disabling the gate. A failed gate must never quietly reopen
  the suggestion flow to unrelated questions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

MODULE_DIR = Path(__file__).resolve().parent

# A small, CPU-friendly model. ~90 MB, no GPU required, fast enough that a
# single short question embeds in a few milliseconds on Streamlit Cloud.
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

OUT_OF_SCOPE_INTENT = "out_of_scope"
THRESHOLD_FILE = MODULE_DIR / "scope_thresholds.json"


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
# Both values are cosine similarities in [-1, 1]; in practice MiniLM sentence
# similarities for unrelated English sentences sit near 0.0-0.15, loosely
# related sentences near 0.2-0.4, and paraphrases above 0.5.
#
# HARD_OUT_OF_SCOPE
#   Below this, the question is treated as unrelated even if the classifier
#   claims to be confident. This is a safety veto and is intentionally low, so
#   it only fires on questions that are genuinely nowhere near the domain.
#
# SUGGESTION_MIN_SIMILARITY
#   The bar a question must clear before ResolveBot is willing to ask a
#   visitor to supply an answer for it. Between the two thresholds the
#   question is redirected as off-topic rather than sent to the suggestion
#   flow -- we would rather decline a borderline real question than accept a
#   swimming question into the phone-support knowledge queue.
#
# These defaults are starting points. Run `python3 calibrate_thresholds.py`
# to check them against the labelled probe set in that script; it writes
# `scope_thresholds.json`, which this module prefers when present.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "hard_out_of_scope": 0.20,
    "suggestion_min_similarity": 0.35,
}


def load_thresholds(path: Path | str = THRESHOLD_FILE) -> dict[str, float]:
    """Prefer calibrated thresholds on disk, falling back to the defaults."""
    thresholds = dict(DEFAULT_THRESHOLDS)
    try:
        with Path(path).open() as handle:
            stored = json.load(handle)
    except (OSError, ValueError):
        return thresholds
    for key in thresholds:
        value = stored.get(key)
        if isinstance(value, (int, float)):
            thresholds[key] = float(value)
    return thresholds


# ---------------------------------------------------------------------------
# Reference set
# ---------------------------------------------------------------------------
def in_scope_questions(rows: Iterable[dict]) -> list[str]:
    """Verified questions ResolveBot can actually help with.

    `out_of_scope` rows are excluded on purpose: the gate measures distance to
    the supported domain, not distance to the refusal examples.
    """
    return [
        str(row["question"])
        for row in rows
        if row.get("intent") != OUT_OF_SCOPE_INTENT and row.get("question")
    ]


# ---------------------------------------------------------------------------
# Lexical fallback gate
# ---------------------------------------------------------------------------
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

# Words that, on their own, make a question plausibly about a phone or an
# order for one. Used only by the fallback gate.
DEVICE_VOCABULARY = frozenset(
    """
    phone mobile device handset smartphone tablet battery charge charging
    charger recharge power screen display touchscreen brightness camera photo
    photos picture lens flash storage memory space sim esim network cellular
    wifi bluetooth hotspot tethering data roaming signal sound speaker volume
    mic microphone headphone earphone earbuds ringtone vibrate silent update
    updates software os android ios firmware version app apps application
    setting settings restart reboot reset factory password passcode pin
    fingerprint biometric face unlock lock account login signin warranty
    repair service servicing technician order delivery shipment courier
    refund return exchange invoice receipt price pricing discount offer deal
    imei serial model variant cable port charging usb adapter overheating
    overheat heating hot hang hangs lag lagging slow freeze crash crashing
    bug glitch notification notifications alarm contacts backup restore sync
    cloud gallery keyboard call calls dial message messages sms share sharing
    airdrop nearby transfer bootloop touch gesture widget homescreen
    """.split()
)

_STOPWORDS = frozenset(
    """
    a an the is are was were do does did how what why when where which who
    can could should would will shall my me i you your it its this that
    to for of in on at with and or not no yes please help me tab
    """.split()
)


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_PATTERN.findall(text.lower()) if t not in _STOPWORDS}


class LexicalDomainGate:
    """Strict, dependency-free fallback used when embeddings are unavailable.

    It cannot recognise meaning, so it is deliberately conservative: a
    question only scores highly if it reuses vocabulary the verified dataset
    already contains, or names a device-related word. Unrelated questions
    score 0 and are redirected, which is the safe direction to fail in.
    """

    mode = "lexical-fallback"
    available = True

    def __init__(self, questions: Sequence[str]) -> None:
        self._reference = [_tokens(question) for question in questions]

    def similarity(self, question: str) -> float:
        asked = _tokens(question)
        if not asked:
            return 0.0

        best_overlap = 0.0
        for reference in self._reference:
            if not reference:
                continue
            shared = len(asked & reference)
            if not shared:
                continue
            # Overlap coefficient: forgiving about length differences between
            # a two-word question and a ten-word stored one.
            best_overlap = max(best_overlap, shared / min(len(asked), len(reference)))

        # A single unambiguous device word is enough to call a question
        # plausibly in-scope, which keeps genuine-but-unseen support questions
        # eligible for the suggestion flow.
        lexicon_hit = 0.45 if asked & DEVICE_VOCABULARY else 0.0
        return max(best_overlap, lexicon_hit)


class SemanticDomainGate:
    """Cosine similarity against verified in-scope questions, via MiniLM."""

    mode = "embedding"

    def __init__(self, questions: Sequence[str], model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.available = False
        self._questions = list(questions)
        self._model = None
        self._matrix = None

        try:
            # Imported lazily so the app still starts if the package or its
            # torch dependency is missing from the environment.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
            self._matrix = self._model.encode(
                self._questions,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            self.available = True
        except Exception:  # noqa: BLE001 - any failure must degrade, not crash
            self._model = None
            self._matrix = None

    def similarity(self, question: str) -> float:
        if not self.available or self._matrix is None:
            return 0.0
        try:
            vector = self._model.encode(
                [question],
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )[0]
            # Both sides are L2-normalised, so the dot product is the cosine.
            return float((self._matrix @ vector).max())
        except Exception:  # noqa: BLE001
            return 0.0


def build_domain_gate(
    rows: Iterable[dict],
    model_name: str = DEFAULT_MODEL_NAME,
    allow_embeddings: bool = True,
):
    """Return the strongest domain gate this environment can support."""
    questions = in_scope_questions(rows)
    if allow_embeddings:
        gate = SemanticDomainGate(questions, model_name=model_name)
        if gate.available:
            return gate
    return LexicalDomainGate(questions)


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------
VERIFIED = "verified"
OFF_TOPIC = "off_topic"
SUGGESTION = "suggestion"


@dataclass
class ScopeDecision:
    """The single decision object the interface renders from."""

    outcome: str  # VERIFIED | OFF_TOPIC | SUGGESTION
    intent: str
    confidence: float
    margin: float
    similarity: float
    gate_mode: str
    reason: str = ""
    thresholds: dict[str, float] = field(default_factory=dict)

    @property
    def allows_suggestion(self) -> bool:
        return self.outcome == SUGGESTION


def decide_scope(
    intent: str,
    confidence: float,
    margin: float,
    similarity: float,
    gate_mode: str = "embedding",
    thresholds: dict[str, float] | None = None,
    classifier_is_confident: bool | None = None,
    confidence_threshold: float = 0.10,
    margin_threshold: float = 0.05,
) -> ScopeDecision:
    """Combine the classifier verdict with the semantic domain score.

    The order of these rules is the whole point, so it is written out
    explicitly rather than folded into a clever expression:

    1. Far outside the domain  -> off-topic, even over a confident classifier.
    2. Classifier says off-topic -> off-topic (unchanged behaviour).
    3. Classifier confident      -> verified answer.
    4. Plausibly in-domain but unsure -> invite a suggestion.
    5. Anything else (unsure AND not plausibly in-domain) -> off-topic.

    Rule 5 is the bug fix: previously this case fell through to the
    suggestion flow.
    """
    limits = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        limits.update(thresholds)

    if classifier_is_confident is None:
        classifier_is_confident = (
            confidence >= confidence_threshold and margin >= margin_threshold
        )

    def build(outcome: str, reason: str) -> ScopeDecision:
        return ScopeDecision(
            outcome=outcome,
            intent=intent,
            confidence=confidence,
            margin=margin,
            similarity=similarity,
            gate_mode=gate_mode,
            reason=reason,
            thresholds=limits,
        )

    if similarity < limits["hard_out_of_scope"]:
        return build(OFF_TOPIC, "semantic gate: far outside the support domain")

    if intent == OUT_OF_SCOPE_INTENT:
        return build(OFF_TOPIC, "classifier predicted out_of_scope")

    if classifier_is_confident:
        return build(VERIFIED, "confident match above confidence and margin thresholds")

    if similarity >= limits["suggestion_min_similarity"]:
        return build(SUGGESTION, "plausibly in-domain but no verified answer")

    return build(OFF_TOPIC, "semantic gate: not close enough to the support domain")
