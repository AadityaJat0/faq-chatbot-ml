"""Calibrate the semantic domain gate against a labelled probe set.

The two thresholds in `semantic_gate.py` should not be guessed. This script
scores a small, hand-labelled set of probe questions against the verified
in-scope dataset, then chooses the cut points that separate them, and writes
the result to `scope_thresholds.json` for the app to pick up.

Run it after changing `faq_data.json` or the embedding model:

    python3 calibrate_thresholds.py

It prints every probe with its similarity so the chosen numbers can be
justified in a viva rather than merely asserted.
"""

from __future__ import annotations

import json
from pathlib import Path

from semantic_gate import DEFAULT_THRESHOLDS, THRESHOLD_FILE, build_domain_gate

DATA_PATH = Path(__file__).resolve().parent / "faq_data.json"

# Questions that clearly belong to phone support, INCLUDING ones the dataset
# does not answer. These must stay eligible for the suggestion flow.
IN_SCOPE_PROBES = [
    "my battery drains fast",
    "how do I reset my password?",
    "the screen keeps flickering after the update",
    "my phone gets very hot while charging",
    "can I use a fast charger with this model?",
    "the fingerprint sensor stopped working",
    "how do I move my contacts to a new handset?",
    "bluetooth will not pair with my car",
    "where has my parcel reached",
    "how do I turn on quick sharing?",
]

# Questions that are clearly nothing to do with phone support. None of these
# may reach the suggestion flow.
OUT_OF_SCOPE_PROBES = [
    "swimming?",
    "what do you mean by dancing?",
    "how do I cook pasta?",
    "who won the cricket match?",
    "will it rain tomorrow?",
    "explain photosynthesis",
    "what do cats eat?",
    "suggest some good music",
    "who is the prime minister?",
    "help me with my maths homework",
]


def main() -> None:
    with DATA_PATH.open() as handle:
        rows = json.load(handle)

    gate = build_domain_gate(rows)
    print(f"Domain gate mode: {gate.mode}\n")
    if gate.mode != "embedding":
        print(
            "WARNING: the embedding model did not load, so these numbers "
            "describe the lexical fallback only. Install sentence-transformers "
            "and rerun before trusting the thresholds.\n"
        )

    in_scores = sorted((gate.similarity(q), q) for q in IN_SCOPE_PROBES)
    out_scores = sorted(((gate.similarity(q), q) for q in OUT_OF_SCOPE_PROBES), reverse=True)

    print("In-scope probes (lowest first):")
    for score, question in in_scores:
        print(f"  {score:0.3f}  {question}")
    print("\nOut-of-scope probes (highest first):")
    for score, question in out_scores:
        print(f"  {score:0.3f}  {question}")

    lowest_in_scope = in_scores[0][0]
    highest_out_of_scope = out_scores[0][0]
    print(f"\nlowest in-scope     : {lowest_in_scope:0.3f}")
    print(f"highest out-of-scope: {highest_out_of_scope:0.3f}")

    if lowest_in_scope <= highest_out_of_scope:
        print(
            "\nThe two groups overlap, so no single cut separates them cleanly.\n"
            "Keeping the defaults, which favour redirecting a borderline question\n"
            "over letting an unrelated one into the suggestion queue."
        )
        thresholds = dict(DEFAULT_THRESHOLDS)
    else:
        # Sit the suggestion bar in the gap, slightly nearer the out-of-scope
        # side so genuine support questions are not redirected by accident.
        gap = lowest_in_scope - highest_out_of_scope
        suggestion_min = round(highest_out_of_scope + gap * 0.4, 3)
        hard_floor = round(max(0.05, highest_out_of_scope * 0.75), 3)
        thresholds = {
            "hard_out_of_scope": hard_floor,
            "suggestion_min_similarity": suggestion_min,
        }
        print(f"\nClean separation of {gap:0.3f}. Calibrated thresholds:")

    for name, value in thresholds.items():
        print(f"  {name} = {value}")

    if gate.mode != "embedding":
        # Numbers derived from the fallback would be meaningless for the
        # embedding gate the deployed app normally uses, so refuse to write.
        print(
            "\nNot writing scope_thresholds.json: these scores came from the "
            "lexical fallback.\nInstall sentence-transformers and rerun."
        )
        return

    with Path(THRESHOLD_FILE).open("w", encoding="utf-8") as handle:
        json.dump(thresholds, handle, indent=2)
        handle.write("\n")
    print(f"\nWrote {THRESHOLD_FILE}")


if __name__ == "__main__":
    main()
