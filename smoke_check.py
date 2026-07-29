"""End-to-end check of the real pipeline (classifier + lexical fallback gate)."""
import json
from train_model import train_model
from semantic_gate import build_domain_gate, decide_scope, VERIFIED, OFF_TOPIC, SUGGESTION

rows = json.load(open("faq_data.json"))
vec, clf, answers = train_model()
gate = build_domain_gate(rows, allow_embeddings=False)  # force fallback path
print(f"gate mode: {gate.mode}\n")

CASES = [
    ("A", OFF_TOPIC, ["swimming?", "what do you mean by dancing?",
                      "how do I cook pasta?", "who won the cricket match?",
                      "tell me a joke", "what is the weather like today"]),
    ("B", SUGGESTION, ["how do I turn on quick sharing?",
                       "my phone keeps overheating while charging",
                       "the fingerprint sensor is not working"]),
    ("C", VERIFIED, ["My battery drains fast", "How do I reset my password?",
                     "Where is my order?", "Hello"]),
]

failures = 0
for label, expected, questions in CASES:
    print(f"--- Group {label}: expect {expected} ---")
    for q in questions:
        p = clf.predict_proba(vec.transform([q]))[0]
        s = sorted(p, reverse=True)
        intent = clf.classes_[p.argmax()]
        d = decide_scope(str(intent), float(s[0]), float(s[0]-s[1]), gate.similarity(q), gate.mode)
        ok = "OK " if d.outcome == expected else "FAIL"
        if d.outcome != expected: failures += 1
        print(f"  {ok} {q!r:52} -> {d.outcome:10} intent={intent:24} conf={d.confidence:.2f} sim={d.similarity:.2f}")
    print()
print("FAILURES:", failures)
