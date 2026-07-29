"""Regenerate faq_data.json.

The dataset is stored as a flat list of {question, intent, answer} rows, which
repeats each answer once per phrasing. Keeping the source of truth here --
one answer per intent, many questions -- makes it far harder to introduce a
typo into one copy of an answer and not the others.

Run:  python3 build_faq_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT_OF_SCOPE_ANSWER = (
    "I'm a phone support assistant, so I can only help with questions about "
    "your device \u2014 things like battery, warranty, orders, or software. "
    "Could you ask something along those lines?"
)

INTENTS: list[tuple[str, str, list[str]]] = [
    (
        "greeting",
        "Hi there! How can I help you with your device today?",
        ["Hello", "Hi", "Hey", "Good morning", "Hi bot"],
    ),
    (
        "battery_life",
        "Battery drain is often caused by background apps or screen brightness. "
        "Try enabling battery saver mode in Settings > Battery, and check which "
        "apps are using the most power.",
        [
            "Why does my battery drain so fast?",
            "My phone battery dies quickly",
            "Battery draining fast, what do I do",
            "How can I improve battery life",
            "Phone battery not lasting long",
        ],
    ),
    (
        "screen_repair",
        "You can book a screen repair through our service center or authorized "
        "partners. Repair cost depends on your model and warranty status, so "
        "check the Support app for an instant quote.",
        [
            "My screen is cracked, what do I do?",
            "Screen broke, can it be repaired?",
            "How much does screen replacement cost?",
            "Cracked display repair options",
            "My phone screen is damaged",
        ],
    ),
    (
        "warranty_period",
        "Standard warranty covers manufacturing defects for 12 months from the "
        "date of purchase. You can check your exact warranty status by entering "
        "your IMEI number on our support page.",
        [
            "How long is my warranty?",
            "What does the warranty cover?",
            "Is my phone still under warranty?",
            "Warranty period for this device",
            "When does my warranty expire?",
        ],
    ),
    (
        "return_policy",
        "Devices can be returned within 7 days of delivery if unused and in "
        "original packaging. Go to Orders > Return Item to start the process.",
        [
            "Can I return my phone?",
            "What is the return policy?",
            "How do I return this device?",
            "I want a refund for my order",
            "Return window for purchases",
            "Can I send this phone back?",
        ],
    ),
    (
        "order_tracking",
        "You can track your order in real time under My Orders > Track Package, "
        "using the tracking ID sent to your email.",
        [
            "Where is my order?",
            "How do I track my package?",
            "My order hasn't arrived yet",
            "Track my shipment",
            "When will my phone be delivered?",
        ],
    ),
    (
        "price_discount",
        "Current offers and student discounts are listed on our Deals page, "
        "updated weekly. Prices vary by storage variant and region.",
        [
            "Is there any discount available?",
            "What's the price of this phone?",
            "Any ongoing offers?",
            "Can I get a student discount?",
            "Price after discount",
        ],
    ),
    (
        "software_update",
        "Go to Settings > System > Software Update to check for and install the "
        "latest version. Make sure you are connected to WiFi and have at least "
        "50 percent battery.",
        [
            "How do I update my phone's software?",
            "Is a new software update available?",
            "My phone needs an update",
            "How to install the latest OS version",
            "Software update failing",
        ],
    ),
    (
        "forgot_password",
        "Click Forgot Password on the login screen and follow the link sent to "
        "your registered email to reset it securely.",
        [
            "I forgot my account password",
            "How do I reset my password?",
            "Can't log into my account",
            "Password reset link not working",
            "Locked out of my account",
        ],
    ),
    (
        "wifi_connection",
        "Try forgetting the network and reconnecting under Settings > WiFi. If it "
        "persists, restart your router and check for a software update.",
        [
            "My phone won't connect to WiFi",
            "WiFi keeps disconnecting",
            "How do I connect to a wireless network?",
            "WiFi not working on my device",
            "Can't join WiFi network",
        ],
    ),
    (
        "storage_full",
        "Go to Settings > Storage to see what's taking up space, and consider "
        "clearing cached data or moving photos to cloud storage.",
        [
            "My storage is full, what do I do?",
            "How do I free up space?",
            "Phone says storage almost full",
            "Not enough storage for update",
            "Clear storage space",
        ],
    ),
    (
        "camera_issue",
        "Try restarting your phone and clearing the camera app's cache under "
        "Settings > Apps > Camera > Storage > Clear Cache. If the issue "
        "persists, it may need a service check.",
        [
            "My camera isn't working",
            "Camera app keeps crashing",
            "Photos are blurry",
            "Camera won't focus",
            "Rear camera not opening",
        ],
    ),
    (
        "contact_support",
        "You can reach our support team via live chat in the Support app, or "
        "call our helpline, available 9 AM to 9 PM daily.",
        [
            "How do I talk to a human agent?",
            "I need to contact customer support",
            "Is there a support phone number?",
            "Connect me to a representative",
            "Talk to customer service",
        ],
    ),
    (
        "does_this_phone_support",
        "Yes, all our current models support ultra-fast 5G networks.",
        ["does this phone support 5G?"],
    ),
    (
        "out_of_scope",
        OUT_OF_SCOPE_ANSWER,
        [
            # --- original ten -------------------------------------------------
            "What is the capital of France?",
            "What's the weather like today?",
            "Can you tell me a joke?",
            "Who won the cricket match yesterday?",
            "What is 2 plus 2?",
            "Hi, how are you?",
            "What is your name?",
            "Recommend me a good movie",
            "How do I cook pasta?",
            "What time is it in New York?",
            # --- added: sports and hobbies -----------------------------------
            # Very short, single-word questions are included on purpose. They
            # were the exact failure case: with no shared vocabulary, every
            # intent scored near zero and `greeting` won by accident.
            "swimming?",
            "dancing",
            "What do you mean by dancing?",
            "Do you play cricket?",
            "How do I learn to swim?",
            "Best exercise for beginners",
            # --- added: food and cooking -------------------------------------
            "Give me a recipe for biryani",
            "What should I eat for dinner?",
            "How long do I boil eggs?",
            # --- added: weather ----------------------------------------------
            "Will it rain tomorrow?",
            "What is the temperature outside?",
            # --- added: school subjects --------------------------------------
            "Explain photosynthesis",
            "Help me with my maths homework",
            "What is the formula for area of a circle?",
            # --- added: animals ----------------------------------------------
            "What do cats eat?",
            "Are dogs colour blind?",
            "Tell me about elephants",
            # --- added: unrelated general conversation ------------------------
            "What is the meaning of life?",
            "Are you a real person?",
            "Sing me a song",
            "Who is the prime minister?",
            "Tell me something interesting",
            # --- added: entertainment ----------------------------------------
            "Which series should I watch tonight?",
            "Who acted in that new film?",
            "Suggest some good music",
        ],
    ),
]


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for intent, answer, questions in INTENTS:
        for question in questions:
            rows.append({"question": question, "intent": intent, "answer": answer})
    return rows


def main() -> None:
    rows = build_rows()
    target = Path(__file__).resolve().parent / "faq_data.json"
    with target.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    intents = {row["intent"] for row in rows}
    in_scope = [row for row in rows if row["intent"] != "out_of_scope"]
    print(f"Wrote {len(rows)} rows across {len(intents)} intents")
    print(f"  in-scope rows:     {len(in_scope)}")
    print(f"  out_of_scope rows: {len(rows) - len(in_scope)}")


if __name__ == "__main__":
    main()
