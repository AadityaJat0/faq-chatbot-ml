"""ResolveBot's Streamlit interface."""

from __future__ import annotations

import json
import secrets
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import extra_streamlit_components as stx
import streamlit as st

from history_store import HistoryStoreError, SupabaseHistoryStore
from train_model import train_model


APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "faq_data.json"
HISTORY_COOKIE_NAME = "resolvebot_history_token"
HISTORY_COOKIE_LIFETIME = timedelta(days=180)
MAX_INPUT_CHARS = 2_000


st.set_page_config(
    page_title="ResolveBot | Phone Support Chatbot",
    page_icon="🤖",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stChatMessage { border-radius: 14px; padding: 4px 2px; }
    div[data-testid="stChatMessageContent"] { font-size: 0.95rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def dataset_hash() -> str:
    """Invalidate the cached model when tracked training data changes."""
    return sha256(DATA_PATH.read_bytes()).hexdigest()


@st.cache_resource(show_spinner="Preparing ResolveBot...")
def build_model(current_dataset_hash: str):
    """Train from the tracked FAQ data; no generated model files are needed."""
    del current_dataset_hash  # The cache key is intentionally the data hash.
    return train_model(data_path=DATA_PATH)


def refresh_session_model() -> None:
    current_dataset_hash = dataset_hash()
    vectorizer, classifier, answers = build_model(current_dataset_hash)
    st.session_state.vectorizer = vectorizer
    st.session_state.classifier = classifier
    st.session_state.answers = answers
    st.session_state.model_dataset_hash = current_dataset_hash


@st.cache_resource(show_spinner=False)
def connect_history_store(url: str, service_role_key: str) -> SupabaseHistoryStore:
    """Create one reusable, server-side Supabase client."""
    return SupabaseHistoryStore(url, service_role_key)


def configured_history_store() -> SupabaseHistoryStore | None:
    """Return persistent storage when private Supabase Secrets are configured."""
    try:
        config = st.secrets.get("supabase", {})
        url = str(config.get("url", "")).strip()
        service_role_key = str(config.get("service_role_key", "")).strip()
    except Exception:
        return None

    if not url or not service_role_key:
        return None
    return connect_history_store(url, service_role_key)


def cookie_manager():
    """Read browser cookies on every rerun.

    Components report their browser-side value back to Streamlit on a rerun, so
    keeping the component object in session state would leave it with its first
    (usually empty) cookie snapshot.
    """
    return stx.CookieManager(key="resolvebot-cookie-manager")


def _cookie_is_secure() -> bool:
    """Use Secure cookies on HTTPS deployments while keeping local testing usable."""
    try:
        return st.context.url.startswith("https://")
    except Exception:
        return False


def _save_history_token(cookie_store, token: str) -> None:
    write_number = st.session_state.get("history_cookie_write_number", 0) + 1
    st.session_state.history_cookie_write_number = write_number
    cookie_store.set(
        HISTORY_COOKIE_NAME,
        token,
        key=f"resolvebot-history-token-{write_number}",
        path="/",
        expires_at=datetime.now(timezone.utc) + HISTORY_COOKIE_LIFETIME,
        secure=_cookie_is_secure(),
        same_site="lax",
    )


def history_access_token(cookie_store) -> str:
    """Get or create the browser-held secret used to find one saved chat."""
    browser_token = cookie_store.get(HISTORY_COOKIE_NAME)

    # On a brand-new Streamlit session the component first returns its default
    # value before the browser has reported its real cookies. Render only that
    # component once, then let its callback rerun the app. This prevents an
    # existing visitor token from being replaced before it can be read.
    if "cookie_read_complete" not in st.session_state and not browser_token:
        st.session_state.cookie_read_complete = True
        st.stop()

    session_token = st.session_state.get("history_access_token")
    if browser_token:
        if browser_token != session_token:
            st.session_state.history_access_token = browser_token
            st.session_state.pop("history_loaded_for_mode", None)
            st.session_state.messages = []
        return browser_token

    if session_token:
        return session_token

    token = secrets.token_urlsafe(32)
    _save_history_token(cookie_store, token)
    st.session_state.history_access_token = token
    return token


def replace_history_access_token(cookie_store) -> str:
    """Forget the deleted conversation and begin with a fresh browser token."""
    token = secrets.token_urlsafe(32)
    _save_history_token(cookie_store, token)
    st.session_state.history_access_token = token
    return token


def new_message(role: str, content: str, badge: str | None = None) -> dict[str, str]:
    """Create a chat message, optionally keeping its prediction indicator."""
    message = {
        "id": str(uuid4()),
        "role": role,
        "content": content,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if badge:
        message["badge"] = badge
    return message


def initialize_chat_history(
    history_store: SupabaseHistoryStore | None, access_token: str
) -> None:
    """Load a persisted transcript once, without breaking a usable live session."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # A cookie component can cause a rerun while the session is being created.
    # Record which storage mode was loaded so that reruns are safe and a later
    # database configuration is still picked up instead of remaining session-only.
    storage_mode = "saved" if history_store is not None else "session_only"
    if st.session_state.get("history_loaded_for_mode") == storage_mode:
        st.session_state.setdefault("history_status", storage_mode)
        return

    st.session_state.history_loaded_for_mode = storage_mode
    if history_store is None:
        st.session_state.history_status = "session_only"
        return

    try:
        st.session_state.messages = history_store.load_messages(access_token)
        if restore_missing_badges(st.session_state.messages):
            # Upgrade chats saved before prediction badges were persisted.
            history_store.save_messages(access_token, st.session_state.messages)
        st.session_state.history_status = "saved"
    except HistoryStoreError:
        st.session_state.history_status = "unavailable"


def persist_messages(
    history_store: SupabaseHistoryStore | None,
    access_token: str,
    messages: list[dict[str, str]],
) -> bool:
    """Save the newest messages, while leaving the chat responsive on failure."""
    if history_store is None:
        return False

    try:
        history_store.save_messages(access_token, messages)
        st.session_state.history_status = "saved"
        return True
    except (HistoryStoreError, ValueError):
        st.session_state.history_status = "unavailable"
        return False


def confidence_badge_html(
    intent: str, confidence: float, margin: float, is_confident: bool
) -> str:
    """Build the green, amber, or red prediction indicator shown below a reply."""
    if intent == "out_of_scope":
        color, background, label = "#b45309", "#fef3c7", "Off-topic"
    elif is_confident:
        color, background, label = "#15803d", "#dcfce7", "Confident match"
    else:
        color, background, label = "#b91c1c", "#fee2e2", "Low confidence"

    return (
        f'<div style="display:inline-block;background:{background};color:{color};'
        "border-radius:999px;padding:2px 12px;font-size:0.75rem;"
        'font-weight:600;margin:6px 0;">'
        f"{label} &middot; {intent} &middot; confidence {confidence:.2f}"
        f" &middot; margin {margin:.2f}</div>"
    )


def review_badge_html() -> str:
    """Give suggestion acknowledgements their own persistent amber indicator."""
    return (
        '<div style="display:inline-block;background:#fef3c7;color:#b45309;'
        "border-radius:999px;padding:2px 12px;font-size:0.75rem;"
        'font-weight:600;margin:6px 0;">Suggestion saved for review</div>'
    )


def resolve_reply(prompt: str) -> tuple[str, str | None, str]:
    """Return a reply, optional feedback question, and its prediction indicator."""
    vectorizer = st.session_state.vectorizer
    classifier = st.session_state.classifier
    answers = st.session_state.answers

    question_vector = vectorizer.transform([prompt])
    probabilities = classifier.predict_proba(question_vector)[0]
    sorted_probabilities = sorted(probabilities, reverse=True)
    best_index = probabilities.argmax()
    best_intent = classifier.classes_[best_index]
    confidence = sorted_probabilities[0]
    margin = sorted_probabilities[0] - sorted_probabilities[1]

    is_confident = confidence >= 0.10 and margin >= 0.05
    badge = confidence_badge_html(best_intent, confidence, margin, is_confident)
    if best_intent == "out_of_scope":
        return answers["out_of_scope"], None, badge
    if is_confident:
        return answers[best_intent], None, badge

    return (
        "I don't have a verified answer for that yet. If you'd like to help "
        "improve ResolveBot, type the answer you expected and I’ll save it for "
        "the project owner to review.",
        prompt,
        badge,
    )


def restore_missing_badges(messages: list[dict[str, str]]) -> bool:
    """Restore indicators for chats created before badges were stored in Supabase."""
    changed = False
    latest_question = None
    for message in messages:
        if message["role"] == "user":
            latest_question = message["content"]
            continue
        if message.get("badge"):
            continue
        if message["content"].startswith("Thanks"):
            message["badge"] = review_badge_html()
        elif latest_question:
            _, _, message["badge"] = resolve_reply(latest_question)
        else:
            continue
        changed = True
    return changed


if st.session_state.get("model_dataset_hash") != dataset_hash():
    refresh_session_model()

if "awaiting_feedback_for" not in st.session_state:
    # Preserve a user who happened to have the old live-learning state open
    # while this version was deployed.
    st.session_state.awaiting_feedback_for = st.session_state.pop(
        "awaiting_answer", None
    )

history_store = configured_history_store()
browser_cookies = cookie_manager()
access_token = history_access_token(browser_cookies)
initialize_chat_history(history_store, access_token)


st.title("🤖 ResolveBot — Phone Support Chatbot")
st.caption("Your phone support assistant for common device questions.")
if st.session_state.history_status == "saved":
    st.caption("✓ This chat is saved automatically for this browser.")
elif st.session_state.history_status == "unavailable":
    st.warning("This chat is visible now, but it could not be saved at the moment.")
else:
    st.caption("This chat is available for the current session.")
st.caption("Avoid sharing passwords, PINs, OTPs, or other sensitive information.")


with st.sidebar:
    st.subheader("Your chat")
    if st.session_state.history_status == "saved":
        st.success("Saved automatically")
        st.caption(
            "History returns on this browser and device. Clearing browser data or "
            "using a different device starts a separate chat."
        )
    elif st.session_state.history_status == "unavailable":
        st.warning("Saving is temporarily unavailable")
    else:
        st.info("Session-only chat")

    delete_label = "Delete saved chat" if history_store else "Clear chat"
    if st.button(delete_label, use_container_width=True):
        st.session_state.confirm_delete = True

    if st.session_state.get("confirm_delete"):
        st.warning("This permanently removes the messages shown in this chat.")
        confirm_col, cancel_col = st.columns(2)
        with confirm_col:
            delete_confirmed = st.button("Delete permanently", type="primary")
        with cancel_col:
            cancel_delete = st.button("Cancel")

        if cancel_delete:
            st.session_state.confirm_delete = False
            st.rerun()

        if delete_confirmed:
            try:
                if history_store is not None:
                    history_store.delete_history(access_token)
                st.session_state.messages = []
                st.session_state.awaiting_feedback_for = None
                st.session_state.confirm_delete = False
                st.session_state.history_loaded_for_mode = (
                    "saved" if history_store is not None else "session_only"
                )
                st.session_state.history_status = (
                    "saved" if history_store is not None else "session_only"
                )
                replace_history_access_token(browser_cookies)
                st.rerun()
            except HistoryStoreError:
                st.error("ResolveBot could not delete this chat. Please try again.")

    st.divider()
    with st.expander("About ResolveBot"):
        with DATA_PATH.open() as data_file:
            current_data = json.load(data_file)
        st.metric("Training examples", len(current_data))
        st.metric("Intents recognized", len({row["intent"] for row in current_data}))
        st.caption(
            "ResolveBot uses TF-IDF features and Logistic Regression to match "
            "common phone-support intents."
        )


for message in st.session_state.messages:
    avatar = "🧑" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.write(message["content"])
        if message.get("badge"):
            st.markdown(message["badge"], unsafe_allow_html=True)


clicked_suggestion = None
if not st.session_state.messages:
    st.write("Try one of these, or type your own question below:")
    suggestions = [
        "My battery drains fast",
        "Track my order",
        "Is there a student discount?",
        "How do I reset my password?",
    ]
    suggestion_columns = st.columns(2)
    for index, suggestion in enumerate(suggestions):
        with suggestion_columns[index % len(suggestion_columns)]:
            if st.button(suggestion, use_container_width=True):
                clicked_suggestion = suggestion


typed_prompt = st.chat_input("Ask ResolveBot a question...", max_chars=MAX_INPUT_CHARS)
prompt = clicked_suggestion or typed_prompt

if prompt:
    user_message = new_message("user", prompt)
    st.session_state.messages.append(user_message)
    with st.chat_message("user", avatar="🧑"):
        st.write(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        if st.session_state.awaiting_feedback_for is not None:
            feedback_question = st.session_state.awaiting_feedback_for
            st.session_state.awaiting_feedback_for = None
            badge = review_badge_html()
            if history_store is None:
                reply = (
                    "Thanks for the suggestion. The feedback service has not been "
                    "configured yet, so it could not be sent for review."
                )
            else:
                try:
                    history_store.save_feedback(access_token, feedback_question, prompt)
                    reply = (
                        "Thanks — your suggestion was saved for review. It will not "
                        "change ResolveBot's verified answers until it has been checked."
                    )
                except (HistoryStoreError, ValueError):
                    reply = "I couldn't save that suggestion right now. Please try again later."
            st.write(reply)
        else:
            reply, feedback_question, badge = resolve_reply(prompt)
            st.session_state.awaiting_feedback_for = feedback_question
            st.write(reply)

        if badge:
            st.markdown(badge, unsafe_allow_html=True)

    assistant_message = new_message("assistant", reply, badge)
    st.session_state.messages.append(assistant_message)
    persist_messages(history_store, access_token, [user_message, assistant_message])

    # A quick-start button is rendered before chat input. Refresh once after it
    # is used so those starter buttons disappear as soon as the chat begins.
    if clicked_suggestion is not None:
        st.rerun()
