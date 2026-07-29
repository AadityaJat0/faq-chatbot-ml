"""Owner-only review of pending knowledge suggestions.

This page replaces hand-typing `approved` or `rejected` into the Supabase
Table Editor. It is not part of the public chatbot: the chat itself stays
anonymous and login-free, and this page is reachable only by someone who
knows the owner password held in Streamlit Secrets.

Deliberately NOT implemented here: approving does not retrain the model and
does not write to GitHub. See the notice rendered at the top of the page.
"""

from __future__ import annotations

import hmac

import streamlit as st

from suggestion_store import (
    APPROVED,
    PENDING,
    REJECTED,
    SuggestionReviewStore,
    SuggestionStoreError,
)

PAGE_SIZE = 10
MAX_ATTEMPTS = 5

st.set_page_config(page_title="ResolveBot | Owner Review", page_icon="🔐", layout="wide")


# ---------------------------------------------------------------------------
# Access control
# ---------------------------------------------------------------------------
def configured_owner_password() -> str | None:
    try:
        return str(st.secrets["owner"]["review_password"]).strip() or None
    except Exception:  # noqa: BLE001 - missing secrets section
        return None


def owner_is_authenticated() -> bool:
    """Gate the page behind a single shared owner password.

    A constant-time comparison is used so the check does not leak the
    password's length or prefix through response timing.
    """
    if st.session_state.get("owner_authenticated"):
        return True

    expected = configured_owner_password()
    if expected is None:
        st.error(
            "Owner review is not configured. Add an `[owner]` section with "
            "`review_password` to Streamlit Secrets, then reload this page."
        )
        st.stop()

    attempts = st.session_state.get("owner_attempts", 0)
    if attempts >= MAX_ATTEMPTS:
        st.error("Too many incorrect attempts. Reload the page to try again.")
        st.stop()

    st.title("🔐 Owner review")
    st.caption("This page is for the project owner. The public chatbot needs no login.")
    entered = st.text_input("Owner password", type="password")
    if st.button("Unlock", type="primary"):
        if entered and hmac.compare_digest(entered, expected):
            st.session_state.owner_authenticated = True
            st.session_state.owner_attempts = 0
            st.rerun()
        else:
            st.session_state.owner_attempts = attempts + 1
            st.error("Incorrect password.")
    st.stop()
    return False


def review_store() -> SuggestionReviewStore:
    try:
        config = st.secrets.get("supabase", {})
        url = str(config.get("url", "")).strip()
        key = str(config.get("service_role_key", "")).strip()
    except Exception:  # noqa: BLE001
        url = key = ""
    if not url or not key:
        st.error("Supabase credentials are not configured, so suggestions cannot be loaded.")
        st.stop()
    return SuggestionReviewStore(url, key)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
owner_is_authenticated()
store = review_store()

st.title("🔐 Owner review — knowledge suggestions")

st.info(
    "**Approving records a decision; it does not publish anything.** ResolveBot "
    "trains only from `faq_data.json`. To make an approved answer live: add it "
    "to `faq_data.json`, commit and push to `main`, and let Streamlit Cloud "
    "redeploy so the model is rebuilt.",
    icon="ℹ️",
)

with st.sidebar:
    st.subheader("Session")
    if st.button("Lock this page", use_container_width=True):
        st.session_state.owner_authenticated = False
        st.rerun()
    st.divider()
    st.caption("Counts refresh when the page reloads.")
    try:
        for status, total in store.count_by_status().items():
            st.metric(status.capitalize(), total)
    except SuggestionStoreError:
        st.caption("Counts are unavailable right now.")

filter_column, page_column = st.columns([3, 1])
with filter_column:
    status_filter = st.radio(
        "Show",
        options=[PENDING, APPROVED, REJECTED, "all"],
        horizontal=True,
        format_func=lambda value: "All" if value == "all" else value.capitalize(),
    )
with page_column:
    page_number = st.number_input("Page", min_value=1, value=1, step=1)

offset = (int(page_number) - 1) * PAGE_SIZE

try:
    suggestions = store.list_suggestions(
        status=None if status_filter == "all" else status_filter,
        limit=PAGE_SIZE,
        offset=offset,
    )
except SuggestionStoreError as error:
    st.error(str(error))
    st.stop()

if not suggestions:
    st.success("Nothing to review on this page.")
    st.stop()


def apply_decision(identifier: str, status: str, note: str) -> None:
    """Update Supabase, then rerun so the list reflects the new status."""
    try:
        store.set_status(identifier, status, note or None)
    except (SuggestionStoreError, ValueError) as error:
        st.session_state.review_feedback = ("error", str(error))
    else:
        st.session_state.review_feedback = (
            "success",
            f"Suggestion marked {status}.",
        )
    st.rerun()


feedback = st.session_state.pop("review_feedback", None)
if feedback:
    level, message = feedback
    (st.success if level == "success" else st.error)(message)

for suggestion in suggestions:
    with st.container(border=True):
        st.markdown(f"**Question asked:** {suggestion.question or '_(empty)_'}")
        st.markdown(f"**Suggested answer:** {suggestion.suggested_answer or '_(empty)_'}")

        meta = [f"Status: `{suggestion.status}`"]
        if suggestion.created_at:
            meta.append(f"Submitted: {suggestion.created_at}")
        if suggestion.reviewed_at:
            meta.append(f"Reviewed: {suggestion.reviewed_at}")
        st.caption(" · ".join(meta))
        if suggestion.reviewer_note:
            st.caption(f"Note: {suggestion.reviewer_note}")

        if not suggestion.is_pending:
            continue

        note = st.text_input(
            "Reviewer note (optional)",
            key=f"note-{suggestion.identifier}",
            placeholder="Why this was approved or rejected",
        )
        approve_column, reject_column = st.columns(2)
        with approve_column:
            if st.button(
                "Approve",
                key=f"approve-{suggestion.identifier}",
                type="primary",
                use_container_width=True,
            ):
                apply_decision(suggestion.identifier, APPROVED, note)
        with reject_column:
            if st.button(
                "Reject",
                key=f"reject-{suggestion.identifier}",
                use_container_width=True,
            ):
                apply_decision(suggestion.identifier, REJECTED, note)

st.caption(
    "Future scope: an 'Approve and create pull request' action could open a "
    "GitHub Pull Request adding the approved entry to `faq_data.json`, leaving "
    "the merge decision with the owner. That automation is not implemented."
)
