# FAQ Chatbot - Streamlit UI

## Importing Libraries

import json                                                              # Import json to read and update the FAQ dataset when the bot learns something new
import re                                                                 # Import re to help turn a new question into a short intent name
from hashlib import sha256                                                # Hash the FAQ contents so a changed dataset gets a new cached model
from pathlib import Path                                                  # Resolve project files without depending on Streamlit's working directory
import streamlit as st                                        # Import streamlit to build the browser-based chat interface
from train_model import train_model                                     # Import the in-memory training function so fresh deployments need no model files

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "faq_data.json"

## Page Setup

st.set_page_config(page_title="Phone Support Chatbot", page_icon="📱", layout="centered")   # Set the browser tab title, icon and page width

st.markdown("""
<style>
.stChatMessage { border-radius: 14px; padding: 4px 2px; }
div[data-testid="stChatMessageContent"] { font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)                                             # Small cosmetic CSS tweak - purely visual, app works identically if this does not apply on some Streamlit versions

st.title("📱 Phone Support Chatbot")                                     # Display a title at the top of the page
st.caption("Ask about battery, warranty, orders, WiFi, and more — powered by a Logistic Regression intent classifier that learns new answers on the fly.")   # Short subtitle explaining what the bot does

## Building the Model from the Tracked Training Data

def dataset_hash():
    return sha256(DATA_PATH.read_bytes()).hexdigest()

@st.cache_resource(show_spinner="Preparing the chatbot model...")
def build_model(current_dataset_hash):
    # The hash invalidates this cache after a learned answer is saved. The web
    # app trains from versioned JSON and never needs pre-generated pickle files.
    return train_model(data_path=DATA_PATH)

def refresh_session_model():
    current_dataset_hash = dataset_hash()
    vectorizer, lr, answers = build_model(current_dataset_hash)
    st.session_state.vectorizer = vectorizer
    st.session_state.lr = lr
    st.session_state.answers = answers
    st.session_state.model_dataset_hash = current_dataset_hash

if st.session_state.get("model_dataset_hash") != dataset_hash():
    refresh_session_model()

if "messages" not in st.session_state:                                  # Only create the chat history the first time this session runs
    st.session_state.messages = []                                     # Start with an empty conversation

if "awaiting_answer" not in st.session_state:                           # Tracks whether the bot just asked the user to teach it something
    st.session_state.awaiting_answer = None                            # None means we are not currently waiting for a taught answer

CONFIDENCE_THRESHOLD = 0.10                                              # Minimum top-class probability required before even considering a real-topic prediction - left unchanged since raising it would also block the known-correct 0.119 return_policy match
MARGIN_THRESHOLD = 0.05                                                  # Minimum gap required between the top and second-best prediction - catches cases where the top score looks acceptable but the model was really just guessing among similar options

## Sidebar

with st.sidebar:                                                         # Sidebar panel with session controls and live model stats
    st.subheader("Session")                                             # Sidebar heading
    if st.button("Clear chat"):                                        # Button to reset the conversation without restarting the app
        st.session_state.messages = []                                 # Empty out the chat history
        st.session_state.awaiting_answer = None                        # Also reset the "waiting to be taught" state
        st.rerun()                                                     # Immediately refresh the page to reflect the cleared chat

    st.divider()                                                        # Visual separator before the stats block
    st.subheader("Model stats")                                        # Heading for the live stats section
    with DATA_PATH.open() as f:                                         # Open the current dataset to compute live stats
        current_data = json.load(f)                                    # Load it so we can count examples and intents
    st.metric("Training examples", len(current_data))                 # Show the total number of question-intent-answer rows
    st.metric("Intents recognized", len({row["intent"] for row in current_data}))   # Show the number of distinct intents, including any learned on the fly - this number visibly increases once you teach it something new

## Helper Function to Name a New Intent

def slugify_question(question, existing_intents):                      # Turn a brand-new question into a short, unique intent name
    words = re.findall(r"[a-zA-Z0-9]+", question.lower())               # Pull out just the alphanumeric words, lowercased
    base_slug = "_".join(words[:4]) or "custom_intent"                  # Join the first four words with underscores, fall back if the question had no usable words
    slug = base_slug                                                    # Start with the base slug
    counter = 2                                                         # Counter used only if the base slug is already taken
    while slug in existing_intents:                                    # Keep checking until we find a slug that is not already used
        slug = f"{base_slug}_{counter}"                                # Append a number to make it unique
        counter += 1                                                   # Increase the counter for the next attempt if needed
    return slug                                                        # Return the final, unique intent name

## Helper Function to Learn a New Answer

def learn_new_answer(question, answer):                                 # Save a brand-new question and answer, then retrain the model on the spot
    with DATA_PATH.open() as f:                                         # Open the current FAQ dataset
        data = json.load(f)                                            # Load it into a list of dictionaries

    existing_intents = {entry["intent"] for entry in data}              # Collect every intent name already in use
    new_intent = slugify_question(question, existing_intents)          # Generate a unique intent name for this new question

    data.append({"question": question, "intent": new_intent, "answer": answer})   # Add the new question, intent and answer as a new entry

    with DATA_PATH.open("w") as f:                                      # Open the file again, this time for writing
        json.dump(data, f, indent=2)                                   # Save the updated dataset back to disk in readable JSON format

    with st.spinner("Learning that one now..."):                       # Show a small spinner while retraining, even though it takes under a second
        refresh_session_model()                                         # Retrain immediately from the updated JSON data

## Helper Function to Build the Confidence Badge

def confidence_badge_html(intent, confidence, margin, is_confident):    # Build a small colored HTML badge summarizing the model's prediction
    if intent == "out_of_scope":                                       # Off-topic questions get an amber badge
        color, bg, label = "#b45309", "#fef3c7", "off-topic"           # Amber text/background and label
    elif is_confident:                                                  # Confident, trusted real-topic predictions get a green badge
        color, bg, label = "#15803d", "#dcfce7", "confident match"     # Green text/background and label
    else:                                                                # Anything that did not clear both checks gets a red badge
        color, bg, label = "#b91c1c", "#fee2e2", "low confidence"      # Red text/background and label
    return (f'<div style="display:inline-block;background:{bg};color:{color};'   # Self-contained inline-styled HTML pill, does not depend on Streamlit internal class names
            f'border-radius:999px;padding:2px 12px;font-size:0.75rem;font-weight:600;margin:6px 0;">'
            f'{label} &middot; {intent} &middot; confidence {confidence:.2f} &middot; margin {margin:.2f}</div>')

## Displaying Past Messages

for message in st.session_state.messages:                               # Loop through every message saved so far in this session
    avatar = "🧑" if message["role"] == "user" else "📱"                 # Use a person avatar for the user and a phone avatar for the assistant
    with st.chat_message(message["role"], avatar=avatar):               # Render it in the correct chat bubble style
        st.write(message["content"])                                   # Display the message text
        if message.get("badge"):                                       # If this message has a stored confidence badge
            st.markdown(message["badge"], unsafe_allow_html=True)      # Display the colored badge — DELETE this "if" block (2 lines) to hide the debug info entirely

## Suggested Questions

clicked_suggestion = None                                                # Will hold whichever suggestion the user clicks, if any
if not st.session_state.messages:                                       # Only show quick-start suggestions before the conversation has started
    st.write("Try one of these, or type your own question below:")     # Small prompt above the suggestion buttons
    suggestions = ["My battery drains fast", "Track my order", "Is there a student discount?", "How do I reset my password?"]   # A handful of common, in-scope example questions
    suggestion_cols = st.columns(len(suggestions))                     # Lay the suggestion buttons out in equal-width columns
    for col, suggestion in zip(suggestion_cols, suggestions):           # Loop through each column and its matching suggestion text
        with col:                                                       # Place this button inside its column
            if st.button(suggestion, use_container_width=True):        # Show the suggestion as a clickable button
                clicked_suggestion = suggestion                        # Record which suggestion was clicked

## Chat Input and Response Logic

typed_prompt = st.chat_input("Ask a question...")                       # Display the chat input box at the bottom of the page
prompt = clicked_suggestion or typed_prompt                             # Use whichever came in - a clicked suggestion takes priority if both somehow fire in the same run

if prompt:                                                               # Only run this block when the user has actually typed something or clicked a suggestion
    st.session_state.messages.append({"role": "user", "content": prompt})   # Save the user's message to the chat history
    with st.chat_message("user", avatar="🧑"):                          # Render the user's message immediately
        st.write(prompt)                                               # Display what the user typed

    with st.chat_message("assistant", avatar="📱"):                     # Render the bot's response in its own chat bubble
        if st.session_state.awaiting_answer is not None:                # Check whether the bot is currently waiting to be taught an answer
            learn_new_answer(st.session_state.awaiting_answer, prompt) # Save the new question and answer, then retrain the model immediately
            reply = "Got it, I'll remember that from now on!"          # Confirm that the bot has learned the new answer
            st.write(reply)                                            # Display the confirmation
            st.session_state.messages.append({"role": "assistant", "content": reply})   # Save the confirmation to the chat history
            st.session_state.awaiting_answer = None                    # Stop waiting, since the bot has now learned the answer

        else:                                                            # Otherwise, this is a normal question, not a taught answer
            vectorizer = st.session_state.vectorizer                   # Use the current session's vectorizer
            lr = st.session_state.lr                                   # Use the current session's classifier
            answers = st.session_state.answers                        # Use the current session's answer lookup

            question_vector = vectorizer.transform([prompt])           # Transform the typed question into the same TF-IDF feature space as training
            probabilities = lr.predict_proba(question_vector)[0]        # Predict probabilities across all classes, including out_of_scope
            sorted_probs = sorted(probabilities, reverse=True)          # Sort every class probability from highest to lowest
            best_index = probabilities.argmax()                         # Find the index of the highest-probability class
            best_intent = lr.classes_[best_index]                       # Look up the actual intent name at that index
            confidence = sorted_probs[0]                                 # The top probability, same value as probabilities[best_index]
            margin = sorted_probs[0] - sorted_probs[1]                  # How much the top pick beat the second-best pick by - a small margin means the model was essentially guessing between options
            is_confident = confidence >= CONFIDENCE_THRESHOLD and margin >= MARGIN_THRESHOLD   # Only trust this prediction if it clears both the absolute floor and the margin check
            badge = confidence_badge_html(best_intent, confidence, margin, is_confident)   # Build the colored badge summarizing this prediction

            if best_intent == "out_of_scope":                          # Check whether the model recognized this as an off-topic question
                reply = answers["out_of_scope"]                        # Use the redirect message for off-topic questions
                st.write(reply)                                        # Display the redirect message
                st.markdown(badge, unsafe_allow_html=True)             # Display the confidence badge
                st.session_state.messages.append({"role": "assistant", "content": reply, "badge": badge})   # Save this response to the chat history

            elif is_confident:                                          # Otherwise, only proceed if the prediction cleared both checks above
                reply = answers[best_intent]                           # Use the stored answer for the predicted intent
                st.write(reply)                                        # Display the answer
                st.markdown(badge, unsafe_allow_html=True)             # Display the confidence badge
                st.session_state.messages.append({"role": "assistant", "content": reply, "badge": badge})   # Save this response to the chat history

            else:                                                        # Otherwise, the model does not reliably recognize this question
                reply = "I don't know that one yet — what should I say? Type your answer below and I'll remember it."   # Ask the user to teach the bot the correct answer
                st.write(reply)                                        # Display the request to teach the bot
                st.markdown(badge, unsafe_allow_html=True)              # Display the confidence badge
                st.session_state.messages.append({"role": "assistant", "content": reply, "badge": badge})   # Save this response to the chat history
                st.session_state.awaiting_answer = prompt               # Remember the original question so the next input is treated as its answer
