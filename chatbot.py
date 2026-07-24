# FAQ Chatbot - Terminal Test

## Importing Libraries

from train_model import train_model                                      # Train from the source FAQ data so the terminal app also works on a fresh clone

## Building the Model

vectorizer, lr, answers = train_model()                                  # Build the model directly from faq_data.json; no pickle files are required

CONFIDENCE_THRESHOLD = 0.10                                              # Minimum top-class probability required before considering a real-topic prediction trustworthy
MARGIN_THRESHOLD = 0.05                                                  # Minimum gap required between the top and second-best prediction

## Chat Loop

print("FAQ Chatbot (terminal test) - type 'quit' to exit\n")            # Print a short instruction before the loop starts

while True:                                                              # Keep asking for input until the user quits
    question = input("You: ")                                          # Read the user's typed question
    if question.lower() == "quit":                                     # Check whether the user wants to exit
        print("Bot: Goodbye!")                                         # Print a goodbye message
        break                                                          # Exit the loop

    question_vector = vectorizer.transform([question])                 # Transform the typed question into the same TF-IDF feature space as training
    probabilities = lr.predict_proba(question_vector)[0]                # Predict probabilities across all classes, including out_of_scope
    sorted_probs = sorted(probabilities, reverse=True)                  # Sort every class probability from highest to lowest
    best_index = probabilities.argmax()                                 # Find the index of the highest-probability class
    best_intent = lr.classes_[best_index]                               # Look up the actual intent name at that index
    confidence = sorted_probs[0]                                        # The top probability
    margin = sorted_probs[0] - sorted_probs[1]                          # Gap between the top and second-best pick

    print(f"   (predicted intent: {best_intent}, confidence: {confidence:.3f}, margin: {margin:.3f})")   # Print prediction, confidence and margin

    if best_intent == "out_of_scope":                                  # Check whether the model recognized this as an off-topic question
        print(f"Bot: {answers['out_of_scope']}\n")                     # Print the redirect message for off-topic questions
    elif confidence >= CONFIDENCE_THRESHOLD and margin >= MARGIN_THRESHOLD:   # Only trust the prediction if it clears both checks
        print(f"Bot: {answers[best_intent]}\n")                        # Print the stored answer for the predicted intent
    else:                                                                # Otherwise, not reliably confident
        print("Bot: Not confident about that one yet - this terminal version does not learn, use app.py for that.\n")   # Clarify that learning now lives only in app.py
