# FAQ Chatbot - Terminal Test

## Importing Libraries

import joblib                                                           # Import joblib to load the saved vectorizer, model and answer lookup

## Loading the Saved Model

vectorizer = joblib.load("model/vectorizer.pkl")                        # Load the fitted TfidfVectorizer saved during training
lr = joblib.load("model/classifier.pkl")                                # Load the trained LogisticRegression model
answers = joblib.load("model/answers.pkl")                              # Load the intent-to-answer lookup dictionary

CONFIDENCE_THRESHOLD = 0.10                                              # Minimum confidence required before trusting a real-topic prediction (out_of_scope is handled separately below, not by this threshold)

## Chat Loop

print("FAQ Chatbot (terminal test) - type 'quit' to exit\n")            # Print a short instruction before the loop starts

while True:                                                              # Keep asking for input until the user quits
    question = input("You: ")                                          # Read the user's typed question
    if question.lower() == "quit":                                     # Check whether the user wants to exit
        print("Bot: Goodbye!")                                         # Print a goodbye message
        break                                                          # Exit the loop

    question_vector = vectorizer.transform([question])                 # Transform the typed question into the same TF-IDF feature space as training
    probabilities = lr.predict_proba(question_vector)[0]                # Predict probabilities across all classes, including out_of_scope
    best_index = probabilities.argmax()                                 # Find the index of the highest-probability class
    best_intent = lr.classes_[best_index]                               # Look up the actual intent name at that index
    confidence = probabilities[best_index]                              # Store the confidence value for the predicted class

    print(f"   (predicted intent: {best_intent}, confidence: {confidence:.3f})")   # Print the predicted intent and confidence for this question

    if best_intent == "out_of_scope":                                  # Check whether the model recognized this as an off-topic question
        print(f"Bot: {answers['out_of_scope']}\n")                     # Print the redirect message for off-topic questions
    elif confidence >= CONFIDENCE_THRESHOLD:                            # Otherwise, check whether the model is confident enough in a real-topic prediction
        print(f"Bot: {answers[best_intent]}\n")                        # Print the stored answer for the predicted intent
    else:                                                                # Otherwise, the model isn't confident in any real intent
        print("Bot: Not confident about that one yet - Step 4 will handle this case.\n")   # Print a placeholder message until Step 4 adds the learning fallback