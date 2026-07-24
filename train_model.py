# Intent Classification Model

## Importing Libraries

import json                                                             # Import json library to load the FAQ dataset
import pandas as pd                                                     # Import pandas library for data manipulation
from sklearn.feature_extraction.text import TfidfVectorizer             # Import TfidfVectorizer from scikit-learn (Reference: TfidfVectorizer class https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
from sklearn.linear_model import LogisticRegression                     # Import LogisticRegression from scikit-learn
import joblib                                                           # Import joblib to save and load the trained model and vectorizer
from pathlib import Path                                                 # Import Path for reliable file paths on local machines and Streamlit Cloud

## Training Function

def train_model(data_path="faq_data.json"):
    data_path = Path(data_path)
    with data_path.open() as f:                                         # Open the FAQ dataset containing question, intent and answer for each entry
        data = json.load(f)                                             # Load the JSON data into a list of dictionaries

    df = pd.DataFrame(data)                                             # Convert the list of dictionaries into a DataFrame for easier handling
    print(f"Loaded {len(df)} training examples across {df['intent'].nunique()} intents")   # Print how many examples and how many distinct intents were loaded

    ## Data Preprocessing

    X = df["question"]                                                  # Extract the question text as the feature variable
    Y = df["intent"]                                                    # Extract the intent label as the target variable

    vectorizer = TfidfVectorizer()                                      # Initialize TfidfVectorizer. Converts each question into a row of word-importance scores (Reference: TfidfVectorizer class https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
    X_vectorized = vectorizer.fit_transform(X)                           # Fit the vectorizer on the questions and transform them into numeric feature vectors

    ## Model Instantiation of Intent Classification and Model training

    lr = LogisticRegression(max_iter=1000, class_weight='balanced')                              # Initialize LogisticRegression with a higher max_iter so it reliably converges across this many intent classes (Reference: LogisticRegression class https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
    lr.fit(X_vectorized, Y)                                             # Fit the logistic regression model using the vectorized questions and their intents (Reference: fit method https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.fit)

    answer_lookup = df.drop_duplicates(subset="intent").set_index("intent")["answer"].to_dict()   # Build a dictionary mapping each intent to its stored answer
    return vectorizer, lr, answer_lookup                                # Return the trained objects directly so the Streamlit app needs no model files

def train_and_save(data_path="faq_data.json", model_dir="model"):
    vectorizer, lr, answer_lookup = train_model(data_path)              # Train from the source JSON before optionally writing local artifacts
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)                        # Create the model folder if it does not already exist
    joblib.dump(vectorizer, model_dir / "vectorizer.pkl")               # Save the fitted vectorizer so the terminal script can reuse it later
    joblib.dump(lr, model_dir / "classifier.pkl")                       # Save the trained LogisticRegression model
    joblib.dump(answer_lookup, model_dir / "answers.pkl")               # Save the intent-to-answer lookup dictionary

    print("Model trained and saved to the 'model' folder.")            # Confirm that training and saving completed successfully
    return vectorizer, lr, answer_lookup

## Run Directly

if __name__ == "__main__":                                              # Only runs this block when the script is executed directly, not when imported
    train_and_save()                                                    # Train and save the model using the default file paths
