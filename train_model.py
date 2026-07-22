# Intent Classification Model

## Importing Libraries

import json                                                             # Import json library to load the FAQ dataset
import pandas as pd                                                     # Import pandas library for data manipulation
from sklearn.feature_extraction.text import TfidfVectorizer             # Import TfidfVectorizer from scikit-learn (Reference: TfidfVectorizer class https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
from sklearn.linear_model import LogisticRegression                     # Import LogisticRegression from scikit-learn
import joblib                                                           # Import joblib to save and load the trained model and vectorizer
import os                                                                # Import os to create the folder that stores the saved model files

## Loading the FAQ dataset

with open("faq_data.json") as f:                                        # Open the FAQ dataset containing question, intent and answer for each entry
    data = json.load(f)                                                 # Load the JSON data into a list of dictionaries

df = pd.DataFrame(data)                                                 # Convert the list of dictionaries into a DataFrame for easier handling
print(f"Loaded {len(df)} training examples across {df['intent'].nunique()} intents")   # Print how many examples and how many distinct intents were loaded

## Data Preprocessing

X = df["question"]                                                      # Extract the question text as the feature variable
Y = df["intent"]                                                        # Extract the intent label as the target variable

vectorizer = TfidfVectorizer()                                          # Initialize TfidfVectorizer. Converts each question into a row of word-importance scores (Reference: TfidfVectorizer class https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
X_vectorized = vectorizer.fit_transform(X)                               # Fit the vectorizer on the questions and transform them into numeric feature vectors (Reference: fit_transform method https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer.fit_transform)

## Model Instantiation of Intent Classification and Model training

lr = LogisticRegression(max_iter=1000)                                  # Initialize LogisticRegression with a higher max_iter so it reliably converges across this many intent classes (Reference: LogisticRegression class https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
lr.fit(X_vectorized, Y)                                                 # Fit the logistic regression model using the vectorized questions and their intents (Reference: fit method https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.fit)

## Saving the Model

os.makedirs("model", exist_ok=True)                                     # Create the 'model' folder if it does not already exist
joblib.dump(vectorizer, "model/vectorizer.pkl")                         # Save the fitted vectorizer so new questions can be transformed the same way later
joblib.dump(lr, "model/classifier.pkl")                                 # Save the trained LogisticRegression model

answer_lookup = df.drop_duplicates(subset="intent").set_index("intent")["answer"].to_dict()   # Build a dictionary mapping each intent to its stored answer
joblib.dump(answer_lookup, "model/answers.pkl")                         # Save the intent-to-answer lookup dictionary

print("Model trained and saved to the 'model' folder.")                 # Confirm that training and saving completed successfully