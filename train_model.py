import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("dataset.csv")

# Keep only needed columns
df = df[["Resume_str", "Category"]]

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\\S+", " ", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\\s+", " ", text)
    return text

df["Resume_str"] = df["Resume_str"].apply(clean_text)

# Features and labels
X = df["Resume_str"]
y = df["Category"]

# Convert text to numbers
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_tfidf = tfidf.fit_transform(X)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

# Save model
joblib.dump(model, "model/resume_classifier.pkl")
joblib.dump(tfidf, "model/tfidf.pkl")

print("Model Saved Successfully")