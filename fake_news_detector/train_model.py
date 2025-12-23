import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
df = pd.read_csv("data/news.csv")

# Features & labels
X = df["text"]
y = df["label"]

# Vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_vec = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_vec, y)

# Save model and vectorizer
pickle.dump(model, open("model/fake_news_model.pkl", "wb"))
pickle.dump(vectorizer, open("model/tfidf_vectorizer.pkl", "wb"))

print("Model trained and saved successfully")
