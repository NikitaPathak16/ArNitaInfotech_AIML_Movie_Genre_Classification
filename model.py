import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Simple text cleaner
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z ]', '', text)
    return text

# Load dataset manually
texts = []
genres = []

with open("data/train_data.txt", encoding="utf-8") as file:
    for line in file:
        parts = line.split(":::")
        if len(parts) >= 4:
            genre = parts[2].strip().split("|")[0]
            plot = parts[3].strip()
            texts.append(clean_text(plot))
            genres.append(genre)

df = pd.DataFrame({"text": texts, "genre": genres})

# Dataset statistics
total_movies = len(df)
genre_counts = df["genre"].value_counts().to_dict()

# Vectorization (Count-based, not TF-IDF)
vectorizer = CountVectorizer(max_features=3000)
X = vectorizer.fit_transform(df["text"])
y = df["genre"]

# Train model
model = MultinomialNB()
model.fit(X, y)

# Predict function
def predict_genre(plot):
    plot = clean_text(plot)
    vector = vectorizer.transform([plot])
    return model.predict(vector)[0]

# Dataset stats function
def dataset_stats():
    return total_movies, genre_counts