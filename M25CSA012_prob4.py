import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Load Data (Focusing on Sports and Politics)
categories = ['rec.sport.hockey', 'rec.sport.baseball', 'talk.politics.mideast', 'talk.politics.guns']
data = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

# 2. Preprocessing & Feature Extraction (TF-IDF with N-grams)
# Use ngram_range=(1, 2) to include both single words and pairs
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)
X = tfidf.fit_transform(data.data)
y = data.target

# 3. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Define and Train ML Techniques
models = {
    "Naive Bayes": MultinomialNB(),
    "SVM": SVC(kernel='linear'),
    "Logistic Regression": LogisticRegression()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"--- {name} Results ---")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
    print(classification_report(y_test, predictions, target_names=data.target_names))