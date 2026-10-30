import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

from feature_extraction import tfidf, fit_tfidf, transform_tfidf


# -------------------------
# LOAD DATASET
# -------------------------
data = pd.read_csv("data/processed/emails.csv")

texts = data["text"]
labels = data["label"]

# -------------------------
# TF-IDF FEATURES
# -------------------------
fit_tfidf(texts)
X = transform_tfidf(texts)

# -------------------------
# SPLIT DATA
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

# -------------------------
# TRAIN MODEL
# -------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------------
# PREDICT
# -------------------------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))

# -------------------------
# SAVE MODEL
# -------------------------
joblib.dump(model, "models/phishing_model.pkl")

print("Model saved successfully!")