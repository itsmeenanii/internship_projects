import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ---------- LOAD DATASET ----------
# Make sure you have spam_dataset.csv with columns: 'text', 'label'
data = pd.read_csv("spam_dataset.csv")

# ---------- FEATURE EXTRACTION ----------
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(data["text"])
y = data["label"]

# ---------- TRAIN / TEST SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- NAIVE BAYES ----------
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_preds = nb_model.predict(X_test)

print("📌 Naive Bayes Results")
print("Accuracy:", accuracy_score(y_test, nb_preds))
print(classification_report(y_test, nb_preds))

# ---------- SVM ----------
svm_model = LinearSVC()
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)

print("\n📌 SVM Results")
print("Accuracy:", accuracy_score(y_test, svm_preds))
print(classification_report(y_test, svm_preds))

# ---------- SAVE MODELS ----------
# Ensure 'models' folder exists
os.makedirs("models", exist_ok=True)

joblib.dump(nb_model, "models/spam_nb_model.pkl")
joblib.dump(svm_model, "models/spam_svm_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\n✅ Models saved successfully in 'models/' folder.")

# ---------- PREDICTION EXAMPLE ----------
sample_email = ["Congratulations! You won a lottery. Click here to claim."]
sample_features = vectorizer.transform(sample_email)

print("\n🔍 Sample Prediction (Naive Bayes):", nb_model.predict(sample_features)[0])
print("🔍 Sample Prediction (SVM):", svm_model.predict(sample_features)[0])
