import joblib

# ---------- LOAD MODELS ----------
nb_model = joblib.load("models/spam_nb_model.pkl")
svm_model = joblib.load("models/spam_svm_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# ---------- FUNCTION TO PREDICT ----------
def classify_email(email_text):
    features = vectorizer.transform([email_text])
    nb_prediction = nb_model.predict(features)[0]
    svm_prediction = svm_model.predict(features)[0]
    return nb_prediction, svm_prediction

# ---------- TEST EXAMPLES ----------
emails = [
    "Congratulations! You won a lottery. Click here to claim.",
    "Reminder: Your meeting is scheduled at 10 AM tomorrow.",
    "Get cheap medicines at unbelievable prices!!!",
    "Hi Naresh, please find the attached project report."
]

for email in emails:
    nb_pred, svm_pred = classify_email(email)
    print(f"\nEmail: {email}")
    print(f"Naive Bayes Prediction: {nb_pred}")
    print(f"SVM Prediction: {svm_pred}")
