# Spam Email Classifier

This project trains machine learning models to detect and filter spam emails based on email content.

## Features
- Uses **Naive Bayes** and **Support Vector Machines (SVM)** for classification
- TF-IDF vectorization for text feature extraction
- Evaluation with accuracy, precision, recall, and F1-score
- Saves trained models for future predictions

## Dataset
Provide a CSV file `spam_dataset.csv` with two columns:
- `text` → email content
- `label` → "spam" or "ham"

## Usage
```bash
pip install -r requirements.txt
python spam_classifier.py
