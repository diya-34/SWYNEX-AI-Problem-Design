"""
SWYNEX Technologies Internship - Task 1: AI Problem Design
Module: model/train_model.py
Description: Trains a real Natural Language Processing (NLP) text classification model 
             (TF-IDF + Logistic Regression) on student announcements and saves model artifacts.
"""

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def train_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "announcements.csv")
    model_dir = os.path.join(base_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 65)
    print(" SWYNEX Technologies Internship - Task 1: ML Model Training")
    print(" Architecture: TF-IDF Vectorizer + Logistic Regression")
    print("=" * 65)

    # 1. Load dataset
    print(f"\n[1/5] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"      Total records loaded: {len(df)}")
    print(f"      Categories: {list(df['label'].unique())}")
    print(f"      Class distribution:\n{df['label'].value_counts().to_string()}")

    # 2. Stratified Train / Test Split (80% Train, 20% Test)
    print("\n[2/5] Performing Stratified Train / Test Split (80% / 20%, random_state=42)...")
    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"      Training set size : {len(X_train)} samples (80%)")
    print(f"      Testing set size  : {len(X_test)} samples (20%)")

    # Save train & test splits for deterministic and reproducible evaluation
    train_df = pd.DataFrame({"text": X_train, "label": y_train})
    test_df = pd.DataFrame({"text": X_test, "label": y_test})
    train_df.to_csv(os.path.join(model_dir, "train_data.csv"), index=False)
    test_df.to_csv(os.path.join(model_dir, "test_data.csv"), index=False)
    print("      Saved train_data.csv and test_data.csv in model/ directory.")

    # 3. TF-IDF Feature Representation
    print("\n[3/5] Extracting TF-IDF Features (ngram_range=(1, 2), stop_words='english')...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
        min_df=1
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    print(f"      Vocabulary size: {len(vectorizer.vocabulary_)} n-gram features")

    # 4. Train Logistic Regression Classifier
    print("\n[4/5] Training Logistic Regression Classifier (C=1.0, max_iter=1000)...")
    classifier = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42
    )
    classifier.fit(X_train_tfidf, y_train)
    print("      Model training completed successfully.")

    # 5. Save Artifacts using Joblib
    vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
    classifier_path = os.path.join(model_dir, "classifier.pkl")

    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(classifier, classifier_path)

    print(f"\n[5/5] Saved Model Artifacts:")
    print(f"      [OK] Vectorizer saved to -> {vectorizer_path}")
    print(f"      [OK] Classifier saved to -> {classifier_path}")

    print("\n" + "=" * 65)
    print(" Model training pipeline complete! Next run: python evaluation/evaluate_model.py")
    print("=" * 65)

if __name__ == "__main__":
    train_model()
