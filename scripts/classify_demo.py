"""
SWYNEX Technologies Internship - Task 1: AI Problem Design
Script: scripts/classify_demo.py
Description: CLI demonstration for Student Announcement Classification using the trained ML model.
"""

import os
import joblib

def load_trained_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vec_path = os.path.join(base_dir, "model", "vectorizer.pkl")
    clf_path = os.path.join(base_dir, "model", "classifier.pkl")
    
    if os.path.exists(vec_path) and os.path.exists(clf_path):
        return joblib.load(vec_path), joblib.load(clf_path), True
    return None, None, False

def main():
    print("=" * 70)
    print(" SWYNEX Technologies Internship - Task 1: AI Problem Design")
    print(" Real ML Classifier CLI Demo (TF-IDF + Logistic Regression)")
    print("=" * 70)

    vectorizer, classifier, is_loaded = load_trained_model()

    if not is_loaded:
        print("\n[WARNING] Model artifacts not found. Please run 'python model/train_model.py' first.")
        return

    sample_tests = [
        ("The DBMS examination will be conducted on Monday morning in Hall B.", "Exam"),
        ("Submit your Machine Learning lab assignment before Friday at 11:59 PM.", "Assignment"),
        ("All students must maintain at least 75 percent attendance for eligibility.", "Attendance"),
        ("Applications for the Summer AI and Data Science Internship are now open.", "Internship"),
        ("Registration for HackFest 2025 national technical hackathon is open!", "Event")
    ]

    print(f"\n[+] Testing {len(sample_tests)} real-world sample announcements:\n")

    for idx, (sample_text, true_label) in enumerate(sample_tests, 1):
        vec = vectorizer.transform([sample_text])
        pred_label = classifier.predict(vec)[0]
        probs = classifier.predict_proba(vec)[0]
        confidence = max(probs)

        print(f"  {idx}. Input        : \"{sample_text}\"")
        print(f"     Ground Truth : {true_label}")
        print(f"     ML Predicted : [{pred_label.upper()}] (Posterior Probability: {confidence*100:.1f}%)\n")

    print("=" * 70)
    print(" [OK] Real ML CLI evaluation complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
