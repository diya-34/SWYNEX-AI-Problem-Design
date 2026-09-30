"""
SWYNEX Technologies Internship - Task 1: AI Problem Design
Script: scripts/classify_demo.py
Description: CLI demonstration for Student Announcement Intelligence System 
             (Classification, Priority Scoring, and Entity Extraction).
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
from model.nlp_utils import extract_information, calculate_priority, generate_extractive_summary

def load_trained_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vec_path = os.path.join(base_dir, "model", "vectorizer.pkl")
    clf_path = os.path.join(base_dir, "model", "classifier.pkl")
    
    if os.path.exists(vec_path) and os.path.exists(clf_path):
        return joblib.load(vec_path), joblib.load(clf_path), True
    return None, None, False

def main():
    print("=" * 75)
    print(" SWYNEX Technologies Internship - Task 1: AI Problem Design")
    print(" Student Announcement Intelligence System CLI Demo")
    print("=" * 75)

    vectorizer, classifier, is_loaded = load_trained_model()

    if not is_loaded:
        print("\n[WARNING] Model artifacts not found. Please run 'python model/train_model.py' first.")
        return

    sample_tests = [
        ("The DBMS examination will be conducted on Monday at 10 AM in Hall B.", "Exam"),
        ("Submit your Machine Learning assignment by 5 October at 11:59 PM in Lab 3.", "Assignment"),
        ("All students must maintain at least 75 percent attendance to be eligible for exams.", "Attendance"),
        ("Applications for the Summer AI and Data Science Internship at TCS are now open.", "Internship"),
        ("Registration for HackFest 2025 annual national technical hackathon is officially open in the auditorium.", "Event")
    ]

    print(f"\n[+] Testing {len(sample_tests)} real-world sample announcements:\n")

    for idx, (sample_text, true_label) in enumerate(sample_tests, 1):
        vec = vectorizer.transform([sample_text])
        pred_label = classifier.predict(vec)[0]
        probs = classifier.predict_proba(vec)[0]
        confidence = max(probs)
        
        info = extract_information(sample_text)
        prio = calculate_priority(sample_text, pred_label)
        summary = generate_extractive_summary(sample_text, pred_label, info)

        print(f"  [{idx}] Input Announcement : \"{sample_text}\"")
        print(f"      Ground Truth Label : {true_label}")
        print(f"      AI Prediction      : [{pred_label.upper()}] (Probability: {confidence*100:.1f}%)")
        print(f"      Priority Level     : {prio['level']} ({prio['reason']})")
        print(f"      Extracted Metadata : Subject='{info['Subject']}', Date='{info['Date']}', Time='{info['Time']}', Venue='{info['Location']}', Deadline='{info['Deadline']}'")
        print(f"      Short Summary      : {summary}\n")

    print("=" * 75)
    print(" [OK] Real ML CLI evaluation complete!")
    print("=" * 75)

if __name__ == "__main__":
    main()
