"""
SWYNEX Technologies Internship - Task 1: AI Problem Design
Script: classify_demo.py
Description: Interactive demonstration for Student Announcement Classification.
"""

import os
import csv

CATEGORIES = ["Exam", "Assignment", "Attendance", "Internship", "Event"]

# Indicative keyword patterns for demo simulation based on problem design
KEYWORD_RULES = {
    "Exam": ["exam", "examination", "mid-term", "end sem", "test", "viva", "hall ticket", "supplementary", "blueprint", "marks"],
    "Assignment": ["assignment", "homework", "submit", "submission", "lab", "project report", "tutorial", "code files", "case study"],
    "Attendance": ["attendance", "present", "absent", "roll call", "biometric", "75 percent", "deficiency", "shortage", "condonation"],
    "Internship": ["internship", "intern", "hiring", "stipend", "career", "interview", "trainee", "assistantship", "research intern"],
    "Event": ["event", "festival", "hackathon", "workshop", "webinar", "contest", "gala", "sports meet", "exhibition", "celebration"]
}

def simulate_classification(text: str):
    """Simple rule-based baseline matcher to simulate text classification prototype."""
    text_lower = text.lower()
    scores = {cat: 0 for cat in CATEGORIES}
    
    for cat, keywords in KEYWORD_RULES.items():
        for kw in keywords:
            if kw in text_lower:
                scores[cat] += 1
                
    best_cat = max(scores, key=scores.get)
    if scores[best_cat] == 0:
        return "General / Uncategorized", 0.50
    
    confidence = min(0.70 + (scores[best_cat] * 0.12), 0.98)
    return best_cat, confidence

def main():
    print("=" * 70)
    print(" SWYNEX Technologies Internship - Task 1: AI Problem Design")
    print(" Interactive Student Announcement Classification Demo")
    print("=" * 70)
    
    print("\n[+] Testing pre-defined sample announcements:\n")
    sample_tests = [
        "The DBMS examination will be conducted on Monday morning in Hall B.",
        "Submit your Machine Learning lab assignment before Friday at 11:59 PM.",
        "All students must maintain at least 75 percent attendance for eligibility.",
        "Applications for the Summer AI and Data Science Internship are now open.",
        "Registration for HackFest 2025 national technical hackathon is open!"
    ]
    
    for idx, sample in enumerate(sample_tests, 1):
        cat, conf = simulate_classification(sample)
        print(f"  {idx}. Input : \"{sample}\"")
        print(f"     Output: [{cat.upper()}] (Confidence: {conf*100:.1f}%)\n")
        
    print("=" * 70)
    print(" [OK] Interactive demo loaded successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
