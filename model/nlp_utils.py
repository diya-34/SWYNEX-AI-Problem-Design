"""
SWYNEX Technologies Internship - Task 1: AI Problem Design
Module: model/nlp_utils.py
Description: Natural Language Processing utility functions for Information Extraction 
             (Date, Time, Location, Subject, Deadline), Priority Scoring, and Extractive Summarization.
"""

import re
from typing import Dict, Any

# Academic subjects database for entity matching
KNOWN_SUBJECTS = [
    "Database Management Systems", "DBMS",
    "Machine Learning", "ML",
    "Data Structures and Algorithms", "DSA", "Data Structures",
    "Artificial Intelligence", "AI",
    "Operating Systems", "OS",
    "Computer Networks", "CN",
    "Software Engineering",
    "Web Development", "Frontend Development", "Full Stack MERN",
    "Digital Signal Processing", "DSP",
    "Theory of Computation", "TOC",
    "Cloud Computing", "AWS", "Azure",
    "Compiler Design",
    "Applied Statistics", "Mathematics II", "Applied Mathematics", "Mathematics",
    "Physics", "Chemistry", "Microprocessors", "Microcontroller",
    "Natural Language Processing", "NLP",
    "Human-Computer Interaction", "HCI",
    "Circuit Theory", "Control Systems", "Digital Logic Design"
]

def extract_information(text: str) -> Dict[str, str]:
    """
    Extracts structured academic metadata (Date, Time, Location, Subject, Deadline) 
    using regex and rule-based NLP heuristics.
    Returns 'Not detected' if no confident entity match is found.
    """
    info = {
        "Date": "Not detected",
        "Time": "Not detected",
        "Location": "Not detected",
        "Subject": "Not detected",
        "Deadline": "Not detected"
    }

    # 1. Date Extraction
    # Match days of week, relative days, and date formats like "October 15th", "5 October", "March 15th", "December 1st"
    date_patterns = [
        r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
        r"\b(?:today|tomorrow|tonight|yesterday|this weekend|next week)\b",
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*-\s*\d{1,2}(?:st|nd|rd|th)?)?\b",
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b"
    ]
    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            info["Date"] = match.group(0).strip()
            break

    # 2. Time Extraction
    # Match "10 AM", "11:59 PM", "9:00 AM", "5:00 PM", "2 PM to 5 PM", "midnight", "noon"
    time_match = re.search(r"\b(?:\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm)(?:\s*(?:to|-)\s*\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm))?|midnight|noon)\b", text)
    if time_match:
        info["Time"] = time_match.group(0).strip()

    # 3. Location / Venue Extraction
    # Match "Hall A", "Room 302", "DSP Lab 2", "auditorium", "Computer Lab 4", "campus health center", "campus ground", "seminar hall"
    loc_match = re.search(
        r"\b(?:(?:Hall|Room|Lab|Computer Lab|DSP Lab|Seminar Hall)\s+[A-Za-z0-9]+|(?:main\s+)?auditorium|campus\s+(?:ground|health center|garden)|student\s+portal|Google\s+Classroom|Moodle|LMS)\b",
        text,
        re.IGNORECASE
    )
    if loc_match:
        info["Location"] = loc_match.group(0).strip()

    # 4. Subject / Course Extraction
    for subj in KNOWN_SUBJECTS:
        pattern = r"\b" + re.escape(subj) + r"\b"
        if re.search(pattern, text, re.IGNORECASE):
            info["Subject"] = subj
            break

    # 5. Deadline Extraction
    # Look for deadline cues e.g. "by Friday at 11:59 PM", "before Thursday", "due on Monday", "closes tomorrow at 5:00 PM"
    deadline_match = re.search(
        r"\b(?:before|by|due\s+on|closes\s+at|deadline\s+is)\s+([^.,;\n]+(?:\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm)|midnight|noon|Friday|Monday|Tuesday|Wednesday|Thursday|Saturday|Sunday|tomorrow|today|\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+))",
        text,
        re.IGNORECASE
    )
    if deadline_match:
        info["Deadline"] = deadline_match.group(0).strip()
    elif "deadline" in text.lower() or "last date" in text.lower():
        if info["Date"] != "Not detected":
            info["Deadline"] = info["Date"]

    return info

def calculate_priority(text: str, category: str) -> Dict[str, Any]:
    """
    Computes an explainable Priority Level (HIGH, MEDIUM, LOW) based on urgency keywords,
    approaching deadlines, and category risk level.
    """
    text_lower = text.lower()
    
    high_urgency_keywords = [
        "today", "tomorrow", "urgent", "immediate", "immediately", "deadline", 
        "last date", "end semester", "final examination", "supplementary", 
        "defaulter", "deficiency", "shortage", "detained", "mandatory", "zero marks", "strictly forbidden"
    ]
    
    medium_urgency_keywords = [
        "submit", "submission", "assignment", "registration", "attendance", 
        "internship", "hiring", "interview", "test", "remedial", "viva-voce", "hall ticket"
    ]

    # Check High Urgency
    matched_high = [kw for kw in high_urgency_keywords if kw in text_lower]
    if matched_high:
        return {
            "level": "HIGH",
            "color": "#dc2626",
            "badge_bg": "#fee2e2",
            "reason": f"High urgency trigger detected: '{matched_high[0]}'"
        }

    # Check Category Specific High Defaults
    if category in ["Exam"]:
        return {
            "level": "HIGH",
            "color": "#dc2626",
            "badge_bg": "#fee2e2",
            "reason": "Exams are classified as critical academic priorities."
        }

    # Check Medium Urgency
    matched_med = [kw for kw in medium_urgency_keywords if kw in text_lower]
    if matched_med or category in ["Assignment", "Attendance", "Internship"]:
        reason_kw = matched_med[0] if matched_med else category.lower()
        return {
            "level": "MEDIUM",
            "color": "#d97706",
            "badge_bg": "#fef3c7",
            "reason": f"Actionable requirement: '{reason_kw}'"
        }

    # Default to Low Urgency (Informational / Event)
    return {
        "level": "LOW",
        "color": "#2563eb",
        "badge_bg": "#dbeafe",
        "reason": "Informational notice or general campus event."
    }

def generate_extractive_summary(text: str, category: str, info: Dict[str, str]) -> str:
    """
    Generates a concise, structured 1-sentence extractive summary of the announcement.
    Clearly labeled as an extractive rule-based baseline.
    """
    clean_text = text.strip()
    if len(clean_text) <= 90:
        return clean_text

    # Extract key components if present
    subj = info.get("Subject", "")
    deadline = info.get("Deadline", "")
    date_val = info.get("Date", "")
    time_val = info.get("Time", "")
    loc = info.get("Location", "")

    prefix = f"[{category.upper()} NOTICE]"
    details = []
    if subj != "Not detected":
        details.append(f"Subject: {subj}")
    if deadline != "Not detected":
        details.append(f"Due: {deadline}")
    elif date_val != "Not detected":
        details.append(f"Date: {date_val}")
    if time_val != "Not detected":
        details.append(f"Time: {time_val}")
    if loc != "Not detected":
        details.append(f"Venue: {loc}")

    if details:
        return f"{prefix} {clean_text.split('.')[0].strip()}. ({', '.join(details)})"
    return f"{prefix} {clean_text.split('.')[0].strip()}."
