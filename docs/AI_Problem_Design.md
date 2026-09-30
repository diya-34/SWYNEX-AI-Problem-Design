# Formal AI Problem Design Specification: Student Announcement Intelligence System

**Organization / Program:** SWYNEX Technologies Internship  
**Task Module:** Task 1 – AI Problem Design & Machine Learning Prototype  
**Domain:** Natural Language Processing (NLP) / Multi-Class Text Classification & Information Extraction  
**Status:** Implemented & Evaluated Prototype  

---

## 1. Executive Summary

This document provides the formal engineering specification for the **Student Announcement Intelligence System**, developed for Task 1 of the **SWYNEX Technologies Internship**. 

The system transforms raw, unstructured college notices into structured, actionable intelligence by:
1. Classifying notices into five discrete academic categories (`Exam`, `Assignment`, `Attendance`, `Internship`, `Event`) using an empirical Machine Learning classifier (**TF-IDF + Logistic Regression**).
2. Calculating an explainable **Priority Level** (`HIGH`, `MEDIUM`, `LOW`).
3. Extracting crucial metadata entities (**Subject/Course**, **Date**, **Time**, **Location/Venue**, **Deadline**).
4. Generating a concise extractive summary for student consumption.

---

## 2. User Analysis & Stakeholder Needs

### 2.1 Primary Users (College Students)
- **Pain Point:** Inundated with notices across disparate communication channels (WhatsApp, Slack, emails, LMS). Important deadlines (exam registrations, assignment submissions) are frequently missed.
- **System Value:** Rapid categorization, deadline extraction, urgency tagging, and searchable archive.

### 2.2 Secondary Users (Faculty Members & TAs)
- **Pain Point:** Difficulty ensuring that circulars reach student subsets reliably.
- **System Value:** Standardized routing and validation of announcement clarity before broadcast.

### 2.3 Secondary Users (Academic Administration & Department Coordinators)
- **Pain Point:** High volume of inquiries regarding missed circulars and exam schedules.
- **System Value:** Automated archival and consistent categorical tagging across departments.

---

## 3. Problem Definition & Scope

Academic notices arrive in unstructured natural language text without standard metadata tags. 

The core AI problem is formally stated as:  
*Given a single-message college announcement $x \in X$, construct a mapping function $f: X \rightarrow Y$ where $Y = \{\text{Exam}, \text{Assignment}, \text{Attendance}, \text{Internship}, \text{Event}\}$, alongside auxiliary entity extraction functions $g(x) \rightarrow \text{Entities}$ and an explainable urgency function $h(x) \rightarrow \{\text{HIGH}, \text{MEDIUM}, \text{LOW}\}$ without requiring student personally identifiable information.*

---

## 4. AI System Architecture & Workflow

```
+-------------------------------------------------------------+
|                      1. Data Ingestion                      |
|           Raw College Announcement Text (English)           |
+-------------------------------------------------------------+
                               │
                               ▼
+-------------------------------------------------------------+
|                    2. Text Preprocessing                    |
|   - Lowercasing, Whitespace Normalization                   |
|   - Stopword Removal & Word Tokenization                    |
+-------------------------------------------------------------+
                               │
                               ▼
+-------------------------------------------------------------+
|               3. Feature Extraction / Representation        |
|   - TF-IDF Vectorizer (Unigram + Bigram N-grams)            |
|   - Sublinear Term Frequency Scaling (sublinear_tf=True)    |
|   - Vocabulary Size: 1,970 features                         |
+-------------------------------------------------------------+
                               │
                               ▼
+-------------------------------------------------------------+
|                4. Machine Learning Inference                |
|   - Multi-Class Logistic Regression Classifier (C=1.0)      |
|   - Softmax Posterior Class Probability Distribution        |
+-------------------------------------------------------------+
                               │
                               ▼
+-------------------------------------------------------------+
|               5. Entity Extraction & Urgency Engine         |
|   - Regex/NLP Entity Matcher: Subject, Date, Time, Venue    |
|   - Deadline Parser & Explainable Priority Scorer           |
|   - Rule-based Extractive Summary Generator                 |
+-------------------------------------------------------------+
                               │
                               ▼
+-------------------------------------------------------------+
|                      6. User Interface                      |
|   - Streamlit AI Dashboard (Classifier, Dataset, Metrics)   |
+-------------------------------------------------------------+
```

---

## 5. Input and Output Specifications

### 5.1 Input Specification
- **Type:** String (UTF-8 encoded text).
- **Length Constraint:** 5 to 150 words.
- **Language Scope:** English (Current production scope).

### 5.2 Output Specification
- **Predicted Category:** Single label from `{"Exam", "Assignment", "Attendance", "Internship", "Event"}`.
- **Prediction Probability:** Float value $[0.0, 1.0]$.
- **Priority Level:** `{"HIGH", "MEDIUM", "LOW"}` with explainable trigger reason.
- **Extracted Entities:**
  - `Subject`: Course/topic name (or *"Not detected"*).
  - `Date`: Specific calendar or relative date (or *"Not detected"*).
  - `Time`: Clock timestamp (or *"Not detected"*).
  - `Location`: Classroom, hall, or portal (or *"Not detected"*).
  - `Deadline`: Submission or registration cutoff (or *"Not detected"*).
- **Short Summary:** Structured 1-sentence synopsis.

---

## 6. Dataset Specification

- **File:** `data/announcements.csv`
- **Volume:** **200 realistic, balanced samples** (exactly 40 per class).
- **Columns:** `id`, `text`, `label`.
- **Privacy & Sanitization:** Synthetic, manually curated dataset containing **zero Personally Identifiable Information (PII)**. All names, student IDs, and private details are completely sanitized.

---

## 7. Empirical Evaluation: Target vs. Actual Results

The model was trained on an **80% stratified training split (160 samples)** and evaluated on a **20% held-out test split (40 unseen samples)** with `random_state=42`.

### 7.1 Target Success Criteria vs. Actual Measured Results

| Metric | Target Success Criteria | Actual Measured Result | Status |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | $\ge 85.00\%$ | **90.00%** | **PASSED (MET)** |
| **Macro F1-Score** | $\ge 0.8000$ | **0.8999** | **PASSED (MET)** |
| **Macro Precision** | - | **90.28%** | **High** |
| **Macro Recall** | $\ge 75.00\%$ | **90.00%** | **PASSED (MET)** |
| **Weighted F1-Score**| - | **0.8999** | **High** |

### 7.2 Detailed Per-Class Classification Report

```text
              precision    recall  f1-score   support

  Assignment       0.89      1.00      0.94         8
  Attendance       0.88      0.88      0.88         8
       Event       1.00      0.88      0.93         8
        Exam       0.75      0.75      0.75         8
  Internship       1.00      1.00      1.00         8

    accuracy                           0.90        40
   macro avg       0.90      0.90      0.90        40
weighted avg       0.90      0.90      0.90        40
```

---

## 8. Explainable Priority & Entity Extraction Logic

### 8.1 Priority Heuristic Logic
- **HIGH:** Triggers when immediate temporal keywords are detected (`"today"`, `"tomorrow"`, `"deadline"`, `"urgent"`, `"last date"`, `"mandatory"`, `"defaulter"`, `"detained"`) or when the category is `Exam`.
- **MEDIUM:** Triggers for actionable deadlines (`"submit"`, `"submission"`, `"registration"`, `"attendance"`, `"interview"`, `"internship"`).
- **LOW:** General informational campus notices and events without immediate action items.

### 8.2 Safe Entity Extraction Rule
- Utilizes deterministic regex patterns over domain vocabulary.
- When an entity cannot be identified with high confidence, the system returns `"Not detected"` rather than generating hallucinations.

---

## 9. Privacy, Ethical & Safety Guardrails

1. **Zero PII Storage:** Student names, contact details, grades, and confidential records are strictly excluded from dataset and inference storage.
2. **Session-Only Memory:** In-browser query history exists solely in memory during the active session and is purged upon refresh or via the "Clear History" button.
3. **Advisory Boundary:** The system is an informational productivity tool and must never be used for disciplinary actions or grade penalties.

---

## 10. Multi-Phase Future Roadmap

- **Phase 2 (Backend & Persistence):** Develop a FastAPI microservice backend with SQLite/PostgreSQL persistence and role-based authentication (Student/Faculty/Admin).
- **Phase 3 (Generative NLP & Multilingual):** Integrate Small Language Models (SLMs) for abstractive summarization and add support for **Gujarati** and Hindi announcements.
- **Phase 4 (RAG System):** Implement Retrieval-Augmented Generation (RAG) over official college syllabus handbooks, exam rulebooks, and academic calendars.
- **Phase 5 (Production Deployment & Alerts):** Deploy on cloud infrastructure (Streamlit Cloud / AWS) with automated WhatsApp/Email notification webhooks.
