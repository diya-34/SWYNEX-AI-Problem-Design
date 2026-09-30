# SWYNEX AI Problem Design & Machine Learning Prototype

[![SWYNEX Internship](https://img.shields.io/badge/SWYNEX%20Technologies-Internship%20Task%201-blue.svg)](https://github.com)
[![Task](https://img.shields.io/badge/Task-AI%20Problem%20Design%20%26%20ML%20Prototype-brightgreen.svg)](https://github.com)
[![Model](https://img.shields.io/badge/Model-TF--IDF%20%2B%20Logistic%20Regression-orange.svg)](https://github.com)
[![Evaluated Accuracy](https://img.shields.io/badge/Test%20Accuracy-90.00%25-success.svg)](https://github.com)

---

## 1. Project Overview

This repository contains the complete problem design specification, balanced dataset, machine learning pipeline, empirical evaluation, and interactive Streamlit web application for **Task 1: AI Problem Design** under the **SWYNEX Technologies Internship**.

The objective of this project is to systematically formulate, build, and evaluate a practical Natural Language Processing (NLP) text classification solution: **"Student Announcement Classification using AI"**.

The system automatically classifies incoming college announcements into five actionable categories:
1. **Exam**
2. **Assignment**
3. **Attendance**
4. **Internship**
5. **Event**

---

## 2. Problem Statement

College students and faculty receive hundreds of notices each semester across disparate channels (WhatsApp groups, Slack/Discord, emails, bulletin boards, and Learning Management Systems).

Because these circulars arrive in unstructured, mixed streams:
- Students frequently miss critical academic deadlines (exam registration forms, laboratory submissions, fee payments).
- Valuable career and internship opportunities get buried under routine administrative announcements.
- Information overload causes high cognitive friction in everyday campus life.

This project delivers an automated NLP pipeline that accurately classifies announcement text into five standardized categories.

---

## 3. Target Users

| User Category | Description | Primary Value Delivered |
| :--- | :--- | :--- |
| **Primary Users: College Students** | Undergraduate & postgraduate students | Triage incoming notices, filter by category, and avoid missing vital deadlines. |
| **Secondary Users: Faculty Members** | Professors and teaching assistants | Automatically route department announcements to targeted category feeds. |
| **Secondary Users: Academic Administration** | Dean's office, exam cell, placement cell | Standardize broadcast messaging across all departments. |

---

## 4. AI Use Case

- **Task Type:** Multi-Class Natural Language Processing (NLP) / Supervised Text Classification.
- **Input:** Single-message college announcement or notice in text format (English).
- **Output:** Exactly one predicted category label among 5 distinct classes (`Exam`, `Assignment`, `Attendance`, `Internship`, `Event`) accompanied by the model's posterior prediction probability.

---

## 5. Illustrative Examples

| ID | Announcement Text (Input) | Ground Truth Label | Model Output |
| :---: | :--- | :---: | :---: |
| 1 | "The End Semester Examination schedule for Computer Engineering has been published on the student portal." | **Exam** | `Exam` |
| 2 | "Submit your Machine Learning lab assignment before Friday at 11:59 PM." | **Assignment** | `Assignment` |
| 3 | "All students must maintain at least 75 percent attendance to be eligible for the semester examinations." | **Attendance** | `Attendance` |
| 4 | "Applications for the Summer AI and Data Science Internship at TCS are now open for final year students." | **Internship** | `Internship` |
| 5 | "Registration for HackFest 2025 annual national technical hackathon is officially open." | **Event** | `Event` |

---

## 6. Dataset Specification

- **File Path:** [`data/announcements.csv`](file:///data/announcements.csv)
- **Total Records:** **200 realistic sample announcements**
- **Balance:** Exactly **40 samples per category** across all 5 classes
- **Dataset Columns:**
  - `id`: Integer unique identifier (1 to 200)
  - `text`: String text of the academic notice
  - `label`: Categorical target (`Exam`, `Assignment`, `Attendance`, `Internship`, `Event`)
- **Sanitization & Ethics:** The dataset is synthetic/manually curated for this prototype and contains **100% anonymized data** with zero Personally Identifiable Information (no student names, IDs, phone numbers, or grades).

---

## 7. Machine Learning Pipeline Architecture

```
+-------------------------------------------------------+
|                 Student Announcement                  |
|     "Submit your Machine Learning assignment by Friday"     |
+-------------------------------------------------------+
                           │
                           ▼
+-------------------------------------------------------+
|                  TF-IDF Vectorizer                    |
|        (Unigrams + Bigrams, Sublinear Term Freq)      |
+-------------------------------------------------------+
                           │
                           ▼
+-------------------------------------------------------+
|             Logistic Regression Classifier            |
|              (Multinomial Softmax Probabilities)      |
+-------------------------------------------------------+
                           │
                           ▼
+-------------------------------------------------------+
|              Predicted Class & Confidence             |
|                 [ ASSIGNMENT : 98.4% ]                |
+-------------------------------------------------------+
```

### Model Components
1. **Feature Extraction (`model/vectorizer.pkl`):** Scikit-Learn `TfidfVectorizer` configured with unigrams and bigrams (`ngram_range=(1, 2)`), English stop-word filtering, and sublinear term frequency scaling (`sublinear_tf=True`), yielding 1,970 vocabulary features.
2. **Classifier (`model/classifier.pkl`):** Scikit-Learn `LogisticRegression` (`C=1.0`, `max_iter=1000`, `random_state=42`) with L2 regularization and multinomial softmax output.

---

## 8. Empirical Model Evaluation (Actual Results)

The model was evaluated on an isolated **20% held-out test split (40 unseen samples)** using stratified partitioning:

### Target Criteria vs. Actual Measured Results

| Evaluation Metric | Target Success Criteria | Actual Measured Result | Status |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | $\ge 85.00\%$ | **90.00%** | **PASSED (MET)** |
| **Macro F1-Score** | $\ge 0.8000$ | **0.8999** | **PASSED (MET)** |
| **Macro Precision** | - | **90.28%** | **High** |
| **Macro Recall** | $\ge 75.00\%$ | **90.00%** | **PASSED (MET)** |
| **Weighted F1-Score**| - | **0.8999** | **High** |

### Per-Class Performance Breakdown

| Category | Precision | Recall | F1-Score | Support (Test Split) |
| :--- | :---: | :---: | :---: | :---: |
| **Assignment** | 0.89 | 1.00 | **0.94** | 8 |
| **Attendance** | 0.88 | 0.88 | **0.88** | 8 |
| **Event** | 1.00 | 0.88 | **0.93** | 8 |
| **Exam** | 0.75 | 0.75 | **0.75** | 8 |
| **Internship** | 1.00 | 1.00 | **1.00** | 8 |
| **Overall Average** | **0.90** | **0.90** | **0.90** | **40** |

### Confusion Matrix

The empirical confusion matrix generated by [`evaluation/evaluate_model.py`](file:///evaluation/evaluate_model.py) is saved at [`evaluation/confusion_matrix.png`](file:///evaluation/confusion_matrix.png):

![Confusion Matrix](screenshots/confusion_matrix.png)

---

## 9. Quick Start & Execution Guide

Follow these exact commands in your terminal:

### 1. Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

### 2. Train the Real ML Model
```powershell
python model/train_model.py
```

### 3. Evaluate the Model & Generate Metrics
```powershell
python evaluation/evaluate_model.py
```

### 4. Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
*(Or `python -m streamlit run app.py`)*

The web application will open automatically at:  
👉 **`http://localhost:8501`**

---

## 10. Streamlit Web Dashboard Features

- **🏠 Home:** Project overview, supported category cards, and architecture diagrams.
- **🔮 Classifier:** Real-time ML inference with posterior class probability, "Why this prediction?" feature attribution, quick sample buttons, and session prediction history.
- **📊 Dataset Explorer:** Searchable, category-filtered table of the 200 announcements with interactive distribution charts.
- **📈 Model Performance:** Live display of authentic test metrics, confusion matrix heatmap, and per-class classification reports.
- **ℹ️ About:** Complete AI problem design specifications, constraints, evaluation plan, and ethical guidelines.

---

## 11. Project Repository Structure

```
SWYNEX-AI-Problem-Design/
│
├── app.py                         # Multi-page Streamlit AI Dashboard Web App
├── requirements.txt               # Dependencies (streamlit, pandas, scikit-learn, etc.)
├── README.md                      # Comprehensive project documentation
├── .gitignore                     # Git ignore rules
│
├── data/
│   └── announcements.csv          # 200 balanced sample announcements (40 per class)
│
├── model/
│   ├── train_model.py             # Script to train TF-IDF + Logistic Regression
│   ├── vectorizer.pkl             # Trained TF-IDF vectorizer artifact
│   ├── classifier.pkl             # Trained Logistic Regression classifier artifact
│   ├── train_data.csv             # 80% Stratified Training split (160 samples)
│   └── test_data.csv              # 20% Held-Out Testing split (40 samples)
│
├── evaluation/
│   ├── evaluate_model.py          # Script to compute actual test metrics
│   ├── metrics.json               # Computed metrics, classification report & confusion matrix
│   └── confusion_matrix.png       # High-resolution Seaborn confusion matrix heatmap
│
├── scripts/
│   ├── classify_demo.py           # CLI classification demo script
│   └── generate_visuals.py        # Visual charts generator
│
├── docs/
│   └── AI_Problem_Design.md       # Formal problem design specification document
│
└── screenshots/
    ├── README.md                  # Visual assets documentation
    ├── confusion_matrix.png       # Mirrored confusion matrix plot
    ├── dataset_distribution.png   # Class distribution bar chart
    └── system_workflow.png        # System pipeline diagram
```

---

## 12. Privacy, Safety & Ethical Guidelines

1. **Zero PII Policy:** No student names, roll numbers, contact information, or sensitive academic records are gathered or processed.
2. **Prediction Transparency:** Probability scores are displayed alongside each prediction so users can gauge confidence levels.
3. **Low-Stakes Scope:** The application serves purely as an informational productivity and sorting tool; it is not permitted for disciplinary, grading, or administrative decision-making.

---

## 13. Limitations & Future Roadmap

- **Multi-Topic Ambiguity:** Announcements that span multiple categories (e.g. *"Exam will be held on Monday and attendance is compulsory"*) are currently forced into a single dominant label. Future versions will support multi-label ranking.
- **Multilingual Support:** Future roadmap includes extending NLP models to Indian regional languages, including **Gujarati** and Hindi.
- **Transformer Embeddings:** Upgrading from TF-IDF to fine-tuned sentence transformers (`all-MiniLM-L6-v2`) for enhanced semantic context.
- **Human-in-the-Loop Review:** Flagging predictions with confidence $< 0.50$ for manual verification.

---

## 14. Conclusion

This project completes the **Task 1: AI Problem Design** milestone for the **SWYNEX Technologies Internship**. By combining formal problem design, a balanced 200-record dataset, a trained and evaluated TF-IDF + Logistic Regression model (**90.00% test accuracy**), and a full-featured Streamlit web application, it establishes a reliable, production-ready foundation for future AI deployments.
