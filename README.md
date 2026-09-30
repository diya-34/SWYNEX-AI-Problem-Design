# SWYNEX AI Problem Design

[![SWYNEX Internship](https://img.shields.io/badge/SWYNEX%20Technologies-Internship%20Task%201-blue.svg)](https://github.com)
[![Task](https://img.shields.io/badge/Task-AI%20Problem%20Design-brightgreen.svg)](https://github.com)
[![Domain](https://img.shields.io/badge/Domain-NLP%20%7C%20Text%20Classification-orange.svg)](https://github.com)

---

## 1. Project Overview

This repository contains the complete design and problem definition documentation for **Task 1: AI Problem Design** of the **SWYNEX Technologies Internship**.

The objective of this project is to systematically formulate and document a practical, narrow Artificial Intelligence problem using **Natural Language Processing (NLP)** and **Text Classification**: **"Student Announcement Classification using AI"**.

Rather than jumping directly to training an overly complex machine learning pipeline, this task focuses on foundational AI problem design—identifying user needs, defining inputs and outputs, establishing realistic constraints, planning evaluation methodologies, and designing ethical safeguards.

---

## 2. Problem Statement

In academic institutions and universities, college students and faculty receive an overwhelming volume of daily announcements across various communication channels (WhatsApp groups, Slack/Discord channels, emails, bulletin boards, and Learning Management Systems).

These notices cover a wide variety of topics, including examinations, assignments, attendance policies, internship opportunities, and extracurricular events. Because these messages often arrive in unstructured, mixed streams, students frequently face the following challenges:
- Difficulty identifying urgent vs. non-urgent notices quickly.
- Risk of missing critical deadlines (exam forms, assignment submissions, internship applications).
- Information overload and clutter in daily academic communication.

The proposed AI system aims to solve this problem by automatically classifying college announcement text into one of five predefined, actionable categories.

---

## 3. Target Users

| User Category | Description | Primary Benefit |
| :--- | :--- | :--- |
| **Primary Users: College Students** | Undergraduate & postgraduate students across departments | Quickly filter, prioritize, and receive alerts for relevant academic and career notices. |
| **Secondary Users: Faculty Members** | Professors, teaching assistants, and subject coordinators | Ensure departmental circulars reach the right category channels automatically. |
| **Secondary Users: Academic Administration** | Dean's office, exam cell, placement & training cell | Streamline broadcast messaging and minimize missed notices. |

---

## 4. AI Use Case

- **Task Type:** Multi-Class Natural Language Processing (NLP) / Text Classification.
- **Input:** Raw text of a college announcement or circular (English).
- **Output:** Exactly one predicted category label among 5 distinct classes:
  1. **Exam**
  2. **Assignment**
  3. **Attendance**
  4. **Internship**
  5. **Event**

---

## 5. Example

### Primary Example
- **Input:** `"Submit your Machine Learning assignment by Friday."`
- **Output:** `Assignment`

### Additional Illustrative Examples

| ID | Announcement Text (Input) | Expected Label (Output) |
| :---: | :--- | :---: |
| 1 | "The End Semester Examination schedule for Computer Engineering has been published on the student portal." | **Exam** |
| 2 | "Submit the literature survey report for your major project to your assigned project guide by Thursday." | **Assignment** |
| 3 | "All students must maintain at least 75 percent attendance to be eligible for the semester examinations." | **Attendance** |
| 4 | "Applications for the Summer AI and Data Science Internship at TCS are now open for final year students." | **Internship** |
| 5 | "Registration for HackFest 2025 annual national technical hackathon is officially open." | **Event** |

---

## 6. Data Source

- **Initial Dataset:** A curated synthetic benchmark dataset ([announcements.csv](file:///data/announcements.csv)) created specifically for this AI problem design and specification task.
- **Content:** 60 balanced, realistic student announcement texts paired with their corresponding ground-truth category labels.
- **Future Expansion:** Future iterations may incorporate open-access academic datasets or anonymized departmental notice archives after thorough review of privacy terms, data licenses, and institutional ethics approvals.

---

## 7. Dataset Structure

The dataset is formatted as a standardized CSV file located in [`data/announcements.csv`](file:///data/announcements.csv):

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `id` | Integer | Unique identifier for each sample record |
| `text` | String | Raw text content of the academic announcement |
| `label` | String | Categorical label (`Exam`, `Assignment`, `Attendance`, `Internship`, `Event`) |

### Category Distribution

- **Total Records:** 60
- **Exam:** 12 records (20%)
- **Assignment:** 12 records (20%)
- **Attendance:** 12 records (20%)
- **Internship:** 12 records (20%)
- **Event:** 12 records (20%)

---

## 8. Constraints

1. **Language Scope:** The initial problem definition strictly covers announcements written in the English language.
2. **Fixed Taxonomy:** The classification schema is restricted to five discrete categories (`Exam`, `Assignment`, `Attendance`, `Internship`, `Event`).
3. **Short & Ambiguous Text:** Extremely brief or vague messages (e.g., *"Meeting today at 4 PM"*) may lead to low model confidence or misclassification.
4. **Multi-Topic Announcements:** Circulars combining multiple topics (e.g., *"Exam will be held on Monday and attendance is compulsory"*) present single-label ambiguity.
5. **Initial Dataset Size:** The sample dataset comprises 60 synthetic samples for problem definition, and model training will require dataset expansion.
6. **Zero Sensitive / PII Data:** No Personally Identifiable Information (student names, roll numbers, phone numbers, email addresses) is stored or processed.
7. **Model Dependence on Data Quality:** Generalization performance in future stages will be heavily reliant on data variety and annotation consistency.

---

## 9. Proposed AI Approach

A future implementation will follow a standard modular Machine Learning NLP pipeline:

```
+------------------------------------+
|        College Announcement        |
+------------------------------------+
                  │
                  ▼
+------------------------------------+
|         Text Preprocessing         |
| (Lowercasing, Tokenization,        |
|  Stopword Removal, Lemmatization)  |
+------------------------------------+
                  │
                  ▼
+------------------------------------+
|    Feature Representation / NLP    |
|   (TF-IDF / N-grams / Embeddings)  |
+------------------------------------+
                  │
                  ▼
+------------------------------------+
|        Classification Model        |
|  (Logistic Regression / NB / BERT) |
+------------------------------------+
                  │
                  ▼
+------------------------------------+
|         Predicted Category         |
|  [Exam|Assignment|Attendance|...]  |
+------------------------------------+
```

### Potential Candidate Models (Future Implementation)
- **Baseline Models:** Term Frequency-Inverse Document Frequency (TF-IDF) representation paired with:
  - Multinomial Naive Bayes
  - Logistic Regression
  - Linear Support Vector Machines (Linear SVM)
- **Advanced / Neural Models:**
  - Fine-tuned lightweight Transformer models (e.g., `distilbert-base-uncased` or `BERT-mini`).

> **Note:** *These models represent proposed future architectures for downstream development. No model has been trained as part of this problem design phase.*

---

## 10. Evaluation Approach

### Proposed Data Split
- **Training Set:** 80% of data (for feature extraction and model parameter estimation)
- **Testing Set:** 20% of held-out unseen data (for evaluation)
- **Validation Scheme:** 5-Fold Stratified Cross-Validation during hyperparameter tuning.

### Proposed Evaluation Metrics
- **Overall Accuracy:** Percentage of total correctly classified announcements.
- **Precision (Macro / Per-Class):** Ratio of correct positive predictions for each category.
- **Recall (Macro / Per-Class):** Ability of the model to capture all announcements belonging to a category.
- **Macro F1-Score:** Harmonic mean of precision and recall averaged evenly across all 5 classes.
- **Confusion Matrix:** Error-analysis grid to identify which classes are most often confused (e.g., Exam vs. Assignment).

> **Disclaimer:** *The metrics described above constitute the proposed verification protocol for future development. No performance numbers are fabricated or claimed.*

---

## 11. Success Criteria

To determine whether a trained model meets production-readiness in subsequent tasks, the following **target criteria** are defined:

- **Target Overall Accuracy:** $\ge 85\%$ on the unseen test split.
- **Target Macro F1-Score:** $\ge 0.80$ across all 5 target categories.
- **Per-Category Recall Balance:** Minimum recall $\ge 0.75$ for every individual category (avoiding class starvation).
- **Generalization:** Consistent performance on newly collected real-world student notices without severe overfitting.

> *These metrics are target engineering benchmarks, not measured experimental outcomes.*

---

## 12. Expected Output

| Announcement Input | Expected Model Prediction |
| :--- | :--- |
| `"DBMS exam starts Monday"` | `Exam` |
| `"Submit your Python assignment tomorrow"` | `Assignment` |
| `"Minimum attendance is required"` | `Attendance` |
| `"Apply for the AI internship"` | `Internship` |
| `"Technical festival registration is open"` | `Event` |

---

## 13. Privacy and Ethical Considerations

1. **No Personal Identifiable Information (PII):** The dataset excludes all student names, enrollment numbers, phone numbers, grades, or personal email addresses.
2. **Anonymization:** Academic notices must be scrubbed of individual identifiers prior to batch training or logging.
3. **Error Transparency:** System outputs must display classification confidence levels where possible, indicating that predictions may occasionally be incorrect.
4. **Low-Stakes Advisory Scope:** The classification system serves as an informational productivity and sorting aid; it must never be used for disciplinary, grading, or high-stakes administrative decisions.

---

## 14. Future Improvements

- **Dataset Scale Expansion:** Collect and annotate a larger corpus of 1,000+ multi-institutional announcements.
- **Multilingual Support:** Add support for regional languages including **Gujarati** and Hindi to cater to regional institutions.
- **Advanced NLP Pipelines:** Explore state-of-the-art sentence transformer embeddings and contextual LLM zero-shot classifiers.
- **Confidence Scoring & Fallback:** Flag low-confidence predictions ($< 0.60$) for human verification or a "General / Uncategorized" fallback bucket.
- **Real-Time Integration & Web Interface:** Build a lightweight Streamlit/FastAPI web interface and mobile push notification system for instant categorization.

---

## 15. Web Application & Quick Start

A modern, interactive **Streamlit web application** is included to demonstrate the problem design, category cards, sample explorer, and prototype classification interface.

### Running the Web Application Locally

```powershell
# 1. Install dependencies
python -m pip install -r requirements.txt

# 2. Run the Streamlit web application
streamlit run app.py
```

The application will launch in your browser (typically at `http://localhost:8501`).

### Running the Terminal Demo & Visual Generator
```powershell
# Run the interactive CLI classification demo
python scripts/classify_demo.py

# Run the dataset analysis & visual charts generator
python scripts/generate_visuals.py
```

---

## 16. Conclusion

This project successfully fulfills **Task 1: AI Problem Design** under the **SWYNEX Technologies Internship**. By establishing a well-bounded problem statement, a balanced dataset blueprint, an interactive prototype web interface, a transparent NLP pipeline, realistic constraints, and stringent privacy guidelines, it lays a solid engineering foundation for the subsequent model development lifecycle.
