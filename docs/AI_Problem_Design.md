# Formal AI Problem Design Specification

**Document Title:** System Specification for Student Announcement Classification  
**Organization / Program:** SWYNEX Technologies Internship  
**Task Module:** Task 1 – AI Problem Design  
**Domain:** Natural Language Processing (NLP) / Text Classification  
**Status:** Proposal / Specification Phase  

---

## 1. Executive Summary

This document outlines the end-to-end problem formulation, technical boundaries, functional requirements, evaluation criteria, and ethical safeguards for an AI-driven academic text classification system. The proposed system, titled **"Student Announcement Classification using AI"**, converts unstructured college announcements into five actionable categories (`Exam`, `Assignment`, `Attendance`, `Internship`, `Event`).

---

## 2. User Analysis

### 2.1 Primary Users
- **College Students:** Need to filter high volumes of academic communication, reduce cognitive overload, prioritize urgent academic deadlines, and quickly locate career opportunities.

### 2.2 Secondary Users
- **Course Instructors & Teaching Staff:** Need assurance that crucial deadlines (e.g., assignment guidelines, test timetables) are properly tagged and delivered.
- **Academic Administration & Department Coordinators:** Need standardized categorizations across campus-wide circulars and departmental communication streams.

---

## 3. Problem Definition

Academic departments frequently transmit essential notices across disparate channels including WhatsApp, Telegram, email distribution lists, and web-based portals. Because circulars are posted in varied formats without standardized subject lines or tags:
1. Students frequently miss critical deadlines for fee payments, exam forms, and assignment submissions.
2. Important career/internship opportunities get buried under routine administrative announcements.
3. Manual tagging and routing by administrators is time-consuming and prone to human error.

The core problem is formulated as:  
*How can an automated, lightweight Natural Language Processing system accurately and reliably classify single-message college announcements into discrete, actionable categories without requiring personally identifiable student information?*

---

## 4. AI Task Formulation

- **Learning Paradigm:** Supervised Multi-Class Text Classification.
- **Input Modality:** Unstructured natural language text strings (English).
- **Target Space:** Discrete set $Y \in \{\text{Exam}, \text{Assignment}, \text{Attendance}, \text{Internship}, \text{Event}\}$.
- **Classification Nature:** Single-label assignment where each input announcement text $x_i$ maps to exactly one primary class label $y_i$.

---

## 5. System Workflow Diagram

```
+-------------------------------------------------------------+
|                      1. Data Ingestion                      |
|       Raw College Announcement Text (e.g., Circular)        |
+-------------------------------------------------------------+
                               │
                               ▼
+-------------------------------------------------------------+
|                    2. Text Preprocessing                    |
|   - Case Normalization (Lowercasing)                        |
|   - Punctuation & Special Character Filtering               |
|   - Stopword Removal & Word Tokenization                    |
|   - Morphological Lemmatization / Stemming                  |
+-------------------------------------------------------------+
                               │
                               ▼
+-------------------------------------------------------------+
|               3. Feature Extraction / Representation        |
|   - TF-IDF Vectorizer (Unigram + Bigram N-grams)            |
|   - Pre-trained Contextual Embeddings (Transformer)         |
+-------------------------------------------------------------+
                               │
                               ▼
+-------------------------------------------------------------+
|                4. Machine Learning Inference                |
|   - Multi-Class Classifier (Logistic Regression / NB / SVM) |
|   - Softmax Probability Distribution over 5 Classes         |
+-------------------------------------------------------------+
                               │
                               ▼
+-------------------------------------------------------------+
|                 5. Output & Post-Processing                 |
|   - Primary Label: [Exam | Assignment | Attendance | ...]   |
|   - Confidence Score (Probability Metric)                   |
|   - Fallback Flag if Confidence < Threshold                 |
+-------------------------------------------------------------+
```

---

## 6. Input and Output Specifications

### 6.1 Input Specification
- **Type:** String (UTF-8 encoded text).
- **Length Constraint:** Typically between 5 and 100 words (single sentence to short paragraph).
- **Language:** English.
- **Example Input:** `"All students must submit their Cloud Computing assignment on GitHub before Sunday."`

### 6.2 Output Specification
- **Type:** Categorical string label + Confidence probability float.
- **Value Domain:** `{"Exam", "Assignment", "Attendance", "Internship", "Event"}`.
- **Example Output:** 
  ```json
  {
    "category": "Assignment",
    "confidence": 0.94
  }
  ```

---

## 7. Data Source & Engineering Plan

### 7.1 Initial Benchmark Dataset
- An initial balanced synthetic dataset of 60 representative academic announcements has been prepared in `data/announcements.csv`.
- Each record contains `id`, `text`, and `label`.
- Classes are evenly distributed (12 records per category) to prevent majority-class bias during preliminary design.

### 7.2 Scaled Data Acquisition Plan (Future Stages)
- Collect publicly accessible historical departmental circulars and student council notices.
- Implement strict PII sanitization to strip names, student IDs, phone numbers, and physical room numbers.
- Perform dual-annotator verification on ambiguous circulars to compute inter-annotator agreement (Cohen's Kappa $\kappa \ge 0.85$).

---

## 8. Technical Constraints & Boundary Conditions

1. **Monolingual Limitation:** Restricted to English text in current design.
2. **Fixed Label Granularity:** The taxonomy does not currently accommodate sub-categories (e.g., "Mid-Term Exam" vs. "Final Exam").
3. **Compound Announcements:** Announcements covering multiple concurrent subjects (e.g., a notice mentioning both exam dates and mandatory attendance) require single dominant-intent resolution.
4. **Computational Footprint:** The classification inference should be lightweight enough to run on standard serverless architectures or client-side web applications.

---

## 9. Evaluation Approach

### 9.1 Data Partitioning Strategy
- **Training Set (80%):** Model parameter training and vocabulary formulation.
- **Testing Set (20%):** Completely isolated hold-out split for objective generalization benchmarking.
- **Cross-Validation:** 5-Fold Stratified Cross-Validation on the training pool.

### 9.2 Formal Metrics
Let $TP_c$, $FP_c$, and $FN_c$ represent True Positives, False Positives, and False Negatives for class $c \in C$:

$$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}$$

$$\text{Macro } F1 = \frac{1}{|C|} \sum_{c \in C} \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

- **Confusion Matrix Analysis:** Systematic inspection of cross-class leakage (specifically between `Exam` and `Assignment`).

---

## 10. Target Success Criteria

| Evaluation Dimension | Benchmark Metric | Target Threshold | Rationale |
| :--- | :--- | :--- | :--- |
| **Overall Accuracy** | Test Accuracy | $\ge 85\%$ | Ensures reliable day-to-day usability for students. |
| **Macro F1-Score** | Macro-Averaged F1 | $\ge 0.80$ | Guards against class imbalance and poor performance on low-frequency classes. |
| **Minimum Class Recall** | Per-Class Recall | $\ge 0.75$ | Critical categories like `Exam` and `Attendance` must not be routinely dropped. |
| **Inference Latency** | Response Time | $< 100\text{ ms}$ | Enables instantaneous categorizations on incoming notification streams. |

> *Note: These figures represent targeted design benchmarks for future model implementation and do not represent fabricated experimental results.*

---

## 11. Risk Analysis & Mitigation

| Risk ID | Identified Risk | Impact | Mitigation Strategy |
| :---: | :--- | :--- | :--- |
| **R1** | Classification error on high-priority notices (e.g., exam timetable mislabeled as event). | High | Integrate confidence thresholding; trigger human verification or alert student if confidence $< 0.65$. |
| **R2** | Ambiguous, multi-topic announcement texts. | Medium | Extract multi-sentence clauses or adopt multi-label ranking in future iterations. |
| **R3** | Non-standard abbreviations and campus slang. | Medium | Build a domain-specific academic token dictionary during text preprocessing. |

---

## 12. Privacy and Ethical Considerations

1. **Zero Personally Identifiable Information (PII):** Student identities, grades, disciplinary records, and financial details are explicitly excluded from the dataset.
2. **Safe Operational Boundary:** The model functions strictly as an informational assistant and productivity filter. It is not empowered to make academic standing, attendance penalty, or administrative decisions.
3. **Data Licensing and Anonymity:** Publicly sourced campus notices must undergo regex-based scrubbing of faculty phone numbers and email addresses.

---

## 13. Future Roadmap & Improvements

- **Phase 2 (Dataset Expansion):** Scale dataset to 1,000+ validated announcements across multiple universities.
- **Phase 3 (Multilingual & Regional NLP):** Expand classification pipelines to support regional Indian languages including **Gujarati** and Hindi.
- **Phase 4 (Contextual Embeddings):** Experiment with fine-tuned Sentence Transformers (`all-MiniLM-L6-v2`) and Small Language Models (SLMs).
- **Phase 5 (User Interface & Deployment):** Deploy as a REST API (FastAPI) paired with an interactive frontend (Streamlit / React) and browser extension.
