"""
SWYNEX Technologies Internship - Task 1: AI Problem Design
Application: Student Announcement Intelligence System
Tech Stack: Python, Streamlit, Pandas, Scikit-Learn, Joblib, Matplotlib, Seaborn
"""

import os
import json
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Import NLP utilities
from model.nlp_utils import extract_information, calculate_priority, generate_extractive_summary

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SWYNEX Technologies – Student Announcement Intelligence System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern AI/ML Dashboard aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #3b82f6 100%);
        padding: 2.2rem 2.4rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.3);
    }
    .main-header h1 {
        color: #ffffff !important;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    .main-header .subtitle {
        color: #93c5fd;
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .main-header .description {
        color: #e2e8f0;
        font-size: 0.92rem;
        margin-bottom: 0;
    }
    
    /* Category Cards */
    .cat-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.1rem;
        height: 100%;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    .cat-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.06);
    }
    .cat-badge {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding: 4px 10px;
        border-radius: 9999px;
        margin-bottom: 0.5rem;
    }
    .badge-exam { background-color: #fee2e2; color: #991b1b; }
    .badge-assignment { background-color: #fef3c7; color: #92400e; }
    .badge-attendance { background-color: #e0e7ff; color: #3730a3; }
    .badge-internship { background-color: #d1fae5; color: #065f46; }
    .badge-event { background-color: #fce7f3; color: #831843; }
    
    /* Output Result Cards */
    .result-box {
        background: #ffffff;
        border: 2px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.5rem;
        margin-top: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
    }
    .result-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }
    .result-category {
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
    }
    .confidence-label {
        font-size: 0.9rem;
        font-weight: 600;
        color: #334155;
        margin-bottom: 0.4rem;
    }
    .disclaimer-note {
        font-size: 0.82rem;
        color: #64748b;
        background-color: #f8fafc;
        padding: 8px 12px;
        border-radius: 8px;
        margin-top: 1rem;
        border-left: 3px solid #3b82f6;
    }
    .explanation-box {
        background: #f0fdf4;
        border-left: 3px solid #22c55e;
        padding: 10px 14px;
        border-radius: 6px;
        margin-top: 0.8rem;
        font-size: 0.85rem;
        color: #166534;
    }
    .info-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 8px;
    }
    .info-label {
        font-size: 0.75rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
    }
    .info-val {
        font-size: 0.95rem;
        font-weight: 600;
        color: #1e293b;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Directory Paths & Model Loader
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = os.path.join(BASE_DIR, "data")
EVAL_DIR = os.path.join(BASE_DIR, "evaluation")

CATEGORIES_META = {
    "Exam": {
        "badge_class": "badge-exam",
        "icon": "📝",
        "color": "#dc2626",
        "desc": "Exam timetables, seating plans, hall tickets, mid-term & end-term tests, re-evaluations, and supplementary examinations."
    },
    "Assignment": {
        "badge_class": "badge-assignment",
        "icon": "📚",
        "color": "#d97706",
        "desc": "Homework submissions, laboratory journals, project milestones, code repository uploads, and grading deadlines."
    },
    "Attendance": {
        "badge_class": "badge-attendance",
        "icon": "⏱️",
        "color": "#4f46e5",
        "desc": "75% minimum eligibility rules, biometric clock-ins, daily roll calls, attendance shortage notices, and medical condonation."
    },
    "Internship": {
        "badge_class": "badge-internship",
        "icon": "💼",
        "color": "#059669",
        "desc": "Summer & winter internships, campus recruitment drives, industry training, research fellowships, and stipend roles."
    },
    "Event": {
        "badge_class": "badge-event",
        "icon": "🎉",
        "color": "#db2777",
        "desc": "College festivals, national hackathons, technical workshops, cultural celebrations, sports meets, and guest lectures."
    }
}

@st.cache_resource
def load_ml_pipeline():
    vectorizer_path = os.path.join(MODEL_DIR, "vectorizer.pkl")
    classifier_path = os.path.join(MODEL_DIR, "classifier.pkl")
    
    if os.path.exists(vectorizer_path) and os.path.exists(classifier_path):
        vectorizer = joblib.load(vectorizer_path)
        classifier = joblib.load(classifier_path)
        return vectorizer, classifier, True
    return None, None, False

def load_evaluation_metrics():
    metrics_path = os.path.join(EVAL_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

vectorizer, classifier, model_loaded = load_ml_pipeline()
eval_metrics = load_evaluation_metrics()

# Session State for History & Input Text
if "history" not in st.session_state:
    st.session_state["history"] = []

if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""

def set_text(text):
    st.session_state["input_text"] = text

# -----------------------------------------------------------------------------
# 3. Sidebar Navigation
# -----------------------------------------------------------------------------
st.sidebar.markdown("""
<div style="text-align: center; padding: 0.8rem 0;">
    <h2 style="margin: 0; color: #1e3a8a; font-weight: 800; font-size: 1.35rem;">SWYNEX AI</h2>
    <p style="margin: 0; color: #64748b; font-size: 0.82rem; font-weight: 600;">Student Announcement Intelligence</p>
</div>
""", unsafe_allow_html=True)

nav_selection = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🔮 AI Classifier", "📊 Dataset Explorer", "📈 Model Performance", "🕒 Prediction History", "ℹ️ About"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ System Status")
if model_loaded:
    st.sidebar.success("✅ Real ML Model Loaded (TF-IDF + Logistic Regression)")
else:
    st.sidebar.warning("⚠️ Model not found. Run `python model/train_model.py`")

if eval_metrics:
    acc_val = eval_metrics["actual_measured_results"]["accuracy"] * 100
    f1_val = eval_metrics["actual_measured_results"]["macro_f1"]
    st.sidebar.metric(label="Evaluated Test Accuracy", value=f"{acc_val:.1f}%")
    st.sidebar.metric(label="Evaluated Macro F1", value=f"{f1_val:.4f}")

st.sidebar.caption("🌐 Language Scope: English (Gujarati/Hinglish in Roadmap)")

# -----------------------------------------------------------------------------
# 4. Page: HOME
# -----------------------------------------------------------------------------
if nav_selection == "🏠 Home":
    st.markdown("""
    <div class="main-header">
        <div class="subtitle">SWYNEX Technologies &bull; Task 1 — AI Problem Design</div>
        <h1>Student Announcement Intelligence System</h1>
        <div class="description">
            An NLP-based AI system that automatically categorizes college announcements and extracts useful information for students.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📌 Supported Announcement Categories")
    cols = st.columns(5)
    for i, (cat_name, meta) in enumerate(CATEGORIES_META.items()):
        with cols[i]:
            st.markdown(f"""
            <div class="cat-card">
                <span class="cat-badge {meta['badge_class']}">{meta['icon']} {cat_name}</span>
                <p style="font-size: 0.83rem; color: #475569; margin: 0; line-height: 1.4;">
                    {meta['desc']}
                </p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.1, 0.9])
    with col1:
        st.markdown("### 🎯 System Objective & Intelligence Features")
        st.markdown("""
        In academic environments, students receive hundreds of unstructured circulars across WhatsApp groups, LMS boards, and emails.
        
        The **Student Announcement Intelligence System** converts noisy messages into structured insights:
        - **1. Category Classification:** Identifies whether a notice is an Exam, Assignment, Attendance rule, Internship opportunity, or Campus Event.
        - **2. Priority Scoring:** Evaluates urgency (High / Medium / Low) based on deadlines and time triggers.
        - **3. Information Extraction:** Extracts Subject, Deadline, Date, Time, and Location automatically.
        - **4. Short Summary:** Produces concise executive summaries for busy students.
        """)
        
        st.info("💡 **Try it live:** Select **🔮 AI Classifier** in the sidebar to test predictions and information extraction!")

    with col2:
        st.markdown("### 🔄 End-to-End System Workflow")
        st.markdown("""
        ```
        +-------------------------------------------------------+
        |                 Student Announcement                  |
        |  "The DBMS exam will be held Monday at 10 AM Hall B"  |
        +-------------------------------------------------------+
                                   │
                                   ▼
        +-------------------------------------------------------+
        |            Text Preprocessing & TF-IDF Vector         |
        +-------------------------------------------------------+
                                   │
                                   ▼
        +-------------------------------------------------------+
        |            Logistic Regression ML Classifier          |
        +-------------------------------------------------------+
                                   │
                                   ▼
        +-------------------------------------------------------+
        |             Predicted Category + Confidence           |
        |                    [ EXAM : 96.2% ]                   |
        +-------------------------------------------------------+
                                   │
                                   ▼
        +-------------------------------------------------------+
        |     Information Extraction (Subject, Date, Venue)     |
        |        Subject: DBMS | Date: Monday | Room: Hall B    |
        +-------------------------------------------------------+
                                   │
                                   ▼
        +-------------------------------------------------------+
        |       Explainable Priority & Extractive Summary       |
        |      Priority: HIGH | Summary: [EXAM NOTICE] ...      |
        +-------------------------------------------------------+
        ```
        """)

# -----------------------------------------------------------------------------
# 5. Page: AI CLASSIFIER
# -----------------------------------------------------------------------------
elif nav_selection == "🔮 AI Classifier":
    st.markdown("""
    <div class="main-header">
        <div class="subtitle">Live AI Inference & Extraction Engine</div>
        <h1>Classify College Announcement</h1>
        <div class="description">
            Enter any announcement or pick a sample notice below to inspect the real-time ML prediction, priority rating, and entity extraction.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Example Announcements
    SAMPLE_EXAMPLES = [
        ("📝 Exam", "The DBMS examination will be conducted on Monday at 10 AM in Hall B."),
        ("📚 Assignment", "Submit your Machine Learning assignment by 5 October at 11:59 PM in Lab 3."),
        ("⏱️ Attendance", "All students must maintain at least 75 percent attendance to be eligible for the semester examinations."),
        ("💼 Internship", "Applications for the Summer AI and Data Science Internship at TCS are now open for final year students."),
        ("🎉 Event", "Registration for HackFest 2025 annual national technical hackathon is officially open in the auditorium.")
    ]

    st.markdown("##### 💡 Try a Sample Announcement")
    ex_cols = st.columns(len(SAMPLE_EXAMPLES))
    for idx, (btn_title, sample_text) in enumerate(SAMPLE_EXAMPLES):
        with ex_cols[idx]:
            if st.button(btn_title, key=f"ex_btn_{idx}", use_container_width=True):
                set_text(sample_text)

    st.markdown("<br>", unsafe_allow_html=True)

    col_input, col_out = st.columns([1.1, 0.9])

    with col_input:
        st.markdown("### ✍️ Announcement Input")
        user_input = st.text_area(
            label="Enter your college announcement...",
            value=st.session_state["input_text"],
            placeholder="Type or paste any student circular, notice, or announcement here (e.g. 'Submit your Python assignment by Friday at 5:00 PM in Lab 2')...",
            height=140,
            label_visibility="collapsed"
        )
        
        classify_btn = st.button("🚀 Classify Announcement", type="primary", use_container_width=True)

    with col_out:
        st.markdown("### 🎯 AI Prediction & Intelligence Output")
        
        if classify_btn:
            clean_input = user_input.strip()
            if not clean_input:
                st.warning("⚠️ Please enter an announcement text before classifying.")
            elif len(clean_input.split()) < 3:
                st.warning("⚠️ Announcement is very short. Please provide a more descriptive notice for reliable classification.")
            elif not model_loaded:
                st.error("❌ Model not loaded! Please run `python model/train_model.py` to train the model first.")
            else:
                # 1. Vectorize and Predict
                input_vec = vectorizer.transform([clean_input])
                pred_label = classifier.predict(input_vec)[0]
                pred_probs = classifier.predict_proba(input_vec)[0]
                
                # Get class index & confidence
                class_idx = list(classifier.classes_).index(pred_label)
                confidence = pred_probs[class_idx]
                meta = CATEGORIES_META.get(pred_label, {})
                
                # Check for low-confidence ambiguity
                is_low_conf = confidence < 0.40
                display_label = pred_label if not is_low_conf else f"{pred_label} (Low Confidence)"
                display_color = meta.get("color", "#64748b") if not is_low_conf else "#64748b"
                display_icon = meta.get("icon", "📄")

                # Priority & Information Extraction
                extracted_info = extract_information(clean_input)
                prio_data = calculate_priority(clean_input, pred_label)
                summary_text = generate_extractive_summary(clean_input, pred_label, extracted_info)

                # Display Main Result Card
                st.markdown(f"""
                <div class="result-box" style="border-color: {display_color};">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="result-title">Predicted Category</span>
                        <span style="background-color: {prio_data['badge_bg']}; color: {prio_data['color']}; font-size: 0.75rem; font-weight: 800; padding: 4px 10px; border-radius: 9999px;">
                            PRIORITY: {prio_data['level']}
                        </span>
                    </div>
                    <div class="result-category" style="color: {display_color};">
                        {display_icon} {display_label}
                    </div>
                    <div class="confidence-label">Prediction Probability: <strong>{confidence*100:.1f}%</strong></div>
                    <div style="font-size: 0.8rem; color: #64748b;">Priority Reason: {prio_data['reason']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.progress(float(confidence))

                # Information Extraction Card Grid
                st.markdown("##### 📌 Extracted Academic Information")
                inf_cols = st.columns(3)
                with inf_cols[0]:
                    st.markdown(f"""<div class="info-card"><div class="info-label">Subject / Course</div><div class="info-val">{extracted_info['Subject']}</div></div>""", unsafe_allow_html=True)
                    st.markdown(f"""<div class="info-card"><div class="info-label">Date</div><div class="info-val">{extracted_info['Date']}</div></div>""", unsafe_allow_html=True)
                with inf_cols[1]:
                    st.markdown(f"""<div class="info-card"><div class="info-label">Deadline</div><div class="info-val">{extracted_info['Deadline']}</div></div>""", unsafe_allow_html=True)
                    st.markdown(f"""<div class="info-card"><div class="info-label">Time</div><div class="info-val">{extracted_info['Time']}</div></div>""", unsafe_allow_html=True)
                with inf_cols[2]:
                    st.markdown(f"""<div class="info-card"><div class="info-label">Location / Venue</div><div class="info-val">{extracted_info['Location']}</div></div>""", unsafe_allow_html=True)

                # Summary Section
                st.markdown(f"""
                <div style="background: #f1f5f9; border-left: 3px solid #64748b; padding: 8px 12px; border-radius: 6px; font-size: 0.85rem; color: #334155; margin-top: 0.6rem;">
                    <strong>📝 Short Summary (Extractive Baseline):</strong> {summary_text}
                </div>
                """, unsafe_allow_html=True)

                # Explain Prediction ("Why this prediction?")
                feature_names = vectorizer.get_feature_names_out()
                row_features = input_vec.tocoo()
                active_terms = []
                for col_i in row_features.col:
                    term = feature_names[col_i]
                    weight = classifier.coef_[class_idx][col_i]
                    active_terms.append((term, weight))
                
                active_terms.sort(key=lambda x: x[1], reverse=True)
                top_terms = [t[0] for t in active_terms if t[1] > 0][:4]

                if top_terms:
                    st.markdown(f"""
                    <div class="explanation-box">
                        <strong>🔍 Why this prediction?</strong> Key detected vocabulary terms: 
                        <code>{", ".join(top_terms)}</code> contributed strongly towards <strong>{pred_label}</strong>.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="explanation-box">
                        <strong>🔍 Why this prediction?</strong> Inferred from global vocabulary distribution across class weights.
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("""
                <div class="disclaimer-note">
                    ℹ️ <strong>Note:</strong> Prediction probability represents the model's confidence for this specific input text, not overall system accuracy.
                </div>
                """, unsafe_allow_html=True)

                # Save to session history
                st.session_state["history"].append({
                    "Timestamp": datetime.now().strftime("%H:%M:%S"),
                    "Announcement": clean_input[:80] + ("..." if len(clean_input) > 80 else ""),
                    "Category": pred_label,
                    "Confidence": f"{confidence*100:.1f}%",
                    "Priority": prio_data["level"],
                    "Subject": extracted_info["Subject"],
                    "Deadline": extracted_info["Deadline"]
                })
        else:
            st.info("👈 Enter an announcement or pick a sample button, then click **Classify Announcement**.")

# -----------------------------------------------------------------------------
# 6. Page: DATASET EXPLORER
# -----------------------------------------------------------------------------
elif nav_selection == "📊 Dataset Explorer":
    st.markdown("""
    <div class="main-header">
        <div class="subtitle">Dataset Analysis & Inspection</div>
        <h1>Benchmark Dataset Explorer</h1>
        <div class="description">
            Explore the balanced 200-sample student announcement dataset (40 records per category) created for this NLP problem.
        </div>
    </div>
    """, unsafe_allow_html=True)

    data_path = os.path.join(DATA_DIR, "announcements.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Records", len(df))
        m2.metric("Target Classes", df["label"].nunique())
        m3.metric("Records Per Class", "40 (Balanced)")
        m4.metric("PII Content", "0% (Fully Sanitized)")

        st.markdown("<br>", unsafe_allow_html=True)

        col_chart, col_filter = st.columns([1, 1])
        with col_chart:
            st.markdown("##### 📊 Class Distribution")
            fig, ax = plt.subplots(figsize=(6, 3.5), dpi=200)
            class_counts = df["label"].value_counts()
            colors = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"]
            bars = ax.bar(class_counts.index, class_counts.values, color=colors, width=0.55, edgecolor="#1e293b", linewidth=1)
            ax.set_ylabel("Count", fontsize=9, fontweight="bold")
            ax.set_ylim(0, max(class_counts.values) + 10)
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, h + 1, f"{int(h)}", ha="center", va="bottom", fontsize=8, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_filter:
            st.markdown("##### 🔍 Search & Filter Dataset")
            cat_filter = st.selectbox("Filter by Category", ["All Categories"] + sorted(list(df["label"].unique())))
            search_query = st.text_input("Search keyword in text", "")

            filtered_df = df.copy()
            if cat_filter != "All Categories":
                filtered_df = filtered_df[filtered_df["label"] == cat_filter]
            if search_query:
                filtered_df = filtered_df[filtered_df["text"].str.contains(search_query, case=False, na=False)]
            
            st.caption(f"Showing {len(filtered_df)} matching records")
            st.dataframe(filtered_df, use_container_width=True, hide_index=True, height=220)
    else:
        st.error("Dataset `data/announcements.csv` not found.")

# -----------------------------------------------------------------------------
# 7. Page: MODEL PERFORMANCE
# -----------------------------------------------------------------------------
elif nav_selection == "📈 Model Performance":
    st.markdown("""
    <div class="main-header">
        <div class="subtitle">Empirical Evaluation & Metrics</div>
        <h1>Real Model Performance</h1>
        <div class="description">
            Actual evaluated metrics measured on the isolated 20% held-out test split (40 unseen samples).
        </div>
    </div>
    """, unsafe_allow_html=True)

    if eval_metrics:
        act = eval_metrics["actual_measured_results"]
        tgt = eval_metrics["target_criteria"]
        
        st.markdown("### 🎯 Actual Measured vs. Target Metrics")
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Overall Accuracy", f"{act['accuracy']*100:.2f}%", f"Target: {tgt['target_accuracy']}")
        m_col2.metric("Macro Precision", f"{act['macro_precision']*100:.2f}%")
        m_col3.metric("Macro Recall", f"{act['macro_recall']*100:.2f}%", f"Target: {tgt['target_min_recall']}")
        m_col4.metric("Macro F1-Score", f"{act['macro_f1']:.4f}", f"Target: {tgt['target_macro_f1']}")

        if eval_metrics.get("target_met"):
            st.success("🎉 **Success Criteria Met:** The trained TF-IDF + Logistic Regression model satisfies all target criteria benchmarks on the unseen test set.")

        st.markdown("---")
        
        col_cm, col_report = st.columns([1, 1])

        with col_cm:
            st.markdown("### 🔲 Confusion Matrix Heatmap")
            cm_img_path = os.path.join(EVAL_DIR, "confusion_matrix.png")
            if os.path.exists(cm_img_path):
                st.image(cm_img_path, caption="Actual Confusion Matrix on Test Set (40 Unseen Samples)", use_column_width=True)
            else:
                st.info("Confusion matrix image not found. Run `python evaluation/evaluate_model.py`")

        with col_report:
            st.markdown("### 📋 Classification Report (Per-Class)")
            report = eval_metrics.get("classification_report", {})
            rows = []
            for cls_name in eval_metrics.get("classes", []):
                if cls_name in report:
                    r = report[cls_name]
                    rows.append({
                        "Category": cls_name,
                        "Precision": f"{r['precision']:.2f}",
                        "Recall": f"{r['recall']:.2f}",
                        "F1-Score": f"{r['f1-score']:.2f}",
                        "Support": int(r['support'])
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            
            st.markdown(r"""
            **Metric Descriptions:**
            - **Accuracy (90.00%):** Proportion of total test announcements categorized correctly.
            - **Precision (90.28%):** Accuracy of positive predictions per class (minimizes false alarms).
            - **Recall (90.00%):** Ability of the model to identify all relevant notices for each category.
            - **Macro F1 (0.8999):** Balanced harmonic mean of precision and recall averaged equally across all 5 classes.
            """)
    else:
        st.warning("No evaluation metrics found. Please run `python evaluation/evaluate_model.py` to generate authentic evaluation results.")

# -----------------------------------------------------------------------------
# 8. Page: PREDICTION HISTORY
# -----------------------------------------------------------------------------
elif nav_selection == "🕒 Prediction History":
    st.markdown("""
    <div class="main-header">
        <div class="subtitle">Session Activity Log</div>
        <h1>Prediction History</h1>
        <div class="description">
            Review announcements classified during your current active browser session.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state["history"]:
        hist_df = pd.DataFrame(st.session_state["history"])
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        
        col_clear, col_count = st.columns([1, 4])
        with col_clear:
            if st.button("🗑️ Clear Session History", key="clear_hist_btn_main"):
                st.session_state["history"] = []
                st.rerun()
        with col_count:
            st.caption(f"Total session queries recorded: {len(hist_df)}")
    else:
        st.info("No predictions recorded in this session yet. Head over to **🔮 AI Classifier** to classify announcements.")

    st.markdown("""
    <div class="disclaimer-note">
        🔒 <strong>Privacy Notice:</strong> History is maintained in temporary browser session memory only. No student input is saved to disk or permanent databases.
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 9. Page: ABOUT
# -----------------------------------------------------------------------------
elif nav_selection == "ℹ️ About":
    st.markdown("""
    <div class="main-header">
        <div class="subtitle">Documentation & AI Problem Design</div>
        <h1>About This Project</h1>
        <div class="description">
            Complete problem design, methodology, and engineering specification developed for SWYNEX Technologies Internship.
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Problem & Users", "AI Task & Architecture", "Dataset Engineering", "Evaluation Strategy", "Ethical & Privacy Guardrails"
    ])

    with tab1:
        st.markdown("""
        ### Problem Statement & Target Users
        - **Organization:** SWYNEX Technologies Internship – Task 1
        - **Problem:** Academic institutions distribute high volumes of notices across mixed channels. Students experience cognitive overload and miss deadlines.
        - **Primary Users:** College students needing rapid triage and alerts for actionable academic notices.
        - **Secondary Users:** Faculty members and administrative staff broadcasting circulars.
        """)

    with tab2:
        st.markdown("""
        ### AI Task Formulation
        - **Task:** Multi-Class Natural Language Processing (NLP) / Text Classification & Information Extraction.
        - **Input:** Single-message college announcement text string (English).
        - **Output:** Category (`Exam`, `Assignment`, `Attendance`, `Internship`, `Event`), Prediction Probability, Priority Level, Extracted Entities, and Short Summary.
        - **Pipeline:** Raw Text $\\rightarrow$ TF-IDF Vectorizer (Unigrams + Bigrams) $\\rightarrow$ Logistic Regression Classifier $\\rightarrow$ Rule-based NLP Entity & Urgency Extraction.
        """)

    with tab3:
        st.markdown("""
        ### Dataset Engineering
        - **Sample Corpus:** 200 curated, realistic sample announcements.
        - **Balance:** Exactly 40 samples per category across all 5 classes.
        - **Sanitization:** 100% anonymized with zero Personally Identifiable Information (no student names, IDs, or phone numbers).
        """)

    with tab4:
        st.markdown(r"""
        ### Evaluation Methodology & Success Criteria
        - **Partitioning:** 80% Stratified Training split (160 samples) / 20% Held-Out Testing split (40 samples).
        - **Target Criteria:** Accuracy $\ge 85\%$, Macro F1 $\ge 0.80$.
        - **Actual Measured:** Accuracy **90.00%**, Macro F1 **0.8999**.
        """)

    with tab5:
        st.markdown("""
        ### Ethical & Privacy Considerations
        1. **Zero PII:** No personal student records or grades are processed or stored.
        2. **Session-Only Logging:** User test queries are stored only in memory for the active browser session.
        3. **Advisory Scope:** The system acts strictly as an informational productivity tool and must not be used for disciplinary or high-stakes academic decisions.
        """)

# -----------------------------------------------------------------------------
# 10. Footer
# -----------------------------------------------------------------------------
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.82rem; padding: 2rem 0 1rem 0;">
    SWYNEX Technologies &bull; Task 1: AI Problem Design &bull; Student Announcement Intelligence System
</div>
""", unsafe_allow_html=True)
