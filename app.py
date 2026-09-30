"""
SWYNEX Technologies Internship - Task 1: AI Problem Design
Web Application: Student Announcement Classification System (ML Prototype)
Tech Stack: Python, Streamlit, Pandas, Scikit-Learn, Joblib, Matplotlib, Seaborn
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# 1. Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SWYNEX Technologies – Task 1",
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
        padding: 2rem 2.2rem;
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
        font-size: 1.1rem;
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
    
    /* Result Box */
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
        margin-bottom: 0.6rem;
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
<div style="text-align: center; padding: 1rem 0;">
    <h2 style="margin: 0; color: #1e3a8a; font-weight: 800; font-size: 1.4rem;">SWYNEX AI</h2>
    <p style="margin: 0; color: #64748b; font-size: 0.82rem; font-weight: 600;">Task 1: AI Problem Design</p>
</div>
""", unsafe_allow_html=True)

nav_selection = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🔮 Classifier", "📊 Dataset Explorer", "📈 Model Performance", "ℹ️ About"],
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
    st.sidebar.metric(label="Evaluated Test Accuracy", value=f"{acc_val:.1f}%")
    st.sidebar.metric(label="Evaluated Macro F1", value=f"{eval_metrics['actual_measured_results']['macro_f1']:.4f}")

# -----------------------------------------------------------------------------
# 4. Page: HOME
# -----------------------------------------------------------------------------
if nav_selection == "🏠 Home":
    st.markdown("""
    <div class="main-header">
        <div class="subtitle">SWYNEX Technologies – Task 1</div>
        <h1>Student Announcement Classification</h1>
        <div class="description">
            A practical NLP Text Classification system designed to categorize unstructured college circulars into 5 actionable academic categories.
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
        st.markdown("### 🎯 Problem Overview")
        st.markdown("""
        In higher education institutions, students and faculty navigate a deluge of daily notices across WhatsApp, emails, LMS, and departmental notice boards.
        
        Because notices arrive unorganized, students face:
        - **Critical deadline omissions:** Missing fee submissions, exam form dates, and homework cutoffs.
        - **Buried career opportunities:** Internship postings and competitive hackathons get overlooked.
        - **Information clutter:** Inability to quickly prioritize high-urgency notifications.

        **Proposed AI Solution:** A lightweight Natural Language Processing pipeline that takes raw announcement text and outputs a high-confidence category classification.
        """)
        
        st.info("💡 **Ready to test?** Navigate to **🔮 Classifier** from the sidebar to test predictions live!")

    with col2:
        st.markdown("### 🏗️ Machine Learning Architecture")
        st.markdown("""
        ```
        +-------------------------------------------------------+
        |                 Student Announcement                  |
        |      "Submit your ML assignment before Friday"        |
        +-------------------------------------------------------+
                                   │
                                   ▼
        +-------------------------------------------------------+
        |                  TF-IDF Vectorizer                    |
        |          (Unigrams + Bigrams, Sublinear TF)           |
        +-------------------------------------------------------+
                                   │
                                   ▼
        +-------------------------------------------------------+
        |             Logistic Regression Classifier            |
        |               (Multinomial Softmax Prob)              |
        +-------------------------------------------------------+
                                   │
                                   ▼
        +-------------------------------------------------------+
        |              Predicted Class & Confidence             |
        |                  [ ASSIGNMENT: 98.4% ]                |
        +-------------------------------------------------------+
        ```
        """)

# -----------------------------------------------------------------------------
# 5. Page: CLASSIFIER
# -----------------------------------------------------------------------------
elif nav_selection == "🔮 Classifier":
    st.markdown("""
    <div class="main-header">
        <div class="subtitle">Live Inference Prototype</div>
        <h1>Classify College Announcement</h1>
        <div class="description">
            Enter any announcement or pick a sample notice below to inspect the real-time ML prediction and class probabilities.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Example Announcements
    SAMPLE_EXAMPLES = [
        ("📝 Exam", "The End Semester Examination schedule for Computer Engineering has been published on the student portal."),
        ("📚 Assignment", "Submit your Machine Learning lab assignment before Friday at 11:59 PM."),
        ("⏱️ Attendance", "All students must maintain at least 75 percent attendance to be eligible for the semester examinations."),
        ("💼 Internship", "Applications for the Summer AI and Data Science Internship at TCS are now open for final year students."),
        ("🎉 Event", "Registration for HackFest 2025 annual national technical hackathon is officially open.")
    ]

    st.markdown("##### 💡 Try a Sample Announcement")
    ex_cols = st.columns(len(SAMPLE_EXAMPLES))
    for idx, (btn_title, sample_text) in enumerate(SAMPLE_EXAMPLES):
        with ex_cols[idx]:
            if st.button(btn_title, key=f"ex_btn_{idx}", use_container_width=True):
                set_text(sample_text)

    st.markdown("<br>", unsafe_allow_html=True)

    col_input, col_out = st.columns([1.2, 0.8])

    with col_input:
        st.markdown("### ✍️ Announcement Input")
        user_input = st.text_area(
            label="Enter your announcement",
            value=st.session_state["input_text"],
            placeholder="Type or paste any student circular, notice, or announcement here...",
            height=140,
            label_visibility="collapsed"
        )
        
        classify_btn = st.button("🚀 Classify Announcement", type="primary", use_container_width=True)

    with col_out:
        st.markdown("### 🎯 Classification Result")
        
        if classify_btn:
            if not user_input.strip():
                st.warning("⚠️ Please enter an announcement text before classifying.")
            elif not model_loaded:
                st.error("❌ Model not loaded! Please run `python model/train_model.py` to train the model first.")
            else:
                # 1. Vectorize and Predict
                input_vec = vectorizer.transform([user_input])
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

                # Display Main Result Card
                st.markdown(f"""
                <div class="result-box" style="border-color: {display_color};">
                    <div class="result-title">Predicted Category</div>
                    <div class="result-category" style="color: {display_color};">
                        {display_icon} {display_label}
                    </div>
                    <div class="confidence-label">Prediction Probability: <strong>{confidence*100:.1f}%</strong></div>
                </div>
                """, unsafe_allow_html=True)
                
                st.progress(float(confidence))

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
                        <code>{", ".join(top_terms)}</code> influenced the model towards <strong>{pred_label}</strong>.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="explanation-box">
                        <strong>🔍 Why this prediction?</strong> Model inferred intent from context distribution across class weights.
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("""
                <div class="disclaimer-note">
                    ℹ️ <strong>Note:</strong> Prediction probability represents the model's confidence for this specific input text, not overall system accuracy.
                </div>
                """, unsafe_allow_html=True)

                # Save to session history
                st.session_state["history"].append({
                    "Announcement": user_input[:80] + ("..." if len(user_input) > 80 else ""),
                    "Category": pred_label,
                    "Confidence": f"{confidence*100:.1f}%"
                })
        else:
            st.info("👈 Enter an announcement or pick a sample button, then click **Classify Announcement**.")

    # Prediction History Section
    st.markdown("---")
    st.markdown("### 🕒 Session Prediction History")
    if st.session_state["history"]:
        hist_df = pd.DataFrame(st.session_state["history"])
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        if st.button("🗑️ Clear History", key="clear_hist_btn"):
            st.session_state["history"] = []
            st.rerun()
    else:
        st.caption("No classifications in this session yet. Test announcements above to view session history.")

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
            **Key Performance Insights:**
            - **Internship & Assignment:** Demonstrated high precision and recall ($F_1 \ge 0.94$) due to distinctive domain vocabularies (`internship`, `hiring`, `submit`, `assignment`).
            - **Exam vs. Attendance:** Handled clean separation, with attendance cues centered around `biometric`, `75 percent`, and `absent`.
            """)
    else:
        st.warning("No evaluation metrics found. Please run `python evaluation/evaluate_model.py` to generate authentic evaluation results.")

# -----------------------------------------------------------------------------
# 8. Page: ABOUT
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
        - **Task:** Multi-Class Natural Language Processing (NLP) / Text Classification.
        - **Input:** Single-message college announcement text string (English).
        - **Output:** Exactly one of 5 classes: `Exam`, `Assignment`, `Attendance`, `Internship`, `Event`.
        - **Pipeline:** Raw Text $\\rightarrow$ TF-IDF Vectorizer (Unigrams + Bigrams) $\\rightarrow$ Logistic Regression Classifier.
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
# 9. Footer
# -----------------------------------------------------------------------------
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.82rem; padding: 2rem 0 1rem 0;">
    SWYNEX Technologies &bull; Task 1: AI Problem Design &bull; Student Announcement Classification System
</div>
""", unsafe_allow_html=True)
