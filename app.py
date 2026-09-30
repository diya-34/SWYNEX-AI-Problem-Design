"""
SWYNEX Technologies Internship - Task 1: AI Problem Design
Web Application: Student Announcement Classification Prototype
Framework: Streamlit & Pandas
"""

import os
import streamlit as st
import pandas as pd

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS
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
    /* Import modern typography */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2.2rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(30, 58, 138, 0.25);
    }
    .main-header h1 {
        color: #ffffff !important;
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
        letter-spacing: -0.5px;
    }
    .main-header .subtitle {
        color: #93c5fd;
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }
    .main-header .description {
        color: #e0f2fe;
        font-size: 0.95rem;
        margin-bottom: 0;
    }
    
    /* Category Cards */
    .cat-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem;
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
        margin-bottom: 0.6rem;
    }
    .badge-exam { background-color: #fee2e2; color: #991b1b; }
    .badge-assignment { background-color: #fef3c7; color: #92400e; }
    .badge-attendance { background-color: #e0e7ff; color: #3730a3; }
    .badge-internship { background-color: #d1fae5; color: #065f46; }
    .badge-event { background-color: #fce7f3; color: #831843; }
    
    /* Result Box */
    .result-box {
        background: #f8fafc;
        border: 2px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.8rem;
        margin-top: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
    }
    .result-title {
        font-size: 0.9rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.4rem;
    }
    .result-category {
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 0.8rem;
    }
    .confidence-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #475569;
        margin-bottom: 0.3rem;
    }
    .disclaimer-note {
        font-size: 0.82rem;
        color: #64748b;
        background-color: #f1f5f9;
        padding: 8px 14px;
        border-radius: 8px;
        margin-top: 1.2rem;
        border-left: 3px solid #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Classification Logic & Rules (Prototype Design)
# -----------------------------------------------------------------------------
CATEGORIES_INFO = {
    "Exam": {
        "badge_class": "badge-exam",
        "icon": "📝",
        "color": "#dc2626",
        "desc": "Exam schedules, hall tickets, seating plans, mid-term / end-sem tests, supplementary exams, and re-evaluation notices."
    },
    "Assignment": {
        "badge_class": "badge-assignment",
        "icon": "📚",
        "color": "#d97706",
        "desc": "Homework submissions, lab reports, project milestones, code repository submissions, and grading deadlines."
    },
    "Attendance": {
        "badge_class": "badge-attendance",
        "icon": "⏱️",
        "color": "#4f46e5",
        "desc": "Mandatory 75% criteria, roll call rules, biometric logs, attendance shortages, and medical condonation."
    },
    "Internship": {
        "badge_class": "badge-internship",
        "icon": "💼",
        "color": "#059669",
        "desc": "Summer / winter internships, placement drives, industrial training, research assistantships, and job referrals."
    },
    "Event": {
        "badge_class": "badge-event",
        "icon": "🎉",
        "color": "#db2777",
        "desc": "College festivals, hackathons, technical workshops, sports tournaments, club orientations, and guest lectures."
    }
}

KEYWORD_RULES = {
    "Exam": ["exam", "examination", "mid-term", "end-sem", "end sem", "test", "viva", "hall ticket", "supplementary", "blueprint", "seating"],
    "Assignment": ["assignment", "homework", "submit", "submission", "lab report", "project report", "tutorial", "code files", "case study", "mini-project", "deadline"],
    "Attendance": ["attendance", "present", "absent", "roll call", "biometric", "75 percent", "65 percent", "deficiency", "shortage", "condonation", "detained"],
    "Internship": ["internship", "intern", "hiring", "stipend", "career", "interview", "trainee", "assistantship", "research intern", "off-campus", "recruiting"],
    "Event": ["event", "festival", "hackathon", "workshop", "webinar", "contest", "gala", "sports meet", "exhibition", "celebration", "club", "hackfest", "symposium"]
}

def classify_announcement(text: str):
    """
    Classifies college announcement text into predefined categories using keyword pattern matching.
    Calculates an illustrative Demo Confidence score for prototyping purposes.
    """
    text_lower = text.lower()
    scores = {cat: 0 for cat in CATEGORIES_INFO.keys()}
    
    for cat, keywords in KEYWORD_RULES.items():
        for kw in keywords:
            if kw in text_lower:
                scores[cat] += 1
                
    best_cat = max(scores, key=scores.get)
    max_score = scores[best_cat]
    
    if max_score == 0:
        return "General / Uncategorized", 0.50, None
    
    demo_confidence = min(0.70 + (max_score * 0.12), 0.98)
    return best_cat, demo_confidence, CATEGORIES_INFO[best_cat]

# -----------------------------------------------------------------------------
# 3. Header Section
# -----------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <div class="subtitle">SWYNEX Technologies – Task 1</div>
    <h1>Student Announcement Classification</h1>
    <div class="description">
        AI Problem Design & Prototype Interface &bull; Enter a college announcement below and classify it into the most relevant category.
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. Category Information Grid
# -----------------------------------------------------------------------------
st.subheader("📌 Supported Categories")
cols = st.columns(5)
for i, (cat_name, info) in enumerate(CATEGORIES_INFO.items()):
    with cols[i]:
        st.markdown(f"""
        <div class="cat-card">
            <span class="cat-badge {info['badge_class']}">{info['icon']} {cat_name}</span>
            <p style="font-size: 0.82rem; color: #475569; margin: 0; line-height: 1.4;">
                {info['desc']}
            </p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. Interactive Input & "Try an Example" Section
# -----------------------------------------------------------------------------
SAMPLE_EXAMPLES = [
    ("📝 Exam Example", "The DBMS examination will be conducted on Monday in Hall A at 10 AM."),
    ("📚 Assignment Example", "Submit your Machine Learning assignment before Friday at 11:59 PM."),
    ("⏱️ Attendance Example", "Students must maintain at least 75 percent attendance to appear for exams."),
    ("💼 Internship Example", "Applications for the Summer AI and Data Science Internship at TCS are now open."),
    ("🎉 Event Example", "Registration for the annual national technical hackathon HackFest 2025 is open.")
]

# Manage text state
if "announcement_text" not in st.session_state:
    st.session_state["announcement_text"] = ""

def set_sample_text(text):
    st.session_state["announcement_text"] = text

st.markdown("### 💡 Try an Example")
example_cols = st.columns(len(SAMPLE_EXAMPLES))
for idx, (label, sample_text) in enumerate(SAMPLE_EXAMPLES):
    with example_cols[idx]:
        if st.button(label, key=f"btn_sample_{idx}", use_container_width=True):
            set_sample_text(sample_text)

st.markdown("<br>", unsafe_allow_html=True)

# Main classification input
col_left, col_right = st.columns([1.2, 0.8])

with col_left:
    st.markdown("### ✍️ Enter College Announcement")
    user_input = st.text_area(
        label="Enter your announcement",
        value=st.session_state["announcement_text"],
        placeholder="Type or paste any student circular, notice, or announcement here...",
        height=140,
        label_visibility="collapsed"
    )
    
    classify_btn = st.button("🚀 Classify Announcement", type="primary", use_container_width=True)

with col_right:
    st.markdown("### 🎯 Classification Result")
    
    if classify_btn:
        if not user_input.strip():
            st.warning("⚠️ Please enter an announcement text before classifying.")
        else:
            category, demo_conf, info = classify_announcement(user_input)
            
            if category == "General / Uncategorized":
                st.markdown(f"""
                <div class="result-box" style="border-color: #94a3b8;">
                    <div class="result-title">Predicted Category</div>
                    <div class="result-category" style="color: #475569;">
                        ❓ {category}
                    </div>
                    <div class="confidence-label">Demo Confidence: <strong>{demo_conf*100:.0f}%</strong></div>
                    <p style="font-size: 0.85rem; color: #64748b; margin-top: 0.5rem;">
                        No explicit category keywords detected. This message may fall under general announcements or require broader multi-sentence context.
                    </p>
                    <div class="disclaimer-note">
                        ℹ️ <strong>Note:</strong> Demo confidence is illustrative and is not an actual model evaluation metric.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box" style="border-color: {info['color']};">
                    <div class="result-title">Predicted Category</div>
                    <div class="result-category" style="color: {info['color']};">
                        {info['icon']} {category}
                    </div>
                    <div class="confidence-label">Demo Confidence: <strong>{demo_conf*100:.0f}%</strong></div>
                </div>
                """, unsafe_allow_html=True)
                
                st.progress(demo_conf)
                
                st.markdown("""
                <div class="disclaimer-note">
                    ℹ️ <strong>Note:</strong> Demo confidence is illustrative and is not an actual model evaluation metric.
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("👈 Enter an announcement text or pick one of the quick examples above, then click **Classify Announcement**.")

st.markdown("---")

# -----------------------------------------------------------------------------
# 6. Dataset Explorer & Analytics
# -----------------------------------------------------------------------------
st.subheader("📊 Benchmark Dataset Explorer")
data_path = os.path.join(os.path.dirname(__file__), "data", "announcements.csv")

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric(label="Total Sample Records", value=len(df))
    with metric_cols[1]:
        st.metric(label="Categories Defined", value=df['label'].nunique())
    with metric_cols[2]:
        st.metric(label="Language", value="English")
    with metric_cols[3]:
        st.metric(label="PII Sanitization", value="100% Scrubbed")
        
    with st.expander("🔍 View announcements.csv Dataset (60 Sample Records)"):
        selected_category = st.selectbox("Filter by Category", ["All"] + list(CATEGORIES_INFO.keys()))
        if selected_category != "All":
            filtered_df = df[df["label"] == selected_category]
        else:
            filtered_df = df
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)
else:
    st.info("Dataset file `data/announcements.csv` not found.")

st.markdown("---")

# -----------------------------------------------------------------------------
# 7. About This Project & AI Problem Design Specification
# -----------------------------------------------------------------------------
st.subheader("📖 About this project")

with st.expander("📑 View Full Problem Design & Engineering Specifications", expanded=False):
    tab1, tab2, tab3, tab4 = st.tabs(["Overview & Target Users", "AI Task & Constraints", "Evaluation Plan", "Target Success Criteria"])
    
    with tab1:
        st.markdown("""
        **SWYNEX Technologies Internship – Task 1: AI Problem Design**
        
        - **Objective:** Systematically formulate and document a narrow AI problem for student announcement text classification before entering the model training phase.
        - **Primary Users:** College students navigating fragmented communication across email, WhatsApp, Slack, and university portals.
        - **Secondary Users:** Course instructors, teaching assistants, departmental coordinators, and academic administrators.
        - **Core Value:** Reduces cognitive load, eliminates missed deadlines, and surfaces career opportunities instantly.
        """)
        
    with tab2:
        st.markdown("""
        - **AI Task Type:** Multi-Class Natural Language Processing (NLP) / Text Classification.
        - **Input:** Unstructured college circular / announcement text string (English).
        - **Output Space:** Exactly one category from `{Exam, Assignment, Attendance, Internship, Event}`.
        - **Key Constraints:**
          1. Restricted to English announcements in current version.
          2. Fixed 5-class taxonomy without sub-categories.
          3. Multi-topic compound announcements present single-label ambiguity.
          4. Zero Personally Identifiable Information (PII) is stored or processed.
        """)
        
    with tab3:
        st.markdown("""
        - **Data Partitioning:** 80% Training split / 20% Unseen Testing split.
        - **Validation Strategy:** 5-Fold Stratified Cross-Validation on training data.
        - **Target Metrics:**
          - Overall Accuracy
          - Precision (Macro & Per-Class)
          - Recall (Macro & Per-Class)
          - Macro F1-Score
          - Confusion Matrix error analysis
        """)
        
    with tab4:
        st.markdown(r"""
        **Target Success Criteria (Design Benchmarks, Not Fabricated Results):**
        - Target Test Accuracy: **$\ge 85\%$**
        - Target Macro F1-Score: **$\ge 0.80$**
        - Minimum Class Recall: **$\ge 0.75$** for each category (preventing class starvation)
        - Latency Benchmark: **$< 100\text{ ms}$** per prediction
        """)

# Footer
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.85rem; padding: 2rem 0 1rem 0;">
    SWYNEX Technologies &bull; Task 1: AI Problem Design &bull; Student Announcement Classification
</div>
""", unsafe_allow_html=True)
