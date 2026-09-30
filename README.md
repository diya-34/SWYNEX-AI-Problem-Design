# 🤖 SWYNEX AI Problem Design

## Student Announcement Intelligence System

An AI-powered system that automatically classifies college/student announcements into meaningful categories such as **Exam, Assignment, Attendance, Internship, and Event**.

This project was developed as part of the **SWYNEX Technologies Internship – Task 1: AI Problem Design**.

---

## 📌 Problem Statement

Students receive many announcements through WhatsApp groups, emails, notice boards, and college portals.

Important information can easily be missed because announcements are mixed together.

For example:

> "Submit your Machine Learning assignment before Friday at 11:59 PM."

The system automatically identifies this announcement as:

**Category → Assignment**

The goal is to make student announcements easier to organize and understand using AI/NLP.

---

## 🎯 Objectives

* Automatically classify student announcements.
* Reduce manual sorting of announcements.
* Identify the category of an announcement.
* Display prediction confidence.
* Provide a simple and user-friendly interface.
* Evaluate the machine learning model using standard metrics.
* Create a foundation for a future real-world student notification system.

---

## 🗂️ Announcement Categories

The current system supports five categories:

| Category      | Example                                               |
| ------------- | ----------------------------------------------------- |
| 📚 Exam       | "The DBMS examination will be conducted on Monday."   |
| 📝 Assignment | "Submit your ML assignment before Friday."            |
| 📊 Attendance | "Students must maintain 75% attendance."              |
| 💼 Internship | "Applications for the Summer AI Internship are open." |
| 🎉 Event      | "Registration for HackFest is now open."              |

---

## 🧠 AI / Machine Learning Approach

The project uses Natural Language Processing (NLP) and Machine Learning.

### Workflow

```text
Student Announcement
        ↓
Text Preprocessing
        ↓
TF-IDF Vectorization
        ↓
Machine Learning Model
        ↓
Category Prediction
        ↓
Confidence Score
        ↓
Result displayed to user
```

### Technologies

* Python
* Natural Language Processing (NLP)
* Scikit-learn
* Pandas
* NumPy
* TF-IDF
* Logistic Regression
* Streamlit
* Joblib
* Matplotlib
* Seaborn

---

## 🖥️ Application Features

### 1. AI Classifier

Users can enter an announcement and receive an automatically predicted category.

Example:

```text
Input:
Submit your Machine Learning assignment before Friday.

Output:
Assignment
```

### 2. Confidence Score

The application displays the model's prediction confidence.

> Note: Prediction confidence is different from overall model accuracy.

### 3. Dataset Explorer

The application can be used to inspect the announcement dataset and understand the distribution of different categories.

### 4. Model Performance

The project supports evaluation using:

* Accuracy
* Precision
* Recall
* F1 Score
* Classification Report
* Confusion Matrix

### 5. Prediction History

Predictions made during the current application session can be reviewed.

---

## 📁 Project Structure

```text
SWYNEX-AI-Problem-Design/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── announcements.csv
│
├── model/
│   ├── train_model.py
│   ├── classifier.pkl
│   └── vectorizer.pkl
│
├── evaluation/
│   ├── evaluate_model.py
│   └── confusion_matrix.png
│
├── scripts/
│   └── classify_demo.py
│
├── docs/
│   └── AI_Problem_Design.md
│
└── screenshots/
```

---

## ⚙️ Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/diya-34/SWYNEX-AI-Problem-Design.git
```

### Step 2: Open the Project

```bash
cd SWYNEX-AI-Problem-Design
```

### Step 3: Create a Virtual Environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will normally open at:

```text
http://localhost:8501
```

---

## 🧪 Example Predictions

### Example 1

```text
The DBMS examination will be conducted on Monday morning in Hall B.
```

**Prediction:** `Exam`

### Example 2

```text
Submit your Machine Learning lab assignment before Friday at 11:59 PM.
```

**Prediction:** `Assignment`

### Example 3

```text
All students must maintain at least 75 percent attendance.
```

**Prediction:** `Attendance`

### Example 4

```text
Applications for the Summer AI and Data Science Internship are now open.
```

**Prediction:** `Internship`

### Example 5

```text
Registration for the national technical hackathon is now open.
```

**Prediction:** `Event`

---

## 📊 Model Evaluation

The model should be evaluated using a separate test set.

Important evaluation metrics include:

```text
Accuracy
Precision
Recall
F1 Score
Confusion Matrix
```

The project should report the **actual measured results** from the trained model.

> Model performance numbers should not be hard-coded or presented as actual results unless they were obtained through evaluation.

---

## 🔮 Future Scope

This project can be expanded into a real-world **Student Announcement Intelligence Platform**.

### Phase 1 — Machine Learning

* Improved dataset
* Better text preprocessing
* Model comparison
* Hyperparameter tuning

### Phase 2 — Advanced NLP

* Automatic summarization
* Deadline extraction
* Date and time extraction
* Location extraction
* Priority detection
* Gujarati/Hinglish support

### Phase 3 — Backend

* FastAPI backend
* Database integration
* User authentication
* Student/Faculty/Admin roles

### Phase 4 — Generative AI

* LLM-based announcement summaries
* Natural language questions and answers
* Personalized student assistance

### Phase 5 — RAG

The system can be connected with college documents such as:

* Academic calendars
* Exam schedules
* College notices
* Internship documents
* Department information
* Student guidelines

This would allow students to ask questions such as:

```text
When is my next exam?
```

```text
What is the assignment deadline?
```

```text
What attendance is required?
```

### Phase 6 — Notifications

Future versions can send notifications through:

* Email
* College portal
* Mobile application
* WhatsApp/other supported notification systems

---

## 🌍 Real-World Vision

The long-term goal is to transform the project from a simple classifier into a complete:

> **AI-powered Student Announcement & Notification Assistant**

```text
College Announcement
        ↓
AI Processing
        ↓
Classification
        ↓
Important Information Extraction
        ↓
Summary
        ↓
Priority Detection
        ↓
Personalized Student Notification
```

---

## 🔐 Privacy

The project should avoid collecting unnecessary personal information.

For real-world deployment:

* Do not store sensitive student data unnecessarily.
* Use authentication and authorization.
* Protect database credentials.
* Use environment variables for API keys.
* Follow applicable privacy and security requirements.

---

## 👩‍💻 Developed By

**Diya Patel**

B.Tech Artificial Intelligence & Machine Learning
CHARUSAT – Charotar University of Science & Technology

### Areas of Interest

* Artificial Intelligence
* Machine Learning
* Deep Learning
* Natural Language Processing
* Generative AI
* Large Language Models

---

## 🏢 Internship

**SWYNEX Technologies Internship**

### Task 1

**AI Problem Design – Student Announcement Classification**

---

## 📌 Project Status

🚧 **Currently in development**

The current version focuses on student announcement classification using NLP and Machine Learning, with future plans for summarization, information extraction, RAG, notifications, and production deployment.

---

## ⭐ Acknowledgement

This project was developed as part of the **SWYNEX Technologies Internship Program** to explore practical AI problem solving and real-world machine learning applications.
