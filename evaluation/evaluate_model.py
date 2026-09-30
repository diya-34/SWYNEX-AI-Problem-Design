"""
SWYNEX Technologies Internship - Task 1: AI Problem Design
Module: evaluation/evaluate_model.py
Description: Evaluates the trained model on the held-out test split, computes real metrics 
             (Accuracy, Precision, Recall, Macro F1), generates a confusion matrix heatmap, 
             and saves actual evaluation metrics to JSON.
"""

import os
import json
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

def evaluate():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "model")
    eval_dir = os.path.join(base_dir, "evaluation")
    screenshots_dir = os.path.join(base_dir, "screenshots")
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(screenshots_dir, exist_ok=True)

    print("=" * 70)
    print(" SWYNEX Technologies Internship - Task 1: Real Model Evaluation")
    print(" Evaluating on Held-Out Test Set (20% Stratified Split)")
    print("=" * 70)

    # 1. Load Model, Vectorizer, and Test Split
    vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
    classifier_path = os.path.join(model_dir, "classifier.pkl")
    test_data_path = os.path.join(model_dir, "test_data.csv")

    if not (os.path.exists(vectorizer_path) and os.path.exists(classifier_path) and os.path.exists(test_data_path)):
        print("\n[ERROR] Model artifacts or test data not found. Please run 'python model/train_model.py' first.")
        return

    vectorizer = joblib.load(vectorizer_path)
    classifier = joblib.load(classifier_path)
    test_df = pd.read_csv(test_data_path)

    X_test = test_df["text"]
    y_test = test_df["label"]
    labels = sorted(list(set(y_test)))

    # 2. Compute Predictions
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = classifier.predict(X_test_tfidf)

    # 3. Calculate Actual Metrics
    acc = float(accuracy_score(y_test, y_pred))
    macro_prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    # 4. Save Metrics to JSON
    metrics_data = {
        "evaluation_split": "Held-Out Test Set (20%)",
        "total_test_samples": len(test_df),
        "target_criteria": {
            "target_accuracy": ">= 85%",
            "target_macro_f1": ">= 0.80",
            "target_min_recall": ">= 0.75"
        },
        "actual_measured_results": {
            "accuracy": round(acc, 4),
            "macro_precision": round(macro_prec, 4),
            "macro_recall": round(macro_rec, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4)
        },
        "target_met": bool(acc >= 0.85 and macro_f1 >= 0.80),
        "classification_report": report_dict,
        "classes": labels,
        "confusion_matrix": cm.tolist()
    }

    metrics_json_path = os.path.join(eval_dir, "metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"\n[OK] Saved actual computed metrics to -> {metrics_json_path}")

    # 5. Generate and Save Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        linewidths=1,
        linecolor="#cbd5e1"
    )
    plt.title("SWYNEX Task 1 - Actual Confusion Matrix on Test Set", fontsize=12, fontweight="bold", pad=15)
    plt.xlabel("Predicted Category", fontsize=11, fontweight="bold")
    plt.ylabel("True Category (Ground Truth)", fontsize=11, fontweight="bold")
    plt.tight_layout()

    cm_eval_path = os.path.join(eval_dir, "confusion_matrix.png")
    cm_screen_path = os.path.join(screenshots_dir, "confusion_matrix.png")
    plt.savefig(cm_eval_path)
    plt.savefig(cm_screen_path)
    plt.close()

    print(f"[OK] Saved Confusion Matrix chart -> {cm_eval_path}")
    print(f"[OK] Mirrored to screenshots directory -> {cm_screen_path}")

    # 6. Print Terminal Report
    print("\n" + "-" * 70)
    print(" ACTUAL MEASURED EVALUATION METRICS (Held-Out Test Set):")
    print("-" * 70)
    print(f"  * Overall Accuracy       : {acc * 100:.2f}%  (Target: >= 85%)")
    print(f"  * Macro Precision        : {macro_prec * 100:.2f}%")
    print(f"  * Macro Recall           : {macro_rec * 100:.2f}%")
    print(f"  * Macro F1-Score         : {macro_f1:.4f}  (Target: >= 0.80)")
    print(f"  * Weighted F1-Score      : {weighted_f1:.4f}")
    print(f"  * Target Criteria Status : {'MET [PASSED]' if metrics_data['target_met'] else 'BELOW THRESHOLD'}")
    print("\n--- Detailed Per-Class Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=labels, zero_division=0))
    print("=" * 70)

if __name__ == "__main__":
    evaluate()
