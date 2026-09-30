"""
SWYNEX Technologies Internship - Task 1: AI Problem Design
Script: generate_visuals.py
Description: Generates dataset analytics charts and workflow visualization images for the screenshots/ directory.
"""

import os
import csv
from collections import Counter
import matplotlib.pyplot as plt

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "announcements.csv")
    screenshots_dir = os.path.join(base_dir, "screenshots")
    os.makedirs(screenshots_dir, exist_ok=True)

    print("=" * 60)
    print(" SWYNEX Technologies Internship - Task 1: AI Problem Design")
    print(" Student Announcement Classification Dataset Analysis")
    print("=" * 60)

    # 1. Load and parse dataset
    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)

    total_samples = len(records)
    counts = Counter(r["label"] for r in records)

    print(f"\n[+] Total Announcements Loaded: {total_samples}")
    print("\n--- Category Breakdown ---")
    for category, count in counts.most_common():
        pct = (count / total_samples) * 100
        print(f"  * {category:<12} : {count:2d} samples ({pct:5.1f}%)")

    # 2. Generate Dataset Distribution Plot
    categories = list(counts.keys())
    values = [counts[c] for c in categories]
    colors = ["#2563eb", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"]

    plt.figure(figsize=(9, 5), dpi=300)
    bars = plt.bar(categories, values, color=colors, width=0.55, edgecolor="#1e293b", linewidth=1.2)
    
    plt.title("SWYNEX AI Problem Design - Student Announcement Dataset Distribution", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Announcement Categories", fontsize=11, fontweight="bold")
    plt.ylabel("Number of Samples", fontsize=11, fontweight="bold")
    plt.ylim(0, max(values) + 4)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.4, f"{int(yval)} samples", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    dist_img_path = os.path.join(screenshots_dir, "dataset_distribution.png")
    plt.savefig(dist_img_path)
    plt.close()
    print(f"\n[OK] Generated Dataset Distribution Chart -> {dist_img_path}")

    # 3. Generate AI System Pipeline Diagram
    fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
    ax.axis("off")

    boxes = [
        ("Input Announcement\n(Raw Text)", (0.12, 0.5), "#3b82f6"),
        ("NLP Preprocessing\n(Tokens / Clean)", (0.37, 0.5), "#10b981"),
        ("Feature Extraction\n(TF-IDF / Embeddings)", (0.62, 0.5), "#f59e0b"),
        ("Text Classifier\n(Logistic Reg / BERT)", (0.87, 0.5), "#8b5cf6"),
    ]

    for title, (x, y), color in boxes:
        ax.text(x, y, title, ha="center", va="center", fontsize=9, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.6", facecolor=color, edgecolor="#1e293b", linewidth=1.5))

    # Add arrows
    for i in range(len(boxes) - 1):
        x1 = boxes[i][1][0] + 0.08
        x2 = boxes[i+1][1][0] - 0.08
        y = 0.5
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="->", lw=2, color="#334155"))

    ax.set_title("SWYNEX Technologies - Student Announcement AI Classification Pipeline", fontsize=12, fontweight="bold", pad=20)
    
    pipeline_img_path = os.path.join(screenshots_dir, "system_workflow.png")
    plt.savefig(pipeline_img_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Generated System Pipeline Diagram  -> {pipeline_img_path}")
    print("\n" + "=" * 60)
    print(" Successfully completed analysis and visual asset generation!")
    print("=" * 60)

if __name__ == "__main__":
    main()
