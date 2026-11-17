
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(model, X_test, y_test, model_name):
    """
    Đánh giá mô hình trên tập Test.
    """

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test, y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred,
        average="macro",
        zero_division=0
    )

    print(f"\n===== {model_name} =====")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"Macro F1 : {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Macro_F1": f1
    }, y_pred


def plot_confusion_matrix(y_test, y_pred, model_name):
    """
    Vẽ và lưu Confusion Matrix.
    """

    os.makedirs("results/figures", exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(9, 7))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=range(10),
        yticklabels=range(10)
    )

    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    filename = (
        model_name.lower()
        .replace(" ", "_")
        .replace("+", "plus")
    )

    plt.savefig(
        f"results/figures/confusion_matrix_{filename}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_model_comparison(results):
    """
    Vẽ biểu đồ so sánh 4 mô hình.
    """

    os.makedirs("results/figures", exist_ok=True)

    df = pd.DataFrame(results)

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "Macro_F1"
    ]

    ax = df.set_index("Model")[metrics].plot(
        kind="bar",
        figsize=(12, 6)
    )

    ax.set_title("Comparison of Machine Learning Models")
    ax.set_xlabel("Models")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)

    plt.xticks(rotation=15)
    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig(
        "results/figures/model_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()
