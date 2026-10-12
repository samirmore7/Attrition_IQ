import os
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from connect_snowflake import get_connection


warnings.filterwarnings("ignore")


# ==================================================
# CONFIGURATION
# ==================================================

TABLE_NAME = "HR_ATTRITION.RAW.HR_ATTRITION_FINAL"

BASE_DIR = os.path.dirname(
    os.path.dirname(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "attrition_model.pkl"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==================================================
# LOAD DATA FROM SNOWFLAKE
# ==================================================

def load_data():

    conn = None
    cursor = None

    try:

        conn = get_connection()

        cursor = conn.cursor()

        query = f"""
            SELECT *
            FROM {TABLE_NAME}
        """

        cursor.execute(query)

        data = cursor.fetchall()

        columns = [
            column[0]
            for column in cursor.description
        ]

        df = pd.DataFrame(
            data,
            columns=columns
        )

        return df

    finally:

        if cursor:
            cursor.close()

        if conn:
            conn.close()


# ==================================================
# PREPARE DATA
# ==================================================

def prepare_data(df):

    df = df.copy()

    # Remove employee ID
    df = df.drop(
        columns=["EMP_ID"]
    )

    # Convert target
    df["ATTRITION"] = (
        df["ATTRITION"]
        .map({
            "No": 0,
            "Yes": 1
        })
    )

    X = df.drop(
        columns=["ATTRITION"]
    )

    y = df["ATTRITION"]

    return X, y


# ==================================================
# CALCULATE THRESHOLD METRICS
# ==================================================

def calculate_threshold_metrics(
    y_test,
    probabilities,
    threshold
):

    # Convert probabilities into predictions
    y_pred = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    return {
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }


# ==================================================
# MAIN
# ==================================================

def main():

    print("=" * 60)
    print("HR ATTRITION - THRESHOLD OPTIMIZATION")
    print("=" * 60)

    # ==================================================
    # 1. CHECK MODEL
    # ==================================================

    if not os.path.exists(MODEL_PATH):

        print(
            "\nERROR: Trained model not found."
        )

        print(
            f"\nExpected location:\n{MODEL_PATH}"
        )

        return

    # ==================================================
    # 2. LOAD MODEL
    # ==================================================

    print(
        "\nLoading tuned Random Forest model..."
    )

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "Model loaded successfully!"
    )

    # ==================================================
    # 3. LOAD DATA
    # ==================================================

    print(
        "\nLoading data from Snowflake..."
    )

    df = load_data()

    print(
        f"Data loaded successfully: "
        f"{df.shape[0]} rows, "
        f"{df.shape[1]} columns"
    )

    # ==================================================
    # 4. PREPARE DATA
    # ==================================================

    X, y = prepare_data(
        df
    )

    # ==================================================
    # 5. SAME TEST SPLIT
    # ==================================================

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    print(
        f"\nTraining records: "
        f"{len(X_train)}"
    )

    print(
        f"Testing records : "
        f"{len(X_test)}"
    )

    # ==================================================
    # 6. GET PREDICTED PROBABILITIES
    # ==================================================

    print(
        "\nGenerating attrition probabilities..."
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # ==================================================
    # 7. ROC-AUC
    # ==================================================

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    print(
        f"\nROC-AUC: {roc_auc:.4f}"
    )

    # ==================================================
    # 8. TEST DIFFERENT THRESHOLDS
    # ==================================================

    thresholds = [
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60
    ]

    results = []

    print("\n")
    print("=" * 60)
    print("THRESHOLD RESULTS")
    print("=" * 60)

    for threshold in thresholds:

        metrics = calculate_threshold_metrics(
            y_test,
            probabilities,
            threshold
        )

        results.append(
            metrics
        )

        print(
            f"\nThreshold: {threshold:.2f}"
        )

        print(
            f"Accuracy : "
            f"{metrics['Accuracy']:.4f}"
        )

        print(
            f"Precision: "
            f"{metrics['Precision']:.4f}"
        )

        print(
            f"Recall   : "
            f"{metrics['Recall']:.4f}"
        )

        print(
            f"F1 Score : "
            f"{metrics['F1']:.4f}"
        )

    # ==================================================
    # 9. CREATE RESULTS DATAFRAME
    # ==================================================

    results_df = pd.DataFrame(
        results
    )

    # ==================================================
    # 10. FIND BEST F1 THRESHOLD
    # ==================================================

    best_index = (
        results_df["F1"]
        .idxmax()
    )

    best_threshold = (
        results_df
        .loc[
            best_index,
            "Threshold"
        ]
    )

    best_accuracy = (
        results_df
        .loc[
            best_index,
            "Accuracy"
        ]
    )

    best_precision = (
        results_df
        .loc[
            best_index,
            "Precision"
        ]
    )

    best_recall = (
        results_df
        .loc[
            best_index,
            "Recall"
        ]
    )

    best_f1 = (
        results_df
        .loc[
            best_index,
            "F1"
        ]
    )

    # ==================================================
    # 11. PRINT BEST THRESHOLD
    # ==================================================

    print("\n")
    print("=" * 60)
    print("BEST THRESHOLD")
    print("=" * 60)

    print(
        f"Threshold : "
        f"{best_threshold:.2f}"
    )

    print(
        f"Accuracy  : "
        f"{best_accuracy:.4f}"
    )

    print(
        f"Precision : "
        f"{best_precision:.4f}"
    )

    print(
        f"Recall    : "
        f"{best_recall:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{best_f1:.4f}"
    )

    # ==================================================
    # 12. SAVE THRESHOLD RESULTS
    # ==================================================

    threshold_results_path = os.path.join(
        OUTPUT_DIR,
        "threshold_optimization_results.csv"
    )

    results_df.to_csv(
        threshold_results_path,
        index=False
    )

    print(
        f"\nThreshold results saved:"
        f"\n{threshold_results_path}"
    )

    # ==================================================
    # 13. SAVE FINAL THRESHOLD
    # ==================================================

    threshold_config = pd.DataFrame(
        [
            {
                "Final_Threshold": best_threshold,
                "Accuracy": best_accuracy,
                "Precision": best_precision,
                "Recall": best_recall,
                "F1": best_f1,
                "ROC-AUC": roc_auc
            }
        ]
    )

    threshold_config_path = os.path.join(
        OUTPUT_DIR,
        "final_threshold_config.csv"
    )

    threshold_config.to_csv(
        threshold_config_path,
        index=False
    )

    print(
        f"\nFinal threshold configuration saved:"
        f"\n{threshold_config_path}"
    )

    # ==================================================
    # 14. FINAL CONFUSION MATRIX
    # ==================================================

    final_predictions = (
        probabilities >= best_threshold
    ).astype(int)

    cm = confusion_matrix(
        y_test,
        final_predictions
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "No Attrition",
            "Attrition"
        ]
    )

    display.plot()

    plt.title(
        f"Final Random Forest - "
        f"Threshold {best_threshold:.2f}"
    )

    plt.tight_layout()

    confusion_path = os.path.join(
        OUTPUT_DIR,
        "final_random_forest_confusion_matrix.png"
    )

    plt.savefig(
        confusion_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close("all")

    print(
        f"\nFinal confusion matrix saved:"
        f"\n{confusion_path}"
    )

    # ==================================================
    # 15. THRESHOLD VS F1 GRAPH
    # ==================================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        results_df["Threshold"],
        results_df["F1"],
        marker="o"
    )

    plt.axvline(
        best_threshold,
        linestyle="--"
    )

    plt.xlabel(
        "Prediction Threshold"
    )

    plt.ylabel(
        "F1 Score"
    )

    plt.title(
        "F1 Score vs Prediction Threshold"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    threshold_plot_path = os.path.join(
        OUTPUT_DIR,
        "threshold_vs_f1.png"
    )

    plt.savefig(
        threshold_plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close("all")

    print(
        f"\nThreshold graph saved:"
        f"\n{threshold_plot_path}"
    )

    # ==================================================
    # 16. COMPLETION
    # ==================================================

    print("\n")
    print("=" * 60)
    print("THRESHOLD OPTIMIZATION COMPLETED")
    print("=" * 60)

    print(
        f"\nFINAL THRESHOLD: "
        f"{best_threshold:.2f}"
    )

    print(
        "\nNext step:"
    )

    print(
        "Use this threshold in the Streamlit "
        "prediction dashboard."
    )


# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":

    main()