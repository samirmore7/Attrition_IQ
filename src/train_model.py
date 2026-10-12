import os
import warnings

# ==================================================
# MATPLOTLIB BACKEND
# ==================================================

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# ==================================================
# LIBRARIES
# ==================================================

import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE

from imblearn.pipeline import Pipeline as ImbPipeline

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

os.makedirs(
    MODEL_DIR,
    exist_ok=True
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

    # Remove employee identifier
    df = df.drop(
        columns=["EMP_ID"]
    )

    # Convert target
    # No = 0
    # Yes = 1
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
# BUILD PREPROCESSOR
# ==================================================

def build_preprocessor(X):

    numerical_features = X.select_dtypes(
        include=[
            "int64",
            "float64"
        ]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    print("\nNumerical features:")

    for feature in numerical_features:
        print(
            f"- {feature}"
        )

    print("\nCategorical features:")

    for feature in categorical_features:
        print(
            f"- {feature}"
        )

    # --------------------------------------------------
    # Numerical pipeline
    # --------------------------------------------------

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            )
        ]
    )

    # --------------------------------------------------
    # Categorical pipeline
    # --------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    # --------------------------------------------------
    # Combined preprocessor
    # --------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    return preprocessor


# ==================================================
# EVALUATE MODEL
# ==================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    # --------------------------------------------------
    # Predictions
    # --------------------------------------------------

    y_pred = model.predict(
        X_test
    )

    y_probability = model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

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

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print("\n")
    print("=" * 60)
    print("TUNED RANDOM FOREST RESULTS")
    print("=" * 60)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    # --------------------------------------------------
    # Classification report
    # --------------------------------------------------

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "No Attrition",
                "Attrition"
            ],
            zero_division=0
        )
    )

    # --------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
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
        "Tuned Random Forest - Confusion Matrix"
    )

    plt.tight_layout()

    confusion_path = os.path.join(
        OUTPUT_DIR,
        "tuned_random_forest_confusion_matrix.png"
    )

    plt.savefig(
        confusion_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close("all")

    print(
        f"\nConfusion matrix saved:"
        f"\n{confusion_path}"
    )

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc
    }


# ==================================================
# MAIN
# ==================================================

def main():

    print("=" * 60)
    print("HR ATTRITION - TUNED RANDOM FOREST")
    print("=" * 60)

    # ==================================================
    # 1. LOAD DATA
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
    # 2. PREPARE DATA
    # ==================================================

    X, y = prepare_data(
        df
    )

    print(
        f"\nFeatures: {X.shape[1]}"
    )

    print(
        "\nTarget distribution:"
    )

    print(
        y.value_counts()
    )

    # ==================================================
    # 3. TRAIN / TEST SPLIT
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
    # 4. BUILD PREPROCESSOR
    # ==================================================

    preprocessor = build_preprocessor(
        X
    )

    # ==================================================
    # 5. BASE RANDOM FOREST
    # ==================================================

    random_forest = RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )

    # ==================================================
    # 6. COMPLETE PIPELINE
    # ==================================================

    pipeline = ImbPipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),

            (
                "smote",
                SMOTE(
                    random_state=42
                )
            ),

            (
                "model",
                random_forest
            )
        ]
    )

    # ==================================================
    # 7. HYPERPARAMETER SEARCH
    # ==================================================

    print("\n")
    print("=" * 60)
    print("STARTING RANDOM FOREST HYPERPARAMETER TUNING")
    print("=" * 60)

    param_distributions = {

        "model__n_estimators": [
            200,
            300,
            400,
            500
        ],

        "model__max_depth": [
            None,
            5,
            8,
            10,
            15,
            20
        ],

        "model__min_samples_split": [
            2,
            5,
            10,
            15
        ],

        "model__min_samples_leaf": [
            1,
            2,
            4,
            6
        ],

        "model__max_features": [
            "sqrt",
            "log2",
            None
        ],

        "model__class_weight": [
            None,
            "balanced",
            "balanced_subsample"
        ]
    }

    # ==================================================
    # 8. RANDOMIZED SEARCH
    # ==================================================

    search = RandomizedSearchCV(

        estimator=pipeline,

        param_distributions=param_distributions,

        n_iter=25,

        scoring="f1",

        cv=5,

        verbose=2,

        random_state=42,

        n_jobs=-1,

        return_train_score=True
    )

    print(
        "\nTraining and tuning model..."
    )

    search.fit(
        X_train,
        y_train
    )

    # ==================================================
    # 9. BEST PARAMETERS
    # ==================================================

    print("\n")
    print("=" * 60)
    print("BEST RANDOM FOREST PARAMETERS")
    print("=" * 60)

    print(
        search.best_params_
    )

    print(
        f"\nBest Cross-Validation F1 Score: "
        f"{search.best_score_:.4f}"
    )

    # ==================================================
    # 10. BEST MODEL
    # ==================================================

    best_model = search.best_estimator_

    print(
        "\nBest Random Forest model selected!"
    )

    # ==================================================
    # 11. TEST SET EVALUATION
    # ==================================================

    results = evaluate_model(
        best_model,
        X_test,
        y_test
    )

    # ==================================================
    # 12. SAVE MODEL
    # ==================================================

    model_path = os.path.join(
        MODEL_DIR,
        "attrition_model.pkl"
    )

    joblib.dump(
        best_model,
        model_path
    )

    print(
        f"\nTuned model saved successfully:"
        f"\n{model_path}"
    )

    # ==================================================
    # 13. SAVE METRICS
    # ==================================================

    metrics_df = pd.DataFrame(
        [results]
    )

    metrics_path = os.path.join(
        OUTPUT_DIR,
        "tuned_random_forest_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False
    )

    print(
        f"\nTuned model metrics saved:"
        f"\n{metrics_path}"
    )

    # ==================================================
    # 14. SAVE BEST PARAMETERS
    # ==================================================

    best_params_df = pd.DataFrame(
        [
            search.best_params_
        ]
    )

    best_params_path = os.path.join(
        OUTPUT_DIR,
        "random_forest_best_parameters.csv"
    )

    best_params_df.to_csv(
        best_params_path,
        index=False
    )

    print(
        f"\nBest parameters saved:"
        f"\n{best_params_path}"
    )

    # ==================================================
    # 15. BASELINE COMPARISON
    # ==================================================

    baseline_metrics = {
        "Accuracy": 0.6862,
        "Precision": 0.4000,
        "Recall": 0.1691,
        "F1": 0.2377,
        "ROC-AUC": 0.6418
    }

    comparison = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "ROC-AUC"
            ],

            "Baseline Random Forest": [
                baseline_metrics["Accuracy"],
                baseline_metrics["Precision"],
                baseline_metrics["Recall"],
                baseline_metrics["F1"],
                baseline_metrics["ROC-AUC"]
            ],

            "Tuned Random Forest": [
                results["Accuracy"],
                results["Precision"],
                results["Recall"],
                results["F1"],
                results["ROC-AUC"]
            ]
        }
    )

    comparison["Improvement"] = (
        comparison["Tuned Random Forest"]
        -
        comparison["Baseline Random Forest"]
    )

    comparison_path = os.path.join(
        OUTPUT_DIR,
        "random_forest_model_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False
    )

    print(
        f"\nModel comparison saved:"
        f"\n{comparison_path}"
    )

    # ==================================================
    # 16. PRINT COMPARISON
    # ==================================================

    print("\n")
    print("=" * 60)
    print("BASELINE VS TUNED RANDOM FOREST")
    print("=" * 60)

    print(
        comparison.to_string(
            index=False
        )
    )

    # ==================================================
    # 17. COMPLETION
    # ==================================================

    print("\n")
    print("=" * 60)
    print("RANDOM FOREST TUNING COMPLETED")
    print("=" * 60)


# ==================================================
# RUN PROGRAM
# ==================================================

if __name__ == "__main__":

    main()