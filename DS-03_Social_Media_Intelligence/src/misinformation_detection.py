
import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.pipeline import Pipeline

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw" / "liar"
REPORT_DIR = BASE_DIR / "reports"
MODEL_DIR = BASE_DIR / "models"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

# LIAR TSV column names
COLUMNS = [
    "id",
    "label",
    "statement",
    "subject",
    "speaker",
    "speaker_job",
    "state",
    "party",
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_fire_counts",
    "context"
]

# Binary label grouping
# 0 = Lower truthfulness labels
# 1 = Higher truthfulness labels
LABEL_MAP = {
    "pants-fire": 0,
    "false": 0,
    "barely-true": 0,
    "half-true": 1,
    "mostly-true": 1,
    "true": 1
}


def load_split(filename):
    file_path = DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    df = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=COLUMNS,
        quoting=3,
        on_bad_lines="skip"
    )

    df["statement"] = df["statement"].fillna("").astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()

    df["target"] = df["label"].map(LABEL_MAP)

    df = df.dropna(subset=["target"])
    df = df[df["statement"].str.len() > 0]

    df["target"] = df["target"].astype(int)

    return df


def main():
    logging.info("Loading LIAR dataset...")

    train_df = load_split("train.tsv")
    valid_df = load_split("valid.tsv")
    test_df = load_split("test.tsv")

    print("\nDATASET SUMMARY")
    print("--------------------------")
    print("Training records:", len(train_df))
    print("Validation records:", len(valid_df))
    print("Testing records:", len(test_df))

    print("\nTraining label distribution:")
    print(train_df["target"].value_counts().sort_index())

    # Prepare text and labels
    X_train = train_df["statement"]
    y_train = train_df["target"]

    X_valid = valid_df["statement"]
    y_valid = valid_df["target"]

    X_test = test_df["statement"]
    y_test = test_df["target"]

    logging.info("Training TF-IDF + Logistic Regression model...")

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=20000,
                ngram_range=(1, 2),
                stop_words="english",
                min_df=2,
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ])

    model.fit(X_train, y_train)

    logging.info("Evaluating validation data...")

    valid_predictions = model.predict(X_valid)

    print("\nVALIDATION RESULTS")
    print("--------------------------")
    print("Accuracy:", round(accuracy_score(y_valid, valid_predictions), 4))
    print("F1-score:", round(f1_score(y_valid, valid_predictions, zero_division=0), 4))

    logging.info("Evaluating test data...")

    test_predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, test_predictions)
    precision = precision_score(y_test, test_predictions, zero_division=0)
    recall = recall_score(y_test, test_predictions, zero_division=0)
    f1 = f1_score(y_test, test_predictions, zero_division=0)

    print("\nFINAL TEST RESULTS")
    print("--------------------------")
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            test_predictions,
            target_names=["Lower truthfulness", "Higher truthfulness"],
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, test_predictions))

    # Save trained model
    model_path = MODEL_DIR / "misinformation_model.joblib"
    joblib.dump(model, model_path)

    # Save predictions
    predictions_df = test_df[["id", "label", "statement", "target"]].copy()
    predictions_df["predicted_target"] = test_predictions
    predictions_df.to_csv(
        REPORT_DIR / "misinformation_predictions.csv",
        index=False
    )

    # Save evaluation metrics
    evaluation = {
        "dataset": "LIAR",
        "training_records": len(train_df),
        "validation_records": len(valid_df),
        "test_records": len(test_df),
        "label_mapping": {
            "0": ["pants-fire", "false", "barely-true"],
            "1": ["half-true", "mostly-true", "true"]
        },
        "test_metrics": {
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4)
        },
        "confusion_matrix": confusion_matrix(
            y_test, test_predictions
        ).tolist()
    }

    with open(
        REPORT_DIR / "misinformation_evaluation.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(evaluation, file, indent=4)

    print("\nMISINFORMATION DETECTION COMPLETED")
    print("---------------------------------")
    print("Model saved:", model_path)
    print("Predictions saved: reports/misinformation_predictions.csv")
    print("Evaluation saved: reports/misinformation_evaluation.json")


if __name__ == "__main__":
    main()
