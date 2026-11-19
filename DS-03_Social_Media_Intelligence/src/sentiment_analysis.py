
import json
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import torch
from transformers import pipeline


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = PROCESSED_DIR / "all_cleaned_records.csv"

# Multilingual sentiment model
MODEL_NAME = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"

DEVICE = 0 if torch.cuda.is_available() else -1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

def load_sentiment_model():

    logger.info("Loading multilingual sentiment model...")

    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        device=DEVICE
    )

    logger.info("Sentiment model loaded successfully.")

    return sentiment_pipeline


# --------------------------------------------------
# SENTIMENT PREDICTION
# --------------------------------------------------

def predict_sentiment(model, texts):

    predictions = []

    for text in texts:

        if not isinstance(text, str) or not text.strip():
            predictions.append({
                "sentiment": "unknown",
                "confidence": 0.0
            })
            continue

        try:

            # Keep input within model token limit
            result = model(
                text[:2000],
                truncation=True,
                max_length=512
            )[0]

            predictions.append({
                "sentiment": result["label"].lower(),
                "confidence": round(float(result["score"]), 4)
            })

        except Exception as error:

            logger.warning("Prediction failed: %s", error)

            predictions.append({
                "sentiment": "unknown",
                "confidence": 0.0
            })

    return predictions


# --------------------------------------------------
# LOAD CLEANED DATA
# --------------------------------------------------

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Cleaned CSV not found. Run preprocessing first."
        )

    dataframe = pd.read_csv(INPUT_FILE)

    if "clean_text" not in dataframe.columns:
        raise ValueError("clean_text column is missing.")

    dataframe["clean_text"] = (
        dataframe["clean_text"].fillna("").astype(str)
    )

    return dataframe


# --------------------------------------------------
# ANALYZE DATA IN BATCHES
# --------------------------------------------------

def analyze_sentiment(dataframe, model):

    results = []

    texts = dataframe["clean_text"].tolist()

    batch_size = 16

    logger.info("Analyzing %d records...", len(texts))

    for start in range(0, len(texts), batch_size):

        batch = texts[start:start + batch_size]

        try:

            batch_results = model(
                batch,
                truncation=True,
                max_length=512,
                batch_size=batch_size
            )

            for result in batch_results:

                results.append({
                    "sentiment": result["label"].lower(),
                    "confidence": round(
                        float(result["score"]), 4
                    )
                })

        except Exception as error:

            logger.warning(
                "Batch failed at record %d: %s",
                start,
                error
            )

            # Retry records individually
            results.extend(
                predict_sentiment(model, batch)
            )

        completed = min(start + batch_size, len(texts))

        logger.info(
            "Processed %d/%d records",
            completed,
            len(texts)
        )

    return results


# --------------------------------------------------
# GENERATE SUMMARY
# --------------------------------------------------

def generate_summary(dataframe):

    sentiment_counts = (
        dataframe["sentiment"]
        .value_counts()
        .to_dict()
    )

    language_summary = (
        dataframe.groupby(
            ["language", "sentiment"],
            dropna=False
        )
        .size()
        .unstack(fill_value=0)
        .to_dict(orient="index")
    )

    summary = {
        "analysis_time": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "total_records": len(dataframe),
        "sentiment_distribution": sentiment_counts,
        "language_wise_sentiment": language_summary,
        "average_confidence": round(
            float(dataframe["confidence"].mean()), 4
        )
    }

    return summary


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

def save_results(dataframe, summary):

    csv_path = REPORTS_DIR / "sentiment_results.csv"

    json_path = REPORTS_DIR / "sentiment_results.json"

    summary_path = REPORTS_DIR / "sentiment_summary.json"

    dataframe.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    dataframe.to_json(
        json_path,
        orient="records",
        force_ascii=False,
        indent=4
    )

    with open(summary_path, "w", encoding="utf-8") as file:

        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=4
        )

    logger.info("Saved sentiment results and summary.")


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def main():

    logger.info("Starting sentiment analysis.")

    dataframe = load_data()

    model = load_sentiment_model()

    predictions = analyze_sentiment(dataframe, model)

    dataframe["sentiment"] = [
        result["sentiment"] for result in predictions
    ]

    dataframe["confidence"] = [
        result["confidence"] for result in predictions
    ]

    summary = generate_summary(dataframe)

    save_results(dataframe, summary)

    logger.info("Sentiment analysis completed.")

    logger.info(
        "Sentiment distribution: %s",
        summary["sentiment_distribution"]
    )


if __name__ == "__main__":
    main()
