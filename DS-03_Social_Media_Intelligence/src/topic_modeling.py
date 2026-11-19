
import json
import logging
from pathlib import Path

import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_data():
    """Load cleaned records for topic modelling."""

    file_path = PROCESSED_DIR / "all_cleaned_records.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Cleaned data not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    if "clean_text" not in df.columns:
        raise ValueError(
            "The dataset does not contain a clean_text column."
        )

    df = df.dropna(subset=["clean_text"]).copy()
    df["clean_text"] = df["clean_text"].astype(str).str.strip()
    df = df[df["clean_text"].str.len() > 20]

    # Remove duplicate text
    df = df.drop_duplicates(subset=["clean_text"])

    documents = df["clean_text"].tolist()

    if len(documents) < 10:
        raise ValueError(
            "Not enough text records for topic modelling."
        )

    return df.reset_index(drop=True), documents


def run_topic_modeling():
    """Discover topics using BERTopic."""

    logging.info("Loading cleaned social media data...")
    df, documents = load_data()

    logging.info("Total documents: %s", len(documents))

    logging.info("Loading sentence transformer model...")

    embedding_model = SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2"
    )

    logging.info("Generating document embeddings...")

    embeddings = embedding_model.encode(
        documents,
        show_progress_bar=True,
        batch_size=16,
        convert_to_numpy=True
    )

    logging.info("Starting BERTopic modelling...")

    topic_model = BERTopic(
        embedding_model=embedding_model,
        language="multilingual",
        min_topic_size=10,
        nr_topics="auto",
        calculate_probabilities=False,
        verbose=True
    )

    topics, probabilities = topic_model.fit_transform(
        documents,
        embeddings
    )

    # Add topic assignments to records
    df["topic_id"] = topics

    topic_info = topic_model.get_topic_info()

    # Save document-level results
    results_path = REPORTS_DIR / "topic_results.csv"
    df.to_csv(results_path, index=False)

    # Save topic information
    topic_info_path = REPORTS_DIR / "topic_summary.csv"
    topic_info.to_csv(topic_info_path, index=False)

    # Create readable topic report
    topic_report = []

    for _, row in topic_info.iterrows():
        topic_id = int(row["Topic"])

        if topic_id == -1:
            topic_name = "Outliers / Unclassified"
            keywords = []
        else:
            topic_words = topic_model.get_topic(topic_id) or []
            keywords = [
                word for word, score in topic_words[:10]
            ]
            topic_name = ", ".join(keywords[:5])

        topic_report.append({
            "topic_id": topic_id,
            "topic_name": topic_name,
            "document_count": int(row["Count"]),
            "keywords": keywords
        })

    report_path = REPORTS_DIR / "topic_modeling_report.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(
            topic_report,
            file,
            indent=4,
            ensure_ascii=False
        )

    # Print results
    print("\nTOPIC MODELLING COMPLETED")
    print("-" * 45)
    print("Documents analysed:", len(documents))
    print("Topics discovered:", len(topic_info) - 1)
    print(
        "Outlier documents:",
        int((df["topic_id"] == -1).sum())
    )

    print("\nTOP DISCOVERED TOPICS:")
    print(topic_info.head(11).to_string(index=False))

    print("\nSaved reports:")
    print(results_path)
    print(topic_info_path)
    print(report_path)


if __name__ == "__main__":
    run_topic_modeling()
