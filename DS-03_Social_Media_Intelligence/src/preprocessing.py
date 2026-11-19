
import re
import json
import html
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
from langdetect import detect, LangDetectException


# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# FIND LATEST RAW FILE
# --------------------------------------------------

def get_latest_file(pattern):
    files = list(RAW_DIR.glob(pattern))

    if not files:
        logger.warning("No file found for pattern: %s", pattern)
        return None

    return max(files, key=lambda file: file.stat().st_mtime)


# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------

def clean_text(text):
    """Clean text while preserving meaningful emojis."""

    if not isinstance(text, str):
        return ""

    text = html.unescape(text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove email addresses
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        " ",
        text
    )

    # Remove control characters
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# --------------------------------------------------
# LANGUAGE DETECTION
# --------------------------------------------------

def detect_language(text):
    """Identify the language of a text record."""

    if not text or len(text.split()) < 3:
        return "unknown"

    try:
        return detect(text)

    except LangDetectException:
        return "unknown"

    except Exception:
        return "unknown"


# --------------------------------------------------
# LOAD JSON DATA
# --------------------------------------------------

def load_json(file_path):
    if file_path is None:
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, list):
            return data

        logger.warning("Expected a list in %s", file_path.name)
        return []

    except (json.JSONDecodeError, OSError) as error:
        logger.error("Could not read %s: %s", file_path, error)
        return []


# --------------------------------------------------
# PROCESS YOUTUBE VIDEOS
# --------------------------------------------------

def process_videos(records):

    processed = []
    seen = set()

    for record in records:

        if not isinstance(record, dict):
            continue

        title = clean_text(record.get("title", ""))
        description = clean_text(record.get("description", ""))

        combined_text = clean_text(title + " " + description)

        if not combined_text:
            continue

        video_id = record.get("video_id", "")

        # Avoid duplicate video records
        unique_key = video_id or combined_text.lower()

        if unique_key in seen:
            continue

        seen.add(unique_key)

        cleaned_record = record.copy()

        cleaned_record["cleaned_title"] = title
        cleaned_record["cleaned_description"] = description
        cleaned_record["clean_text"] = combined_text
        cleaned_record["language"] = detect_language(combined_text)
        cleaned_record["text_length"] = len(combined_text)

        processed.append(cleaned_record)

    return processed


# --------------------------------------------------
# PROCESS YOUTUBE COMMENTS
# --------------------------------------------------

def find_comment_text(record):
    """Find the comment text across possible API field names."""

    possible_fields = [
        "text",
        "comment",
        "text_original",
        "content",
        "comment_text"
    ]

    for field in possible_fields:
        value = record.get(field)

        if isinstance(value, str) and value.strip():
            return value

    # Handle YouTube API's nested comment structure
    snippet = record.get("snippet", {})

    if isinstance(snippet, dict):

        top_comment = snippet.get("topLevelComment", {})

        if isinstance(top_comment, dict):
            comment_snippet = top_comment.get("snippet", {})

            if isinstance(comment_snippet, dict):
                return comment_snippet.get("textDisplay", "")

    return ""


def process_comments(records):

    processed = []
    seen = set()

    for record in records:

        if not isinstance(record, dict):
            continue

        original_text = find_comment_text(record)
        cleaned = clean_text(original_text)

        if not cleaned:
            continue

        comment_id = record.get("comment_id", "")

        unique_key = (
            comment_id
            or (
                record.get("video_id", ""),
                cleaned.lower()
            )
        )

        if unique_key in seen:
            continue

        seen.add(unique_key)

        cleaned_record = record.copy()

        cleaned_record["clean_text"] = cleaned
        cleaned_record["language"] = detect_language(cleaned)
        cleaned_record["text_length"] = len(cleaned)

        processed.append(cleaned_record)

    return processed


# --------------------------------------------------
# SAVE PROCESSED DATA
# --------------------------------------------------

def save_json(data, filename):

    output_path = PROCESSED_DIR / filename

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=4
        )

    logger.info("Saved: %s", output_path.name)


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def main():

    logger.info("Starting text preprocessing.")

    video_file = get_latest_file("youtube_videos_*.json")
    comment_file = get_latest_file("youtube_comments_*.json")

    raw_videos = load_json(video_file)
    raw_comments = load_json(comment_file)

    logger.info("Raw videos: %d", len(raw_videos))
    logger.info("Raw comments: %d", len(raw_comments))

    cleaned_videos = process_videos(raw_videos)
    cleaned_comments = process_comments(raw_comments)

    # Save separate JSON files
    save_json(cleaned_videos, "youtube_videos_cleaned.json")
    save_json(cleaned_comments, "youtube_comments_cleaned.json")

    # Combine videos and comments for analysis
    all_records = []

    for record in cleaned_videos:
        item = record.copy()
        item["record_type"] = "video"
        all_records.append(item)

    for record in cleaned_comments:
        item = record.copy()
        item["record_type"] = "comment"
        all_records.append(item)

    # Save combined CSV
    if all_records:

        dataframe = pd.DataFrame(all_records)

        dataframe.to_csv(
            PROCESSED_DIR / "all_cleaned_records.csv",
            index=False,
            encoding="utf-8-sig"
        )

    # Language distribution
    language_counts = {}

    for record in all_records:
        language = record.get("language", "unknown")
        language_counts[language] = (
            language_counts.get(language, 0) + 1
        )

    summary = {
        "processing_time": datetime.now().isoformat(),
        "raw_videos": len(raw_videos),
        "cleaned_videos": len(cleaned_videos),
        "raw_comments": len(raw_comments),
        "cleaned_comments": len(cleaned_comments),
        "total_cleaned_records": len(all_records),
        "language_distribution": language_counts
    }

    save_json(summary, "preprocessing_summary.json")

    logger.info("Preprocessing completed.")
    logger.info("Summary: %s", summary)


if __name__ == "__main__":
    main()

