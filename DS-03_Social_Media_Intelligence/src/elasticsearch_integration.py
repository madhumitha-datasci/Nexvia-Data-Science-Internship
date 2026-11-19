
import json
from pathlib import Path

from elasticsearch import Elasticsearch, helpers

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "all_cleaned_records.csv"

ES_URL = "http://localhost:9200"
INDEX_NAME = "ds03-social-media"

def connect_elasticsearch():
    client = Elasticsearch(ES_URL)

    if not client.ping():
        raise ConnectionError("Could not connect to Elasticsearch.")

    print("Connected to Elasticsearch successfully.")
    return client


def create_index(client):
    if client.indices.exists(index=INDEX_NAME):
        print("Index already exists.")
        return

    mapping = {
        "mappings": {
            "properties": {
                "platform": {"type": "keyword"},
                "record_type": {"type": "keyword"},
                "video_id": {"type": "keyword"},
                "channel_id": {"type": "keyword"},
                "channel_title": {"type": "text"},
                "author": {"type": "keyword"},
                "author_channel_id": {"type": "keyword"},
                "title": {"type": "text"},
                "clean_text": {"type": "text"},
                "language": {"type": "keyword"},
                "search_query": {"type": "keyword"},
                "published_at": {"type": "date", "ignore_malformed": True},
                "sentiment": {"type": "keyword"},
                "confidence": {"type": "float"}
            }
        }
    }

    client.indices.create(index=INDEX_NAME, **mapping)
    print("Elasticsearch index created.")


def index_records(client):
    import pandas as pd

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")

    dataframe = pd.read_csv(DATA_FILE)
    dataframe = dataframe.fillna("")

    actions = []

    for record in dataframe.to_dict(orient="records"):
        clean_record = {}

        for key, value in record.items():
            if hasattr(value, "item"):
                value = value.item()

            if isinstance(value, float) and value != value:
                value = ""

            clean_record[key] = value

        actions.append({
            "_index": INDEX_NAME,
            "_source": clean_record
        })

    success, errors = helpers.bulk(
        client,
        actions,
        raise_on_error=False
    )

    print("Records indexed:", success)
    print("Indexing errors:", len(errors))


def verify_index(client):
    client.indices.refresh(index=INDEX_NAME)

    result = client.count(index=INDEX_NAME)

    print("Total indexed records:", result["count"])


def main():
    client = connect_elasticsearch()

    try:
        create_index(client)
        index_records(client)
        verify_index(client)

        print("\nELASTICSEARCH INTEGRATION COMPLETED")

    finally:
        client.close()


if __name__ == "__main__":
    main()
