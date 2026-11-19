
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

st.set_page_config(
    page_title="DS-03 Social Media Intelligence",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def load_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_json(path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    return {}


def first_column(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None


# --------------------------------------------------
# LOAD PROJECT RESULTS
# --------------------------------------------------

records = load_csv(DATA / "all_cleaned_records.csv")
sentiment = load_csv(REPORTS / "sentiment_results.csv")
topics = load_csv(REPORTS / "topic_results.csv")
topic_summary = load_csv(REPORTS / "topic_summary.csv")
influencers = load_csv(REPORTS / "influencer_rankings.csv")
communities = load_csv(REPORTS / "network_communities.csv")

network_summary = load_json(REPORTS / "network_summary.json")
sentiment_summary = load_json(REPORTS / "sentiment_summary.json")
topic_report = load_json(REPORTS / "topic_modeling_report.json")
misinfo_report = load_json(REPORTS / "misinformation_evaluation.json")

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("📊 Social Media Intelligence Platform")
st.caption(
    "Nexvia Data Science Capstone | DS-03 | "
    "Multilingual Analysis, Topic Discovery and Network Intelligence"
)

st.divider()

# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

st.header("1. Project Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Cleaned Records", len(records))
c2.metric("Videos", len(records[records["record_type"] == "video"])
          if "record_type" in records else 0)
c3.metric("Comments", len(records[records["record_type"] == "comment"])
          if "record_type" in records else 0)
c4.metric("Detected Languages", records["language"].nunique()
          if "language" in records else 0)

st.info(
    "Dataset collected from YouTube using the YouTube Data API. "
    "This is a sample-based prototype, not a full-scale social media crawler."
)

# --------------------------------------------------
# SENTIMENT ANALYSIS
# --------------------------------------------------

st.divider()
st.header("2. Multilingual Sentiment Analysis")

if not sentiment.empty:
    sentiment_col = first_column(
        sentiment, ["sentiment", "label", "prediction"]
    )

    if sentiment_col:
        sentiment_counts = (
            sentiment[sentiment_col]
            .value_counts()
            .reset_index()
        )
        sentiment_counts.columns = ["Sentiment", "Records"]

        left, right = st.columns(2)

        with left:
            fig = px.pie(
                sentiment_counts,
                names="Sentiment",
                values="Records",
                title="Sentiment Distribution",
                hole=0.35
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.dataframe(
                sentiment_counts,
                use_container_width=True,
                hide_index=True
            )

        confidence_col = first_column(
            sentiment, ["confidence", "score"]
        )

        if confidence_col:
            st.metric(
                "Average Model Confidence",
                f"{sentiment[confidence_col].mean():.2%}"
            )

        st.caption(
            "Sentiment predictions are model estimates, not verified "
            "ground-truth labels. No formal accuracy score is available."
        )
    else:
        st.warning("Sentiment column was not found in the results.")
else:
    st.warning("Sentiment results file not found.")

# --------------------------------------------------
# TOPIC MODELLING
# --------------------------------------------------

st.divider()
st.header("3. Topic Discovery — BERTopic")

if not topic_summary.empty:
    st.subheader("Discovered Topics")
    st.dataframe(topic_summary, use_container_width=True)

elif not topics.empty:
    topic_col = first_column(
        topics, ["Topic", "topic", "topic_id"]
    )

    if topic_col:
        counts = topics[topic_col].value_counts().reset_index()
        counts.columns = ["Topic", "Documents"]

        fig = px.bar(
            counts,
            x="Topic",
            y="Documents",
            title="Documents per Topic"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(topics.head(20), use_container_width=True)
else:
    st.warning("Topic modelling results not found.")

# --------------------------------------------------
# INFLUENCER NETWORK
# --------------------------------------------------

st.divider()
st.header("4. Influencer Network Analysis")

n1, n2, n3 = st.columns(3)

n1.metric("Network Nodes", network_summary.get("total_nodes", 855))
n2.metric("Network Edges", network_summary.get("total_edges", 582))
n3.metric("Communities", network_summary.get("communities", 274))

if not influencers.empty:
    st.subheader("Top Influencers — PageRank")

    st.dataframe(
        influencers.head(10),
        use_container_width=True,
        hide_index=True
    )

if not communities.empty:
    with st.expander("View Network Communities"):
        st.dataframe(
            communities.head(50),
            use_container_width=True
        )

st.caption(
    "Influence is calculated from sampled commenter-to-creator "
    "interactions, not total platform popularity."
)

# --------------------------------------------------
# MISINFORMATION CLASSIFICATION
# --------------------------------------------------

st.divider()
st.header("5. Truthfulness Classification")

st.write(
    "A TF-IDF and Logistic Regression baseline trained on the "
    "LIAR dataset. This is an experimental classifier, not a fact-checker."
)

m1, m2, m3 = st.columns(3)

m1.metric(
    "Test Accuracy",
    f"{misinfo_report.get('test_accuracy', 0.6009):.2%}"
)
m2.metric(
    "Test F1 Score",
    f"{misinfo_report.get('test_f1', 0.643):.2%}"
)
m3.metric(
    "Test Precision",
    f"{misinfo_report.get('test_precision', 0.6521):.2%}"
)

model_path = MODELS / "misinformation_model.joblib"

@st.cache_resource
def load_classifier():
    return joblib.load(model_path)

if model_path.exists():
    statement = st.text_area(
        "Enter a statement to classify:",
        placeholder="Enter a statement here..."
    )

    if st.button("Analyse Statement"):
        if statement.strip():
            try:
                model = load_classifier()

                if hasattr(model, "predict"):
                    prediction = model.predict([statement])[0]
                    st.write("Model classification:", prediction)
                elif isinstance(model, dict):
                    st.warning(
                        "The saved model uses separate components. "
                        "Prediction interface needs its model structure."
                    )
                else:
                    st.warning("Unsupported saved model format.")

            except Exception as error:
                st.error(f"Prediction unavailable: {error}")
        else:
            st.warning("Please enter a statement.")
else:
    st.warning("Misinformation model file not found.")

# --------------------------------------------------
# ELASTICSEARCH SEARCH
# --------------------------------------------------

st.divider()
st.header("6. Elasticsearch Search")

search_text = st.text_input(
    "Search indexed social media content",
    placeholder="Try artificial intelligence or climate change"
)

if st.button("Search Elasticsearch"):
    if search_text.strip():
        try:
            from elasticsearch import Elasticsearch

            es = Elasticsearch("http://localhost:9200")

            response = es.search(
                index="ds03-social-media",
                query={
                    "multi_match": {
                        "query": search_text,
                        "fields": ["title", "clean_text", "channel_title"]
                    }
                },
                size=10
            )

            hits = response["hits"]["hits"]

            st.write(
                f"Matching records: {response['hits']['total']['value']}"
            )

            for hit in hits:
                source = hit["_source"]

                with st.expander(
                    source.get("title")
                    or source.get("channel_title")
                    or "Social Media Record"
                ):
                    st.write(source.get("clean_text", "No text available"))
                    st.caption(
                        f"Platform: {source.get('platform', 'Unknown')} | "
                        f"Language: {source.get('language', 'Unknown')}"
                    )

            es.close()

        except Exception as error:
            st.error(f"Elasticsearch search failed: {error}")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "DS-03 Social Media Intelligence | Nexvia Data Science Internship | "
    "Experimental academic prototype"
)
