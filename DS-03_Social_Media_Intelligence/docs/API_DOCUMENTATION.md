
# DS-03 Social Media Intelligence Platform
## API and Technical Documentation

### 1. Project Overview

The Social Media Intelligence Platform analyses collected social media content using Natural Language Processing, machine learning, graph analytics and search technologies.

Current implementation uses a sample of YouTube videos and comments.

### 2. External API Integration

#### YouTube Data API v3

Purpose:
- Search for videos.
- Retrieve video metadata.
- Retrieve available comments.

API operations:
- `search.list` – Search videos by keywords.
- `videos.list` – Retrieve video details and statistics.
- `commentThreads.list` – Retrieve available video comments.

Configuration:
- Environment variable: `YOUTUBE_API_KEY`
- Python library: `google-api-python-client`

The API key is stored in the local `.env` file and must not be committed to GitHub.

Limitation: Some videos have comments disabled. Those comments cannot be collected.

#### Reddit API

Purpose:
- Planned collection of Reddit posts and comments.

Library: `praw`

Status: Not configured because Reddit API credentials were unavailable.

### 3. Internal Python Modules

| Module | Responsibility |
|---|---|
| config.py | Loads environment variables |
| data_collection.py | Collects YouTube and Reddit data |
| preprocessing.py | Cleans text and detects language |
| sentiment_analysis.py | Performs multilingual sentiment analysis |
| topic_modeling.py | Discovers topics using BERTopic |
| influencer_network.py | Builds interaction graph and calculates PageRank |
| misinformation_detection.py | Trains and evaluates truthfulness classifier |
| neo4j_integration.py | Imports graph into Neo4j |
| elasticsearch_integration.py | Indexes cleaned records |

### 4. Machine Learning Models

#### Sentiment Analysis
Model: `lxyuan/distilbert-base-multilingual-cased-sentiments-student`

Output:
- Positive
- Negative
- Neutral

Dataset processed: 921 records.

Average model confidence: 57.24%.

No labelled ground-truth dataset was available for formal accuracy evaluation.

#### Topic Modelling
Algorithm: BERTopic

Documents analysed: 772.

Topics discovered: 3, excluding the outlier category.

Identified themes:
- Business and stock-market content
- Artificial Intelligence
- Climate Change

#### Truthfulness Classification
Dataset: LIAR

Features: TF-IDF text features.

Algorithm: Logistic Regression.

Test results:
- Accuracy: 60.09%
- Precision: 65.21%
- Recall: 63.41%
- F1-score: 64.30%

This is an experimental binary classification baseline, not an authoritative fact-checking system.

### 5. Graph Database

Database: Neo4j

Node label: `Channel`

Relationship: `COMMENTED_ON`

Imported:
- Nodes: 855
- Relationships: 582

Graph analysis uses NetworkX PageRank and community detection.

### 6. Search and Analytics

Technology: Elasticsearch 9.5.6

Index name: `ds03-social-media`

Indexed records: 921.

Kibana is used for:
- Language distribution
- Video and comment distribution
- Records by search topic
- Content publication timeline

### 7. Dashboard

Framework: Streamlit

Run command:

`python -m streamlit run dashboard/app.py`

Dashboard features:
- Dataset overview
- Sentiment distribution
- Topic modelling results
- Influencer rankings
- Truthfulness classification
- Elasticsearch text search

### 8. Environment Configuration

Required environment variables:

- `YOUTUBE_API_KEY`
- `REDDIT_CLIENT_ID`
- `REDDIT_CLIENT_SECRET`
- `REDDIT_USER_AGENT`
- `NEO4J_URI`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`

Keep credentials in `.env`. Never publish actual credentials.

### 9. Limitations

1. Data collection currently uses YouTube only.
2. The collected dataset is a sample, not a large-scale real-time crawl.
3. Sentiment predictions have not been validated against labelled ground truth.
4. Topic keywords contain some noise.
5. Influencer rankings represent sampled comment interactions, not overall channel popularity.
6. Truthfulness classification has moderate predictive performance and should not be used as a standalone fact-checker.
7. Language detection uses `langdetect`; Spark NLP has not been integrated.

### 10. Future Enhancements

- Configure Reddit API integration.
- Expand data collection.
- Improve multilingual text processing.
- Evaluate sentiment using labelled data.
- Improve topic quality.
- Add real-time monitoring and alerts.
- Improve truthfulness classification with additional datasets.
- Deploy the dashboard for public access.
