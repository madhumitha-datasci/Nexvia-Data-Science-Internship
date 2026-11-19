
# DS-03 Social Media Intelligence Platform
## System Architecture

```mermaid
flowchart TD
    A["YouTube Data API"] --> B["Data Collection Module"]
    B --> C["Raw JSON Data"]
    C --> D["Text Preprocessing"]
    D --> E["Cleaned Dataset"]

    E --> F["Multilingual Sentiment Analysis"]
    E --> G["BERTopic Topic Modelling"]
    E --> H["NetworkX Influencer Analysis"]
    E --> I["Elasticsearch Indexing"]

    J["LIAR Dataset"] --> K["TF-IDF + Logistic Regression"]
    K --> L["Misinformation Classification Model"]

    H --> M["Neo4j Graph Database"]
    I --> N["Elasticsearch"]
    N --> O["Kibana Dashboards"]

    F --> P["Analysis Reports"]
    G --> P
    H --> P
    K --> P

    E --> Q["Streamlit Dashboard"]
    P --> Q
    L --> Q
    N --> Q

    Q --> R["Interactive Social Media Intelligence Platform"]
```

## Main Components

1. **Data Collection:** Collects YouTube videos and comments.
2. **Preprocessing:** Cleans text, detects languages and removes duplicates.
3. **Sentiment Analysis:** Performs multilingual sentiment classification.
4. **Topic Modelling:** Discovers topics using BERTopic.
5. **Network Analysis:** Identifies influential channels using PageRank and community detection.
6. **Misinformation Classification:** Uses TF-IDF and Logistic Regression trained on LIAR.
7. **Neo4j:** Stores the influencer interaction graph.
8. **Elasticsearch:** Indexes cleaned social media records.
9. **Kibana:** Provides search and visualization dashboards.
10. **Streamlit:** Integrates project results into one interactive dashboard.

## Data Collection Limitation

Reddit API credentials were unavailable during implementation. Therefore, the current prototype uses collected YouTube data rather than data from both platforms.
