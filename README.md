
# Nexvia Data Science Internship Projects

This repository contains the projects I worked on as part of my Data Science Internship at Nexvia.

Through these projects, I got the opportunity to apply concepts I learned in Data Science, statistics, machine learning, and Python to practical problems. I also explored how to build applications and make analytical results easier to understand.

## Projects

### DS-04 – Experimentation and Causal Inference Platform

**Project Overview**

This project focuses on statistical experimentation and causal inference. The idea was to build a platform where users can upload datasets, perform different statistical analyses, and understand the results through a simple interface.

Instead of running every analysis separately through Python scripts, the platform brings multiple experimentation and causal analysis methods together in one place.

**Key Features**
- Sample size calculation
- A/B testing and hypothesis testing
- Multiple hypothesis testing corrections
- A/B test simulation
- Sequential testing
- Bayesian A/B testing
- Propensity Score Matching
- Difference-in-Differences analysis
- Synthetic Control analysis
- DoWhy and EconML-based causal analysis
- Dataset upload and analysis

**Technologies Used**
- Python
- FastAPI
- React.js
- Vite
- Pandas and NumPy
- SciPy and Statsmodels
- Scikit-learn
- DoWhy and EconML
- Render (Backend Deployment)
- Vercel (Frontend Deployment)

**Live Application:** https://nexvia-ds04-frontend.vercel.app/

**Backend API:** https://nexvia-ds04-api.onrender.com

**API Documentation:** https://nexvia-ds04-api.onrender.com/docs

---

### DS-03 – Second Project

## Project Details

**Project Name:** Large-Scale Social Media Intelligence Platform  
**Project ID:** DS-03  
**Track:** Data Science  
**Organization:** Nexvia  

**Objective:** To develop a social media intelligence platform that analyses collected social media content using Natural Language Processing, Machine Learning, Graph Analytics and Search Technologies.

The platform processes YouTube videos and comments to identify sentiment, discover trending topics, analyse interaction networks and perform experimental truthfulness classification.

### Key Technologies
- Python, Pandas and NumPy
- Transformers and multilingual DistilBERT
- BERTopic for topic discovery
- Scikit-learn for machine learning
- NetworkX for influencer analysis
- Neo4j for graph storage
- Elasticsearch and Kibana for search and visualization
- Streamlit for the interactive dashboard
- Docker for Elasticsearch and Kibana services

## Implementation

1. **Data Collection:** Integrated YouTube Data API v3 to collect video metadata and available comments. Collected 300 videos and 632 comments.

2. **Text Preprocessing:** Cleaned text, removed duplicates, handled URLs and HTML content, and detected languages. Generated 921 cleaned records.

3. **Sentiment Analysis:** Applied a multilingual DistilBERT transformer model to classify records as positive, negative or neutral. Average model confidence was 57.24%.

4. **Topic Modelling:** Implemented BERTopic on 772 documents and identified three main topics: business and stock-market content, Artificial Intelligence and Climate Change.

5. **Influencer Network Analysis:** Built a directed interaction graph using NetworkX and calculated PageRank scores and communities. Identified 855 nodes, 582 edges and 274 communities.

6. **Truthfulness Classification:** Trained a TF-IDF and Logistic Regression model using the LIAR dataset. Achieved 60.09% test accuracy and 64.30% F1-score.

7. **Neo4j Integration:** Imported 855 channel nodes and 582 relationships into the Neo4j graph database.

8. **Elasticsearch and Kibana:** Indexed 921 cleaned records into Elasticsearch and created Kibana visualizations for exploring the collected data.

9. **Interactive Dashboard:** Developed a Streamlit application integrating dataset statistics, sentiment analysis, topic modelling, influencer rankings, truthfulness classification and Elasticsearch search.

### Current Implementation Scope

The current implementation is a sample-based YouTube intelligence prototype. Reddit API integration remains unconfigured because credentials were unavailable. The sentiment and truthfulness models are experimental and their predictions should not be treated as verified facts.

---

## What I Learned

Working on these projects helped me improve my understanding of statistical concepts and their practical applications. I also gained experience in:

- Building backend APIs using FastAPI
- Developing frontend interfaces using React
- Working with datasets using Python
- Applying statistical and machine learning techniques
- Connecting frontend and backend applications
- Deploying applications using cloud platforms
- Organizing and maintaining projects using Git and GitHub

## Repository Structure

```text
Nexvia-Data-Science-Internship/
│
├── DS-04-Experimentation-Causal-Inference/
│   ├── backend/
│   └── frontend/
│
├── DS-03-Second-Project/
│
└── README.md
```

## Note

This repository represents my learning and project work during the Nexvia Data Science Internship. The results of statistical and causal analyses depend on the quality of the dataset, selected method, and underlying assumptions.
