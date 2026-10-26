# Customer Review Intelligence using Transformers + RAG

## Project Overview

Customer Review Intelligence is a Natural Language Processing (NLP) project that analyzes movie reviews to help users explore viewer opinions.

The project uses a Transformer model for sentiment prediction and Sentence Transformers for semantic search. It retrieves relevant reviews and provides simple, keyword-based complaint insights through an interactive Streamlit application.

## Project Objectives

- Predict whether a movie review is positive or negative.
- Explore the sentiment distribution of reviews.
- Find relevant reviews using semantic search.
- Identify possible complaint themes using keyword matching.
- Display review insights through an interactive web application.

## Dataset

- **Dataset:** Rotten Tomatoes Movie Reviews
- **Source:** Hugging Face Datasets
- **Dataset identifier:** `cornell-movie-review-data/rotten_tomatoes`
- **Sample size:** 2,000 reviews
- **Input:** Movie review text
- **Label:** Positive (1) or Negative (0)

The project uses a reproducible sample from the training split.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Hugging Face Transformers
- Sentence Transformers
- Scikit-learn
- Streamlit
- Pytest

## Main Features

### 1. Exploratory Data Analysis
Explores the dataset, checks data quality, and generates visualizations of sentiment distribution and review length.

### 2. Transformer-Based Sentiment Analysis
Uses a pretrained DistilBERT sentiment model to predict positive or negative sentiment.

### 3. Semantic Review Search
Uses sentence embeddings to retrieve reviews that are semantically similar to a user's question.

### 4. Review Insights
Uses simple keyword matching to identify possible complaint themes in retrieved reviews.

### 5. Interactive Dashboard
Provides a Streamlit interface where users can enter questions and inspect relevant movie reviews.

## Project Structure

```text
Customer_Review_Intelligence/
├── app.py
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── reports/
│   └── figures/
├── src/
│   ├── download_dataset.py
│   ├── eda.py
│   ├── sentiment_analysis.py
│   ├── review_retrieval.py
│   ├── search_reviews.py
│   └── rag_answer.py
└── tests/
    └── test_rag_answer.py
```

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/Ilakiya056/customer-review-intelligence.git
cd customer-review-intelligence
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## How to Run

Run the following scripts from the project root in the indicated order.

### Step 1: Download the dataset

```bash
python -m src.download_dataset
```

### Step 2: Perform exploratory data analysis

```bash
python -m src.eda
```

### Step 3: Run sentiment analysis

```bash
python -m src.sentiment_analysis
```

### Step 4: Prepare review embeddings for retrieval

```bash
python -m src.review_retrieval
```

### Step 5: Launch the web application

```bash
python -m streamlit run app.py --server.port 8502 --server.fileWatcherType none
```

Open the local address displayed in the terminal to use the application.

## Running Tests

Run the tests from the project root:

```bash
python -m pytest tests -v
```

The current tests check how the application handles empty search results and positive-review questions.

## Current Results

- Processed 2,000 movie reviews.
- Generated 384-dimensional sentence embeddings.
- Created exploratory data visualizations.
- Achieved approximately 90% sentiment accuracy on the sampled data used during development.

**Important:** The reported accuracy was measured on the sampled data used during development, not on a separate held-out test set. It should not be treated as an independent benchmark of model performance.

## Limitations and Future Improvements

- The current dataset contains movie reviews, not general product reviews.
- Complaint themes are identified using keyword matching.
- The current question-answering component retrieves relevant reviews and summarizes limited evidence; it does not yet generate answers using a large language model.
- Positive themes are not automatically categorized yet.
- Future work can include held-out evaluation, better theme extraction, and a genuine retrieval-augmented generation pipeline using a language model.

## Author

**Ilakiya S.**

Aspiring Data Scientist | Statistics Graduate