# Thesis Implementation

This repository contains the implementation of a thesis project focused on event extraction, sentiment analysis, and financial data processing using machine learning models.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Contributing](#contributing)

## Overview
This project integrates natural language processing (NLP) techniques and machine learning models to analyze financial news and stock data. It includes modules for event extraction, sentiment analysis, topic modeling, and data preprocessing.

## Features
- **Event Extraction**: Extracts events and arguments from financial news articles.
- **Sentiment Analysis**: Analyzes the sentiment of news articles and correlates it with stock price changes.
- **Topic Modeling**: Groups news articles into topics using BERTopic.
- **Stock Data Processing**: Downloads, cleans, and processes stock price data.
- **Grid Search**: Performs hyperparameter tuning for decay models.

## Directory Structure
```
├── data
│   ├── get_news
│   │   ├── __main__.py          # Main script for fetching and processing news
│   │   ├── reuters_crawler.py   # Reuters news crawler implementation
│   │   ├── database_connection.py # Database connection utilities
│   ├── raw_news                 # Directory for raw financial news data
│   ├── raw_stocks               # Directory for raw stock price data
├── models
│   ├── decay_models.py          # Sentiment decay models
│   ├── pretrained_models        # Pretrained models like BERTopic
├── dataset                      # Processed datasets for event extraction
├── utils
│   ├── sentiment.py             # Sentiment analysis utilities
│   ├── topic_modeling.py        # Topic modeling utilities
├── main.ipynb                   # Main Jupyter notebook for analysis
├── README.md                    # Project documentation
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/wiawiaqen/SentimentDecay.git
   cd SentimentDecay
   ```
2. Set up a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Fetch and Process News**:
   Run the news fetching script:
   ```bash
   python -m data.get_news
   ```
   This will fetch financial news, analyze sentiment, and save the data to a CSV file.

2. **Run Decay Models**:
   Use the `models/decay_models.py` module to apply sentiment decay models to the processed data.

3. **Analyze Results**:
   Open `main.ipynb` to visualize and analyze the results.

## Dependencies
View the `requirements.txt` file for a list of dependencies.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.
