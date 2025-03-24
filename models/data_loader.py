"""
This module contains classes and functions for data preprocessing.
"""
import pandas as pd
from datetime import timedelta
from bertopic import BERTopic
import yfinance as yf

class NewsDataHandler:
    def __init__(self, news_path, headline_col="Headlines", time_col="Time", description_col="Description"):
        """
        Args:
            news_path (str): Path to the Kaggle financial news dataset (CSV).
            headline_col (str): Column name for news headlines.
            time_col (str): Column name for timestamps.
            description_col (str): Column name for news descriptions.
        """
        self.news_path = news_path
        self.headline_col = headline_col
        self.time_col = time_col
        self.description_col = description_col
        self.news_df = self._load_news_data()
        self.topic_model = None
        self.topic_info = None

    def _load_news_data(self):
        """Load and preprocess the Kaggle financial news dataset."""
        news_df = pd.read_csv(self.news_path)

        # Check if the 'Date' column exists
        if "Date" not in news_df.columns:
            raise KeyError("Missing 'Date' column in news data. Ensure the input file has the correct format.")

        # Rename 'Date' to 'published_time' for consistency
        news_df.rename(columns={"Date": "published_time"}, inplace=True)

        # Ensure the time column exists
        if self.time_col not in news_df.columns:
            raise KeyError(f"Expected column '{self.time_col}' not found in news data.")

        # Parse the "Time" column with ISO 8601 or mixed formats
        try:
            news_df[self.time_col] = pd.to_datetime(news_df[self.time_col], format="ISO8601")
        except ValueError:
            news_df[self.time_col] = pd.to_datetime(news_df[self.time_col], errors="coerce")

        # Rename columns if necessary
        if "description" in news_df.columns:
            news_df.rename(columns={"description": "Description"}, inplace=True)
        elif "Description" not in news_df.columns:
            raise KeyError("Missing 'Description' column in news data. Ensure the input file has the correct format.")

        return news_df

    def add_topic_modeling(self, n_topics=15):
        """Perform topic modeling on news headlines."""
        # Create and fit BERTopic
        self.topic_model = BERTopic(
            language="english", 
            nr_topics=n_topics,
            calculate_probabilities=True
        )
        topics, _ = self.topic_model.fit_transform(self.news_df[self.headline_col])
        
        # Store results
        self.news_df["topic"] = topics
        self.topic_info = self.topic_model.get_topic_info()
        
    def get_topic_sentiment(self, sentiment_col="sentiment"):
        """Aggregate sentiment per topic."""
        return self.news_df.groupby("topic")[sentiment_col].value_counts(normalize=True)

    def filter_by_time_range(self, start_date, end_date):
        """Filter news within a specific time range."""
        mask = (
            (self.news_df[self.time_col] >= start_date) &
            (self.news_df[self.time_col] <= end_date)
        )
        return self.news_df[mask]

    def align_to_stock_data(self, stock_timestamps, time_window="1D"):
        """
        Align news to stock timestamps with a lag window.
        Args:
            stock_timestamps (pd.Series): Timestamps from stock data.
            time_window (str): Max time between news and stock data (e.g., "1D" for 1 day).
        Returns:
            pd.DataFrame: News articles aligned to stock timestamps.
        """
        # Ensure the time column is timezone-naive
        self.news_df[self.time_col] = self.news_df[self.time_col].dt.tz_localize(None)
        stock_timestamps = stock_timestamps.dt.tz_localize(None)

        aligned_news = []
        for stock_time in stock_timestamps:
            # Filter news within the time window
            mask = (
                (self.news_df[self.time_col] >= stock_time - pd.Timedelta(time_window)) &
                (self.news_df[self.time_col] <= stock_time)
            )
            filtered_news = self.news_df[mask].copy()
            filtered_news["aligned_stock_time"] = stock_time  # Critical for decay
            aligned_news.append(filtered_news)
        return pd.concat(aligned_news)

    def load_and_align_news(self):
        """
        Load and preprocess news data, aligning it as needed.
        Returns:
            pd.DataFrame: Aligned news data.
        """
        news_data = pd.read_csv(self.news_path)
        # Add any preprocessing or alignment logic here
        # For example, renaming columns, handling missing values, etc.
        news_data = news_data.rename(columns={"headline": "Headlines", "description": "Description"})
        return news_data


def download_stock_data(symbol, start_date, end_date, output_path):
    """
    Download stock data from Yahoo Finance and save it as a CSV file.
    Args:
        symbol (str): Stock symbol (e.g., "AAPL").
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        output_path (str): Path to save the CSV file.
    """
    data = yf.download(symbol, start=start_date, end=end_date)
    data.to_csv(output_path)
    print(f"Stock data saved to {output_path}")
    
def clean_stock_data(input_path, output_path):
    """
    Clean the stock data CSV file and save it in a standard format.
    Args:
        input_path (str): Path to the raw stock data CSV.
        output_path (str): Path to save the cleaned CSV.
    """
    # Load the raw CSV
    df = pd.read_csv(input_path)

    # Attempt to rename columns if they don't match the expected format
    if "Date" not in df.columns:
        if "Unnamed: 0" in df.columns:  # Common case where the date column is unnamed
            df.rename(columns={"Unnamed: 0": "Date"}, inplace=True)
        elif df.index.name == "Date":  # Handle cases where 'Date' is the index
            df.reset_index(inplace=True)
        else:
            raise ValueError("Missing 'Date' column in stock data.")

    # Drop rows with missing or invalid 'Date' values
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # Ensure required columns exist
    required_columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in stock data.")

    # Save the cleaned CSV
    df.to_csv(output_path, index=False)
    print(f"Cleaned stock data saved to {output_path}")
