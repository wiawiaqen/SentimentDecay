import asyncio
import os
from datetime import datetime
import pandas as pd
import yfinance as yf
from sqlalchemy.orm import sessionmaker
from data.get_news.database_connection import get_sql_server_connection
from data.models import Article
from data.get_news.__main__ import fetch_news
from utils.data_loader import NewsDataHandler, clean_stock_data  
from utils.sentiment import SentimentAnalyzer
from models.decay_models import ExponentialDecay
import matplotlib.pyplot as plt

def fetch_latest_stock_price(symbol="AAPL", output_file="results/latest_stock_data.csv"):
    """
    Fetch the latest stock prices for the given symbol from Yahoo Finance.
    The date range is determined from the database (earliest to latest article date).
    Args:
        symbol (str): Stock symbol (e.g., "AAPL").
        output_file (str): Path to save the stock data.
    """
    # Connect to the database and fetch the date range
    engine = get_sql_server_connection()
    Session = sessionmaker(bind=engine)
    session = Session()

    earliest_date = session.query(Article.published_time).order_by(Article.published_time.asc()).first()
    latest_date = session.query(Article.published_time).order_by(Article.published_time.desc()).first()

    if not earliest_date or not latest_date:
        print("No articles found in the database to determine the date range.")
        return

    start_date = earliest_date[0].strftime("%Y-%m-%d")
    end_date = latest_date[0].strftime("%Y-%m-%d")

    # Fetch stock data from Yahoo Finance with auto_adjust explicitly set to False
    data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False)
    data.reset_index(inplace=True)  # Ensure 'Date' is a column, not an index
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    data.to_csv(output_file, index=False)  # Save with 'Date' as a column
    print(f"Latest stock data saved to {output_file}.")

    # Clean the stock data
    clean_stock_data(output_file, output_file)

def perform_analytics(news_file="financial_news_data.csv", stock_file="results/latest_stock_data.csv"):
    """
    Perform analytics by aligning news data with stock data and calculating decayed sentiment.
    Args:
        news_file (str): Path to the news data CSV file.
        stock_file (str): Path to the stock data CSV file.
    """
    # Load news and stock data
    news_handler = NewsDataHandler(news_path=news_file, time_col="published_time")  # Specify the correct time column
    stock_data = pd.read_csv(stock_file)

    # Ensure the 'Date' column exists and is parsed as datetime
    if "Date" not in stock_data.columns:
        raise ValueError("Missing 'Date' column in stock data.")
    stock_data["Date"] = pd.to_datetime(stock_data["Date"], errors="coerce").dt.tz_localize(None)  # Remove timezone
    if stock_data["Date"].isnull().any():
        raise ValueError("Invalid or missing dates in the 'Date' column of stock data.")

    # Align news data to stock data
    aligned_news = news_handler.align_to_stock_data(stock_data["Date"], time_window="1D")

    # Perform sentiment analysis
    sentiment_analyzer = SentimentAnalyzer() 
    if "Description" not in aligned_news.columns:
        raise KeyError("Missing 'Description' column in aligned news data. Ensure the news data is preprocessed correctly.")
    aligned_news["sentiment_score"] = aligned_news["Description"].apply(
        lambda text: sentiment_analyzer.get_sentiment(text)["confidence"]
    )

    # Apply exponential decay with updated parameters
    decay_model = ExponentialDecay(decay_rate=0.4, period_type="hour", period_amount=1, lag_days=3)
    decayed_scores_df = decay_model.compute_decayed_scores(aligned_news, stock_data["Date"])  # Get decayed scores DataFrame
    print("Decayed scores computed successfully.")

    # # Ensure the 'Date' columns in both DataFrames are properly formatted and aligned
    # decayed_scores_df["Date"] = pd.to_datetime(decayed_scores_df["Date"], errors="coerce").dt.tz_localize(None)
    # stock_data["Date"] = pd.to_datetime(stock_data["Date"], errors="coerce").dt.tz_localize(None)

    # Drop rows with invalid or missing dates
    # decayed_scores_df = decayed_scores_df.dropna(subset=["Date"])
    # stock_data = stock_data.dropna(subset=["Date"])

    # Ensure the returned DataFrame has the required columns
    # if "Date" not in decayed_scores_df.columns or "decayed_sentiment" not in decayed_scores_df.columns:
        # raise ValueError("The decayed scores DataFrame must contain 'Date' and 'decayed_sentiment' columns.")

    # Align the 'Date' columns to ensure compatibility for merging
    decayed_scores_df["Date"] = decayed_scores_df["Date"].dt.normalize()
    stock_data["Date"] = stock_data["Date"].dt.normalize()

    # Merge decayed_sentiment back into the original stock_data
    stock_data = stock_data.merge(decayed_scores_df, on="Date", how="left")

    # Check if 'decayed_sentiment' exists after the merge
    if "decayed_sentiment" not in stock_data.columns:
        raise ValueError("Column 'decayed_sentiment' not found in stock data after applying decay model.")

    # Ensure the 'Close' column exists
    if "Close" not in stock_data.columns:
        raise KeyError("Missing 'Close' column in stock data. Ensure the input file has the correct format.")

    # Calculate correlation
    stock_data["price_change"] = stock_data["Close"].pct_change()
    correlation = stock_data["decayed_sentiment"].corr(stock_data["price_change"])
    print(f"Correlation between decayed sentiment and price changes: {correlation:.4f}")

    # Save analytics results
    stock_data.to_csv("results/analytics_stock_data.csv", index=False)
    aligned_news.to_csv("results/analytics_news_data.csv", index=False)
    print("Analytics results saved to 'results/analytics_stock_data.csv' and 'results/analytics_news_data.csv'.")

def visualize_grouped_data(grouped_file, stock_file="results/analytics_stock_data.csv"):
    """
    Visualize grouped data (price changes, sentiment scores, and decayed sentiment vs. prices).
    Args:
        grouped_file (str): Path to the grouped data CSV file.
        stock_file (str): Path to the stock data CSV file.
    """
    grouped_data = pd.read_csv(grouped_file)
    stock_data = pd.read_csv(stock_file, parse_dates=["Date"])

    # # Plot sentiment scores
    # plt.figure(figsize=(10, 6))
    # plt.bar(grouped_data.iloc[:, 0], grouped_data["sentiment_score_avg"], yerr=grouped_data["sentiment_score_std"])
    # plt.title("Average Sentiment Scores by Group")
    # plt.xlabel(grouped_data.columns[0])
    # plt.ylabel("Sentiment Score")
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.savefig(f"results/sentiment_scores_{grouped_file.split('_')[-1].split('.')[0]}.png")
    # plt.show()

    # # Plot stock price changes
    # stock_data["price_change"] = stock_data["Close"].pct_change()
    # print(stock_data.columns)
    # plt.figure(figsize=(10, 6))
    # plt.plot(stock_data["Date"], stock_data["price_change"], label="Price Change")
    # plt.title("Stock Price Changes Over Time")
    # plt.xlabel("Date")
    # plt.ylabel("Price Change")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("results/price_changes.png")
    # plt.show()

    # Plot decayed sentiment vs. stock prices
    if "decayed_sentiment" in stock_data.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(stock_data["Date"], stock_data["price_change"], label="Close Price", color="blue")
        plt.plot(stock_data["Date"], stock_data["decayed_sentiment"], label="Decayed Sentiment", color="orange")
        plt.title("Decayed Sentiment vs. Stock Prices")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig("results/decayed_sentiment_vs_prices.png")
        plt.show()
    else:
        print("Column 'decayed_sentiment' not found in stock data. Ensure analytics were performed.")

async def main_pipeline():
    """
    Main pipeline to fetch the latest stock prices, fetch the latest articles, and perform analytics.
    """
    # Step 1: Fetch the latest stock prices
    # fetch_latest_stock_price()

    # # Step 2: Fetch the latest articles
    # await fetch_news()

    # Step 3: Perform analytics
    perform_analytics()

    # # Step 4: Visualize grouped data
    visualize_grouped_data("grouped_by_section_id.csv")
    # visualize_grouped_data("grouped_by_topic.csv")

if __name__ == "__main__":
    asyncio.run(main_pipeline())
