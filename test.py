import argparse
import pandas as pd
import torch
import numpy as np
from datetime import datetime
import os
from utils.data_loader import NewsDataHandler, download_stock_data, clean_stock_data
from utils.sentiment import SentimentAnalyzer
from utils.topic_modeling import TopicModeler
from models.decay_models import ExponentialDecay, HarmonicDecay, PowerLawDecay, AdaptiveExponentialDecay
import matplotlib.pyplot as plt

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Stock Sentiment Analysis')
    parser.add_argument('--decay_rate', type=float, default=0.1,
                       help='Decay rate (lambda) for sentiment decay')
    parser.add_argument('--period_type', type=str, default='day',
                       choices=['hour', 'day', 'week'], help='Time period type')
    parser.add_argument('--period_amount', type=int, default=1,
                       help='Number of periods for decay window')
    parser.add_argument('--news_path', type=str, 
                       default='results/processed_news_data.csv',
                       help='Path to news CSV file')
    parser.add_argument('--stock_path', type=str,
                       default='data/raw_stocks/AAPL_cleaned.csv',
                       help='Path to stock data CSV')
    parser.add_argument('--output_dir', type=str,
                       default='results', help='Output directory')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create unique output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"results_{timestamp}_{args.decay_rate}_{args.period_type}_{args.period_amount}.txt")
    
    with open(output_file, 'w') as f:
        # Write parameters
        f.write("PARAMETERS:\n")
        f.write(f"Decay Rate: {args.decay_rate}\n")
        f.write(f"Period Type: {args.period_type}\n")
        f.write(f"Period Amount: {args.period_amount}\n")
        f.write("\nRESULTS:\n")

        # Load and process data
        try:
            # 1. News Data
            news_handler = NewsDataHandler(news_path=args.news_path)
            aligned_news = news_handler.load_and_align_news()  # Updated to call the appropriate method
            
            # 2. Stock Data
            stock_data = pd.read_csv(args.stock_path, parse_dates=["Date"])
            
            # 3. Sentiment Analysis
            sentiment_analyzer = SentimentAnalyzer()
            topic_modeler = TopicModeler()
            if 'sentiment_score' not in aligned_news.columns:
                aligned_news["combined_text"] = aligned_news["Headlines"] + " " + aligned_news["Description"]
                sentiment_scores = sentiment_analyzer.batch_process(aligned_news["combined_text"].tolist())
                aligned_news["sentiment_label"] = [s["label"] for s in sentiment_scores]
                aligned_news["sentiment_score"] = [
                    s["score"] * (1 if s["label"] == "positive" else (-1 if s["label"] == "negative" else 0))
                    for s in sentiment_scores
                ]  
            if 'topic' not in aligned_news.columns:
                topics = [topic_modeler.get_topic(text) for text in aligned_news["Headlines"].tolist()]
                aligned_news["topic"] = [t[0] for t in topics]
                aligned_news["topic_probability"] = [t[1] for t in topics]
            
            # 4. Apply Decay
            decay_model = ExponentialDecay(
                decay_rate=args.decay_rate,
                period_type=args.period_type,
                period_amount=args.period_amount
            )
            
            stock_data["decayed_sentiment"] = decay_model.compute_decayed_scores(
                aligned_news, stock_data["Date"]
            )
            
            # 5. Analyze Data by Topic
            topic_groups = aligned_news.groupby("topic")
            f.write("\nAnalysis by Topic:\n")
            for topic, group in topic_groups:
                avg_sentiment = group["sentiment_score"].mean()
                f.write(f"Topic: {topic}, Average Sentiment: {avg_sentiment:.4f}\n")
            
            # 6. Calculate Correlations
            stock_data["price_change"] = stock_data["Close"].pct_change()
            correlation = stock_data["decayed_sentiment"].corr(stock_data["price_change"])
            
            f.write(f"\nCorrelation between decayed sentiment and price changes: {correlation:.4f}\n")
            
            # 7. Save processed data
            stock_data.to_csv(os.path.join(args.output_dir, "processed_stock_data.csv"), index=False)
            aligned_news.to_csv(os.path.join(args.output_dir, "processed_news_data.csv"), index=False)
            f.write(f"\nOutput files saved to: {args.output_dir}")
        except Exception as e:
            f.write(f"\nERROR OCCURRED: {str(e)}")
            raise

def grid_search(report_file="results/grid_search_report.txt", lag_days=1):
    """Perform a grid search over decay parameters for Exponential, Harmonic, and Power Law Decay with lagged data."""
    decay_rates = [0.05, 0.1, 0.2, 0.01, 0.02, 0.3, 0.4, 0.5]
    decay_exponents = [0.5, 1.0, 1.5]  # For Power Law Decay
    period_types = ["hour", "day", "week"]
    period_amounts = [1, 2, 3]

    results = []

    for decay_model_type in ["Exponential", "Harmonic", "PowerLaw"]:
        for period_type in period_types:
            for period_amount in period_amounts:
                if decay_model_type == "PowerLaw":
                    for decay_exponent in decay_exponents:
                        print(f"Testing: {decay_model_type} Decay, decay_exponent={decay_exponent}, period_type={period_type}, period_amount={period_amount}")
                        try:
                            # Set up arguments
                            args = argparse.Namespace(
                                decay_exponent=decay_exponent,
                                period_type=period_type,
                                period_amount=period_amount,
                                news_path="results/processed_news_data.csv",
                                stock_path="data/raw_stocks/AAPL_cleaned.csv",
                                output_dir=f"results/{decay_model_type.lower()}_decay_{decay_exponent}_{period_type}_{period_amount}"
                            )

                            # Create output directory
                            os.makedirs(args.output_dir, exist_ok=True)

                            # Load and process data
                            # 1. News Data
                            news_handler = NewsDataHandler(news_path=args.news_path)
                            aligned_news = news_handler.load_and_align_news()

                            # Apply lag to align news data with stock data
                            stock_data = pd.read_csv(args.stock_path, parse_dates=["Date"])
                            stock_data["lagged_date"] = stock_data["Date"] + pd.Timedelta(days=lag_days)
                            aligned_news["Time"] = pd.to_datetime(aligned_news["Time"])  # Ensure consistent datetime type
                            aligned_news = aligned_news.merge(
                                stock_data[["lagged_date"]],
                                left_on="Time",
                                right_on="lagged_date",
                                how="inner"
                            )

                            # 2. Apply Decay
                            decay_model = PowerLawDecay(
                                decay_exponent=decay_exponent,
                                period_type=period_type,
                                period_amount=period_amount
                            )

                            stock_data["decayed_sentiment"] = decay_model.compute_decayed_scores(
                                aligned_news, stock_data["Date"]
                            )

                            # 3. Calculate Correlations
                            stock_data["price_change"] = stock_data["Close"].pct_change()
                            correlation = stock_data["decayed_sentiment"].corr(stock_data["price_change"])
                            results.append((decay_model_type, decay_exponent, period_type, period_amount, correlation))

                        except Exception as e:
                            print(f"Error for {decay_model_type} Decay, decay_exponent={decay_exponent}, period_type={period_type}, period_amount={period_amount}: {e}")
                else:
                    for decay_rate in decay_rates:
                        print(f"Testing: {decay_model_type} Decay, decay_rate={decay_rate}, period_type={period_type}, period_amount={period_amount}")
                        try:
                            # Set up arguments
                            args = argparse.Namespace(
                                decay_rate=decay_rate,
                                period_type=period_type,
                                period_amount=period_amount,
                                news_path="results/processed_news_data.csv",
                                stock_path="data/raw_stocks/AAPL_cleaned.csv",
                                output_dir=f"results/{decay_model_type.lower()}_decay_{decay_rate}_{period_type}_{period_amount}"
                            )

                            # Create output directory
                            os.makedirs(args.output_dir, exist_ok=True)

                            # Load and process data
                            # 1. News Data
                            news_handler = NewsDataHandler(news_path=args.news_path)
                            aligned_news = news_handler.load_and_align_news()

                            # Apply lag to align news data with stock data
                            stock_data = pd.read_csv(args.stock_path, parse_dates=["Date"])
                            stock_data["lagged_date"] = stock_data["Date"] + pd.Timedelta(days=lag_days)
                            aligned_news["Time"] = pd.to_datetime(aligned_news["Time"])  # Ensure consistent datetime type
                            aligned_news = aligned_news.merge(
                                stock_data[["lagged_date"]],
                                left_on="Time",
                                right_on="lagged_date",
                                how="inner"
                            )

                            # 2. Apply Decay
                            if decay_model_type == "Exponential":
                                decay_model = ExponentialDecay(
                                    decay_rate=decay_rate,
                                    period_type=period_type,
                                    period_amount=period_amount
                                )
                            elif decay_model_type == "Harmonic":
                                decay_model = HarmonicDecay(
                                    period_type=period_type,
                                    period_amount=period_amount
                                )

                            stock_data["decayed_sentiment"] = decay_model.compute_decayed_scores(
                                aligned_news, stock_data["Date"]
                            )

                            # 3. Calculate Correlations
                            stock_data["price_change"] = stock_data["Close"].pct_change()
                            correlation = stock_data["decayed_sentiment"].corr(stock_data["price_change"])
                            results.append((decay_model_type, decay_rate, period_type, period_amount, correlation))

                        except Exception as e:
                            print(f"Error for {decay_model_type} Decay, decay_rate={decay_rate}, period_type={period_type}, period_amount={period_amount}: {e}")

    # Save results to report file
    with open(report_file, "w") as report:
        report.write("Grid Search Results with Lag:\n\n")
        for result in results:
            report.write(
                f"Decay Model: {result[0]}, Parameter: {result[1]}, Period Type: {result[2]}, "
                f"Period Amount: {result[3]}, Correlation: {result[4]:.4f}\n"
            )
    print(f"Grid search results with lag saved to {report_file}")

def adaptive_decay_experiment(report_file="results/adaptive_decay_report.txt"):
    """Experiment with Adaptive Exponential Decay."""
    # Load topic correlation data (example: from a precomputed file or database)
    topic_correlation_file = "results/topic_correlation.csv"
    topic_decay_rate_map = {}

    if os.path.exists(topic_correlation_file):
        topic_correlation_df = pd.read_csv(topic_correlation_file)
        # Map topics to decay rates: higher correlation -> slower decay (lower decay rate)
        max_correlation = topic_correlation_df["correlation"].max()
        min_correlation = topic_correlation_df["correlation"].min()
        topic_decay_rate_map = {
            row["topic"]: 0.05 + (0.2 - 0.05) * (1 - (row["correlation"] - min_correlation) / (max_correlation - min_correlation))
            for _, row in topic_correlation_df.iterrows()
        }
    else:
        print(f"Topic correlation file not found: {topic_correlation_file}")
        return

    # Default decay rate for topics not in the map
    default_decay_rate = 0.1

    # Load news and stock data
    news_handler = NewsDataHandler(news_path="results/processed_news_data.csv")
    aligned_news = news_handler.load_and_align_news()
    stock_data = pd.read_csv("data/raw_stocks/AAPL_cleaned.csv", parse_dates=["Date"])

    # Apply Adaptive Exponential Decay
    adaptive_decay_model = AdaptiveExponentialDecay(
        topic_decay_rate_map=topic_decay_rate_map,
        default_decay_rate=default_decay_rate,
        period_type="day",
        period_amount=1
    )

    stock_data["adaptive_decayed_sentiment"] = adaptive_decay_model.compute_decayed_scores(
        aligned_news, stock_data["Date"]
    )

    # Calculate correlation
    stock_data["price_change"] = stock_data["Close"].pct_change()
    correlation = stock_data["adaptive_decayed_sentiment"].corr(stock_data["price_change"])
    print(f"Correlation between adaptive decayed sentiment and price changes: {correlation:.4f}")

    # Save results
    output_dir = "results/adaptive_decay_experiment"
    os.makedirs(output_dir, exist_ok=True)
    stock_data.to_csv(os.path.join(output_dir, "adaptive_processed_stock_data.csv"), index=False)
    aligned_news.to_csv(os.path.join(output_dir, "adaptive_processed_news_data.csv"), index=False)

    # Save scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(stock_data["adaptive_decayed_sentiment"], stock_data["price_change"], alpha=0.5)
    plt.title("Adaptive Decay: Sentiment vs Price Change")
    plt.xlabel("Adaptive Decayed Sentiment")
    plt.ylabel("Price Change")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "adaptive_sentiment_vs_price_change.png"))
    plt.close()

    # Save 2-year line plot
    stock_data_2y = stock_data.set_index("Date").last("2Y")
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data_2y.index, stock_data_2y["Close"], label="Price", color="blue")
    plt.plot(stock_data_2y.index, stock_data_2y["adaptive_decayed_sentiment"], label="Adaptive Decayed Sentiment", color="orange")
    plt.title("Adaptive Decay: 2-Year Price and Sentiment Trend")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "adaptive_price_sentiment_2y.png"))
    plt.close()

    with open(report_file, "w") as report:
        report.write("Adaptive Decay Experiment Results:\n\n")
        report.write(f"Correlation between adaptive decayed sentiment and price changes: {correlation:.4f}\n")
        report.write(f"Results saved to {output_dir}\n")
    print(f"Adaptive decay experiment results saved to {report_file}")

def calculate_topic_correlation(news_path, stock_path, output_file="results/topic_correlation.csv", lag_days=1):
    """
    Calculate the correlation between sentiment scores and price changes for each topic with a lag.

    Args:
        news_path (str): Path to the news data CSV file.
        stock_path (str): Path to the stock data CSV file.
        output_file (str): Path to save the topic correlation file.
        lag_days (int): Number of days to lag the stock data.
    """
    # Load news and stock data
    news_data = pd.read_csv(news_path)
    stock_data = pd.read_csv(stock_path, parse_dates=["Date"])

    # Ensure the necessary columns exist
    if "topic" not in news_data.columns or "sentiment_score" not in news_data.columns:
        print("Error: News data must contain 'topic' and 'sentiment_score' columns.")
        return

    # Align news data to stock data with a lag
    news_data["Time"] = pd.to_datetime(news_data["Time"])
    stock_data["Date"] = pd.to_datetime(stock_data["Date"])
    stock_data["price_change"] = stock_data["Close"].pct_change()
    stock_data["lagged_date"] = stock_data["Date"] + pd.Timedelta(days=lag_days)

    # Merge stock data with news data by aligning timestamps
    aligned_news = news_data.merge(
        stock_data[["lagged_date", "price_change"]],
        left_on="Time",
        right_on="lagged_date",
        how="inner"
    )

    # Calculate correlation between sentiment scores and price changes for each topic
    topic_correlation = aligned_news.groupby("topic").apply(
        lambda group: group["sentiment_score"].corr(group["price_change"])
    ).reset_index()
    topic_correlation.columns = ["topic", "correlation"]

    # Save the topic correlation to a CSV file
    topic_correlation.to_csv(output_file, index=False)
    print(f"Topic correlation file saved to {output_file}.")

if __name__ == "__main__":
    calculate_topic_correlation(
        news_path="results/processed_news_data.csv",
        stock_path="data/raw_stocks/AAPL_cleaned.csv",
        lag_days=2
    )
    adaptive_decay_experiment(report_file="results/adaptive_decay_report_lag_2.txt")
    # grid_search(lag_days=4, report_file="results/grid_search_report_lag_4.txt")