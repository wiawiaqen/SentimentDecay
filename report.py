import os
import pandas as pd
from bertopic import BERTopic


def load_results(results_dir: str = "results") -> pd.DataFrame:
    """
    Load all results from the grid search output directories for each lag.
    Args:
        results_dir (str): Path to the directory containing grid search results.
    Returns:
        pd.DataFrame: DataFrame containing all results with parameters, correlations, and lag information.
    """
    results = []
    for lag_folder in os.listdir(results_dir):
        lag_path = os.path.join(results_dir, lag_folder)
        if os.path.isdir(lag_path) and lag_folder.startswith("lag_"):
            lag = int(lag_folder.split("_")[1])  # Extract lag number
            for root, dirs, files in os.walk(lag_path):
                for file in files:
                    if file == "grid_search_report.txt":
                        result_path = os.path.join(root, file)
                        with open(result_path, "r") as f:
                            lines = f.readlines()
                            for line in lines:
                                if line.startswith("Decay Model:"):
                                    parts = line.strip().split(", ")
                                    params = {
                                        "decay_model": parts[0].split(": ")[1],
                                        "parameter": float(parts[1].split(": ")[1]),
                                        "period_type": parts[2].split(": ")[1],
                                        "period_amount": int(parts[3].split(": ")[1]),
                                        "correlation": float(parts[4].split(": ")[1]),
                                        "lag": lag,
                                    }
                                    results.append(params)
    return pd.DataFrame(results)


def report_top_correlations(results_df, output_file="top_correlations.txt"):
    """
    Report the top 5 highest correlations for positive correlations and the lowest 5 for negative correlations
    for each lag and timeframe.
    Args:
        results_df (pd.DataFrame): DataFrame containing results with parameters, correlations, and lag information.
        output_file (str): Path to save the report.
    """
    with open(output_file, "w") as f:
        f.write(
            "Top 5 Parameter Combinations with Highest Positive Correlation and Lowest Negative Correlation "
            "for Each Lag and Timeframe:\n\n"
        )

        for lag in sorted(results_df["lag"].unique()):
            f.write(f"Lag: {lag}\n")
            for period_type in results_df["period_type"].unique():
                f.write(f"  Timeframe: {period_type}\n")
                
                # Handle positive correlations
                positive_results = results_df[
                    (results_df["lag"] == lag) & 
                    (results_df["period_type"] == period_type) & 
                    (results_df["correlation"] > 0)
                ]
                top_positive = positive_results.nlargest(5, "correlation")
                if not top_positive.empty:
                    f.write("    Top 5 Positive Correlations:\n")
                    for _, row in top_positive.iterrows():
                        f.write(
                            f"      Decay Model: {row['decay_model']}, Parameter: {row['parameter']}, "
                            f"Period Amount: {row['period_amount']}, Correlation: {row['correlation']:.4f}\n"
                        )
                
                # Handle negative correlations
                negative_results = results_df[
                    (results_df["lag"] == lag) & 
                    (results_df["period_type"] == period_type) & 
                    (results_df["correlation"] < 0)
                ]
                top_negative = negative_results.nsmallest(5, "correlation")
                if not top_negative.empty:
                    f.write("    Top 5 Negative Correlations:\n")
                    for _, row in top_negative.iterrows():
                        f.write(
                            f"      Decay Model: {row['decay_model']}, Parameter: {row['parameter']}, "
                            f"Period Amount: {row['period_amount']}, Correlation: {row['correlation']:.4f}\n"
                        )
                f.write("\n")
    print(f"Top 5 correlations (positive and negative) for each lag and timeframe saved to {output_file}")


def get_top_topics_with_texts(results_dir="results", model_path="models/pretrained_models/bertopic_model", top_n=10):
    """
    Get the top N topics with the highest correlation and translate them to text arrays.
    Args:
        results_dir (str): Path to the directory containing grid search results.
        model_path (str): Path to the pretrained BERTopic model.
        top_n (int): Number of top topics to retrieve.
    Returns:
        list: List of tuples containing topic ID and its associated text array.
    """
    # Load results
    results = load_results(results_dir)
    if results.empty:
        print("No results found in the specified directory.")
        return []

    # Load BERTopic model
    topic_model = BERTopic.load(model_path)

    # Get the top N topics with the highest correlation
    top_topics = results.nlargest(top_n, "correlation")

    topic_texts = []
    for _, row in top_topics.iterrows():
        topic_id = row.get("topic", None)
        if topic_id is not None:
            texts = topic_model.get_topic(topic_id)
            topic_texts.append((topic_id, texts))

    return topic_texts


def report_top_topics_with_texts(
    results_dir="results",
    model_path="models/pretrained_models/bertopic_model",
    output_file="top_topics.txt",
    top_n=10,
):
    """
    Report the top N topics with the highest correlation and their associated text arrays.
    Args:
        results_dir (str): Path to the directory containing grid search results.
        model_path (str): Path to the pretrained BERTopic model.
        output_file (str): Path to save the report.
        top_n (int): Number of top topics to retrieve.
    """
    top_topics = get_top_topics_with_texts(results_dir, model_path, top_n)

    with open(output_file, "w") as f:
        f.write("Top Topics with Highest Correlation and Their Associated Text Arrays:\n\n")
        for topic_id, texts in top_topics:
            f.write(f"Topic ID: {topic_id}\n")
            f.write("Associated Texts:\n")
            for text in texts:
                f.write(f"- {text}\n")
            f.write("\n")
    print(f"Top topics with their associated texts saved to {output_file}")


def include_grid_search_results(results_dir="results", output_file="final_report.txt"):
    """
    Include grid search results from all lag folders in the final report.
    Args:
        results_dir (str): Path to the directory containing grid search results.
        output_file (str): Path to save the final report.
    """
    with open(output_file, "a") as final_report:
        final_report.write("\nGrid Search Results:\n")
        for lag_folder in os.listdir(results_dir):
            lag_path = os.path.join(results_dir, lag_folder)
            if os.path.isdir(lag_path) and lag_folder.startswith("lag_"):
                report_file = os.path.join(lag_path, "grid_search_report.txt")
                if os.path.exists(report_file):
                    with open(report_file, "r") as grid_search_report:
                        final_report.write(f"\nResults for {lag_folder}:\n")
                        final_report.write(grid_search_report.read())
                else:
                    print(f"Warning: {report_file} not found.")


def include_adaptive_decay_results(report_file="results/adaptive_decay_report.txt", output_file="final_report.txt"):
    """
    Include adaptive decay experiment results in the final report.
    Args:
        report_file (str): Path to the adaptive decay report file.
        output_file (str): Path to save the final report.
    """
    with open(report_file, "r") as adaptive_decay_report, open(output_file, "a") as final_report:
        final_report.write("\nAdaptive Decay Experiment Results:\n")
        final_report.write(adaptive_decay_report.read())


if __name__ == "__main__":
    # Load results from the grid search
    results_df = load_results()

    # Ensure results are available
    if results_df.empty:
        print("No results found in the specified directory.")
    else:
        # Report top 10 correlations for each decay model
        report_top_correlations(results_df)

        # Report top 10 topics with the highest correlation and their text arrays
        report_top_topics_with_texts()

        # Include grid search and adaptive decay results in the final report
        include_grid_search_results()
        include_adaptive_decay_results()
