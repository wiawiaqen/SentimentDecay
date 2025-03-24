import asyncio
import sys
from pathlib import Path

# Adjust sys.path to allow relative imports when running as a script
if __name__ == "__main__" and __package__ is None:
    current_file = Path(__file__).resolve()
    sys.path.append(str(current_file.parents[2]))
    __package__ = "data.get_news"

import pandas as pd
from ..models import Article
from .reuters_crawler import ReutersCrawler
from sqlalchemy.orm import sessionmaker

from .database_connection import get_sql_server_connection
from utils.sentiment import SentimentAnalyzer
from utils.topic_modeling import TopicModeler


async def fetch_news() -> None:
    """Main function to fetch, process, save financial news articles, and insert into the database."""
    crawler = ReutersCrawler()
    sentiment_analyzer = SentimentAnalyzer()
    topic_modeler = TopicModeler()
    section_ids = [
        "/markets/us/",
        "/business/finance/",
        "/markets/currencies/",
        "/business/future-of-money/",
        "/markets/wealth/",
        "/markets/stocks/",
    ]
    max_offset = 200
    page_size = 9

    articles = []
    for section_id in section_ids:
        responses = await crawler.fetch_articles([section_id], max_offset, page_size)
        for response in responses:
            articles.extend(
                crawler.extract_article_data(
                    response.get("result", {}).get("articles", []), section_id
                )
            )

    # Predict sentiment scores and topics for each article
    for article in articles:
        sentiment = sentiment_analyzer.get_sentiment(article["description"])
        article["sentiment_label"] = sentiment["sentiment"]
        article["sentiment_score"] = sentiment["confidence"]
        topic, topic_prob = topic_modeler.get_topic(article["description"])
        article["topic"] = topic
        article["topic_probability"] = topic_prob

    # Save data to CSV
    df = pd.DataFrame(articles)

    # Ensure specified columns exist before applying astype
    dtype_mapping = {
        "word_count": "float32",
        "company_rics": "string",
        "ad_topics": "string",
        "sentiment_label": "string",
        "sentiment_score": "float32",
        "topic": "string",
        "topic_probability": "float32",
    }
    existing_columns = {col: dtype for col, dtype in dtype_mapping.items() if col in df.columns}
    df = df.astype(existing_columns)

    # Explode list columns
    list_columns = ["company_rics", "ad_topics"]
    for col in list_columns:
        if col in df.columns:
            df[col] = df[col].str.join(",")

    output_file = "financial_news_data.csv"
    df.to_csv(
        output_file, index=False, encoding="utf-8", date_format="%Y-%m-%dT%H:%M:%S.%fZ"
    )
    print(f"Data successfully saved to {output_file}.")

    # Push data to the database
    engine = get_sql_server_connection()
    Session = sessionmaker(bind=engine)
    session = Session()

    for article in articles:
        db_article = Article(
            id=article["id"],
            title=article["title"],
            description=article["description"],
            published_time=article["published_time"],
            word_count=article["word_count"],
            authors=article["authors"],
            section_id=article["section_id"],
            sentiment_label=article["sentiment_label"],
            sentiment_score=article["sentiment_score"],
            topic=article["topic"],
            topic_probability=article["topic_probability"],
        )
        session.merge(db_article)
    session.commit()

    print("Data successfully pushed to the database.")

    # Group articles by section and topic for visualization
    group_articles_by_section_or_topic(articles, group_by="section_id")
    group_articles_by_section_or_topic(articles, group_by="topic")


def group_articles_by_section_or_topic(articles, group_by="section_id"):
    """
    Group articles by section or topic and save the grouped data for visualization.
    Args:
        articles (list): List of article dictionaries.
        group_by (str): Field to group by ("section_id" or "topic").
    """
    df = pd.DataFrame(articles)
    grouped = df.groupby(group_by).agg(
        sentiment_score_avg=("sentiment_score", "mean"),
        sentiment_score_std=("sentiment_score", "std"),
        article_count=("id", "count"),
    ).reset_index()

    output_file = f"grouped_by_{group_by}.csv"
    grouped.to_csv(output_file, index=False)
    print(f"Grouped data saved to {output_file}.")


if __name__ == "__main__":
    asyncio.run(fetch_news())
