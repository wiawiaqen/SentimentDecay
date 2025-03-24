"""
This module contains a wrapper for FinBERT sentiment analysis.
"""
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from tqdm import tqdm 

class SentimentAnalyzer:
    def __init__(self, model_name="ProsusAI/finbert"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.nlp = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)

    def get_sentiment(self, text):
        """Returns sentiment score (positive/negative/neutral) and confidence."""
        result = self.nlp(text)[0]
        return {
            "sentiment": result["label"],
            "confidence": result["score"]
        }

    def batch_process(self, texts):
        """
        Process multiple news articles in bulk with a progress bar.
        Args:
            texts (list): List of text strings to process.
        Returns:
            list: List of sentiment results.
        """
        results = []
        # Use tqdm to show progress and remaining time
        for text in tqdm(texts, desc="Processing", unit="text"):
            result = self.nlp(text)[0]
            results.append(result)
        return results

    def add_decayed_scores(self, df, decay_model):
        """Add decayed sentiment scores to DataFrame."""
        df["sentiment_score"] = df["sentiment_label"].map(self.sentiment_map)
        df["decayed_score"] = decay_model.apply_decay(
            df["sentiment_score"], 
            df["time_elapsed"]
        )
        return df