"""
This module contains a wrapper for BERTopic topic modeling.
"""
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

class TopicModeler:
    def __init__(self, model_path="models/pretrained_models/bertopic_model", embedding_model_name="all-MiniLM-L6-v2"):
        """
        Initialize the TopicModeler with a pretrained BERTopic model.
        Args:
            model_path (str): Path to the pretrained BERTopic model.
            embedding_model_name (str): Name of the embedding model for BERTopic.
        """
        embedding_model = SentenceTransformer(embedding_model_name)
        self.model = BERTopic.load(model_path, embedding_model=embedding_model)

    def preprocess_text(self, text):
        """
        Preprocess the input text for topic modeling.
        Args:
            text (str): Raw text input.
        Returns:
            str: Preprocessed text.
        """
        # Add any text preprocessing logic here if needed
        return text.strip().lower()

    def get_topic(self, text):
        """
        Predict the topic for the given text.
        Args:
            text (str): Input text.
        Returns:
            tuple: Predicted topic and its probability.
        """
        preprocessed_text = self.preprocess_text(text)
        topics, probs = self.model.transform([preprocessed_text])
        return topics.tolist()[0], probs[0]

    def get_topic_texts(self, topic_id):
        """
        Retrieve the text array associated with a given topic ID.
        Args:
            topic_id (int): The topic ID to retrieve texts for.
        Returns:
            list: List of texts associated with the topic.
        """
        return self.model.get_topic(topic_id)
