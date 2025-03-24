"""
This module contains implementations for sentiment decay models.
"""

import abc
import numpy as np
from typing import Literal
import logging
import pandas as pd

# Configure logging
logging.basicConfig(
    filename="debug.log",  # Save debug information to a log file
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class DecayModel(abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def apply_decay(self, sentiment_scores, timestamps):
        """Apply decay to sentiment scores over time."""
        pass


class ExponentialDecay:
    def __init__(self, decay_rate, period_type="day", period_amount=1):
        """
        Args:
            decay_rate (float): The rate of decay.
            period_type (str): The type of period ("day", "hour", etc.).
            period_amount (int): The amount of the period.
            lag_days (int): Number of days to lag the decay calculation.
        """
        self.decay_rate = decay_rate
        self.period_type = period_type
        self.period_amount = period_amount

    def _get_period_seconds(self):
        """Convert period_type to seconds."""
        if self.period_type == "hour":
            return 3600
        elif self.period_type == "day":
            return 3600 * 24
        elif self.period_type == "week":
            return 3600 * 24 * 7
        else:
            raise ValueError(f"Unsupported period_type: {self.period_type}")

    def compute_decayed_scores(self, aligned_news, stock_dates):
        """
        Compute decayed sentiment scores for stock dates based on aligned news.
        Args:
            aligned_news (pd.DataFrame): Aligned news data with sentiment scores.
            stock_dates (pd.Series): Stock dates to compute decayed sentiment for.
        Returns:
            np.ndarray: Decayed sentiment scores for each stock date.
        """
        # Ensure aligned_news contains 'sentiment_score'
        if "sentiment_score" not in aligned_news.columns:
            raise KeyError("Missing 'sentiment_score' column in aligned news data.")

        # Convert timestamps to numpy datetime64 (for vectorization)
        news_times = aligned_news["published_time"].values.astype("datetime64[s]")
        stock_times = stock_dates.values.astype("datetime64[s]")

        # Compute time deltas (in seconds) between stock dates and news times
        delta_time = (stock_times[:, None] - news_times[None, :]).astype(float)

        # Check for invalid delta_time values
        if np.any(np.isnan(delta_time)):
            raise ValueError("NaN values detected in delta_time computation.")

        # Convert time deltas to periods (e.g., days/weeks)
        period_seconds = self._get_period_seconds()
        delta_periods = delta_time / period_seconds

        # Apply exponential decay: S(t) = S0 * decay_rate^t
        decay_factors = np.power(self.decay_rate, delta_periods)

        # Mask invalid times (news published after stock date)
        valid_mask = delta_time >= 0
        decay_factors[~valid_mask] = 0

        # Multiply by sentiment scores and sum across news articles
        sentiment_scores = aligned_news["sentiment_score"].values
        decayed_scores = (decay_factors * sentiment_scores).sum(axis=1)

        # Check for NaN values in decayed_scores
        if np.any(np.isnan(decayed_scores)):
            raise ValueError("NaN values detected in decayed_scores. Check input data and decay parameters.")

        return decayed_scores


class HarmonicDecay:
    def __init__(
        self,
        period_type: Literal["day", "hour", "week"] = "day",
        period_amount: int = 1,
    ):
        self.period_type = period_type.lower()
        self.period_amount = period_amount
        self.period_seconds = self._get_period_seconds()

    def _get_period_seconds(self):
        """Convert period_type to seconds."""
        if self.period_type == "hour":
            return 3600
        elif self.period_type == "day":
            return 3600 * 24
        elif self.period_type == "week":
            return 3600 * 24 * 7
        else:
            raise ValueError(f"Unsupported period_type: {self.period_type}")

    def compute_decayed_scores(self, aligned_news, stock_dates):
        """
        Compute harmonic decayed sentiment scores accumulated for each stock date.

        Args:
            aligned_news (pd.DataFrame): News data with columns ["Time", "sentiment_score"].
            stock_dates (pd.Series): Trading dates to compute decay for.

        Returns:
            np.ndarray: Accumulated harmonic decayed sentiment scores for each stock date.
        """
        # Convert timestamps to numpy datetime64 (for vectorization)
        news_times = aligned_news["Time"].values.astype("datetime64[s]")
        stock_times = stock_dates.values.astype("datetime64[s]")

        # Compute time deltas (in seconds) between stock dates and news times
        delta_time = (stock_times[:, None] - news_times[None, :]).astype(float)

        # Check for invalid delta_time values
        if np.any(np.isnan(delta_time)):
            raise ValueError("NaN values detected in delta_time computation.")

        # Convert time deltas to periods (e.g., days/weeks)
        delta_periods = delta_time / (self.period_seconds * self.period_amount)

        # Avoid division by zero by setting a minimum value for delta_periods
        delta_periods = np.maximum(delta_periods, 1e-10)

        # Apply harmonic decay: S(t) = S0 / (1 + t)
        decay_factors = 1 / (1 + delta_periods)

        # Mask invalid times (news published after stock date)
        valid_mask = delta_time >= 0
        decay_factors[~valid_mask] = 0

        # Multiply by sentiment scores and sum across news articles
        sentiment_scores = aligned_news["sentiment_score"].values
        decayed_scores = (decay_factors * sentiment_scores).sum(axis=1)

        # Check for NaN values in decayed_scores
        if np.any(np.isnan(decayed_scores)):
            raise ValueError("NaN values detected in decayed_scores. Check input data and decay parameters.")

        return decayed_scores


class PowerLawDecay:
    def __init__(
        self,
        decay_exponent: float = 1.0,
        period_type: Literal["day", "hour", "week"] = "day",
        period_amount: int = 1,
    ):
        self.decay_exponent = decay_exponent
        self.period_type = period_type.lower()
        self.period_amount = period_amount
        self.period_seconds = self._get_period_seconds()

    def _get_period_seconds(self):
        """Convert period_type to seconds."""
        if self.period_type == "hour":
            return 3600
        elif self.period_type == "day":
            return 3600 * 24
        elif self.period_type == "week":
            return 3600 * 24 * 7
        else:
            raise ValueError(f"Unsupported period_type: {self.period_type}")

    def compute_decayed_scores(self, aligned_news, stock_dates):
        """
        Compute power-law decayed sentiment scores accumulated for each stock date.

        Args:
            aligned_news (pd.DataFrame): News data with columns ["Time", "sentiment_score"].
            stock_dates (pd.Series): Trading dates to compute decay for.

        Returns:
            np.ndarray: Accumulated power-law decayed sentiment scores for each stock date.
        """
        # Convert timestamps to numpy datetime64 (for vectorization)
        news_times = aligned_news["Time"].values.astype("datetime64[s]")
        stock_times = stock_dates.values.astype("datetime64[s]")

        # Compute time deltas (in seconds) between stock dates and news times
        delta_time = (stock_times[:, None] - news_times[None, :]).astype(float)

        # Check for invalid delta_time values
        if np.any(np.isnan(delta_time)):
            raise ValueError("NaN values detected in delta_time computation.")

        # Convert time deltas to periods (e.g., days/weeks)
        delta_periods = delta_time / (self.period_seconds * self.period_amount)

        # Apply power-law decay: S(t) = S0 / (1 + t^exponent)
        decay_factors = 1 / (1 + np.power(delta_periods, self.decay_exponent))

        # Mask invalid times (news published after stock date)
        valid_mask = delta_time >= 0
        decay_factors[~valid_mask] = 0

        # Multiply by sentiment scores and sum across news articles
        sentiment_scores = aligned_news["sentiment_score"].values
        decayed_scores = (decay_factors * sentiment_scores).sum(axis=1)

        # Check for NaN values in decayed_scores
        if np.any(np.isnan(decayed_scores)):
            raise ValueError("NaN values detected in decayed_scores. Check input data and decay parameters.")

        return decayed_scores


class AdaptiveExponentialDecay:
    def __init__(
        self,
        topic_decay_rate_map: dict,
        default_decay_rate: float = 0.1,
        period_type: Literal["day", "hour", "week"] = "day",
        period_amount: int = 1,
    ):
        """
        Adaptive Exponential Decay that adjusts decay rate based on topic correlation.

        Args:
            topic_decay_rate_map (dict): A mapping of topic IDs to decay rates.
            default_decay_rate (float): Default decay rate for topics not in the map.
            period_type (str): Time period type ("day", "hour", "week").
            period_amount (int): Number of periods for decay window.
        """
        self.topic_decay_rate_map = topic_decay_rate_map
        self.default_decay_rate = default_decay_rate
        self.period_type = period_type.lower()
        self.period_amount = period_amount
        self.period_seconds = self._get_period_seconds()

    def _get_period_seconds(self):
        """Convert period_type to seconds."""
        if self.period_type == "hour":
            return 3600
        elif self.period_type == "day":
            return 3600 * 24
        elif self.period_type == "week":
            return 3600 * 24 * 7
        else:
            raise ValueError(f"Unsupported period_type: {self.period_type}")

    def compute_decayed_scores(self, aligned_news, stock_dates):
        """
        Compute decayed sentiment scores with adaptive decay rates based on topics.

        Args:
            aligned_news (pd.DataFrame): News data with columns ["Time", "sentiment_score", "topic"].
            stock_dates (pd.Series): Trading dates to compute decay for.

        Returns:
            np.ndarray: Accumulated decayed sentiment scores for each stock date.
        """
        # Convert timestamps to numpy datetime64 (for vectorization)
        news_times = aligned_news["Time"].values.astype("datetime64[s]")
        stock_times = stock_dates.values.astype("datetime64[s]")

        # Compute time deltas (in seconds) between stock dates and news times
        delta_time = (stock_times[:, None] - news_times[None, :]).astype(float)

        # Check for invalid delta_time values
        if np.any(np.isnan(delta_time)):
            raise ValueError("NaN values detected in delta_time computation.")

        # Convert time deltas to periods (e.g., days/weeks)
        delta_periods = delta_time / (self.period_seconds * self.period_amount)

        # Check for invalid delta_periods
        if np.any(np.isnan(delta_periods)):
            raise ValueError("NaN values detected in delta_periods computation.")

        # Ensure topics are valid and do not contain NaN values
        if aligned_news["topic"].isna().any():
            raise ValueError("NaN values detected in aligned_news['topic']. Ensure all topics are valid.")

        # Determine decay rates based on topics
        topics = aligned_news["topic"].values
        logging.debug(f"Topics in aligned_news:\n{aligned_news['topic'].values}")
        logging.debug(f"topic_decay_rate_map keys:\n{self.topic_decay_rate_map.keys()}")

        decay_rates = np.array([
            self.topic_decay_rate_map.get(topic, self.default_decay_rate)
            for topic in topics
        ])

        # Check for invalid decay rates
        if np.any(np.isnan(decay_rates)):
            logging.debug(f"Decay rates with NaN values:\n{decay_rates}")
            logging.debug(f"Topics causing NaN values:\n{[topics[i] for i in range(len(decay_rates)) if np.isnan(decay_rates[i])]}")
            raise ValueError("NaN values detected in decay_rates. Check topic_decay_rate_map and input data.")

        # Expand decay_rates to match the shape of delta_periods
        decay_rates = decay_rates[None, :]  # Add a new axis for broadcasting

        # Apply exponential decay: S(t) = S0 * e^(-λ * t)
        decay_factors = np.exp(-decay_rates * delta_periods)

        # Debugging: Check decay factors
        logging.debug(f"Decay factors sample:\n{decay_factors[:5, :5]}")

        # Mask invalid times (news published after stock date)
        valid_mask = delta_time >= 0
        decay_factors[~valid_mask] = 0

        # Multiply by sentiment scores and sum across news articles
        sentiment_scores = aligned_news["sentiment_score"].values
        decayed_scores = (decay_factors * sentiment_scores).sum(axis=1)

        # Debugging: Check decayed scores
        logging.debug(f"Decayed scores sample:\n{decayed_scores[:5]}")

        # Check for NaN values in decayed_scores
        if np.any(np.isnan(decayed_scores)):
            logging.debug(f"delta_time:\n{delta_time}")
            logging.debug(f"delta_periods:\n{delta_periods}")
            logging.debug(f"decay_rates:\n{decay_rates}")
            logging.debug(f"decay_factors:\n{decay_factors}")
            logging.debug(f"sentiment_scores:\n{sentiment_scores}")
            raise ValueError("NaN values detected in decayed_scores. Check input data and decay parameters.")

        return decayed_scores
