from abc import ABC, abstractmethod
from typing import List, Dict, Any


class NewsCrawler(ABC):
    """Abstract base class for news crawlers."""

    @abstractmethod
    def get_headers(self) -> Dict[str, str]:
        """
        Get the headers required for HTTP requests.

        Returns:
            Dict[str, str]: A dictionary of HTTP headers.
        """
        pass

    @abstractmethod
    async def fetch(self, session, url: str) -> Dict[str, Any]:
        """Fetch JSON data from a given URL.

        Args:
            session: The aiohttp session to use for the request.
            url (str): The URL to fetch data from.

        Returns:
            Dict[str, Any]: The JSON response as a dictionary.
        """
        pass

    @abstractmethod
    async def fetch_articles(
        self, section_ids: List[str], max_offset: int, page_size: int
    ) -> List[Dict[str, Any]]:
        """Fetch articles from multiple sections asynchronously.

        Args:
            section_ids (List[str]): List of section IDs to fetch articles from.
            max_offset (int): Maximum offset for pagination.
            page_size (int): Number of articles per page.

        Returns:
            List[Dict[str, Any]]: A list of JSON responses containing article data.
        """
        pass

    @abstractmethod
    def extract_article_data(
        self, articles: List[Dict[str, Any]], section_id: str
    ) -> List[Dict[str, Any]]:
        """Extract and transform article data into a structured format.

        Args:
            articles (List[Dict[str, Any]]): List of articles in JSON format.
            section_id (str): The section ID associated with the articles.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing transformed article data.
        """
        pass
