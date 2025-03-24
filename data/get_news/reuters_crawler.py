import json
import urllib.parse
import pandas as pd
import asyncio
from typing import List, Dict, Any
from aiohttp import ClientSession
from tenacity import retry, wait_fixed, stop_after_attempt
from user_agent import generate_user_agent
from .base_crawler import NewsCrawler

class ReutersCrawler(NewsCrawler):
    """Implementation of NewsCrawler for Reuters."""

    def get_headers(self) -> Dict[str, str]:
        """Get the headers required for Reuters HTTP requests.

        Returns:
            Dict[str, str]: A dictionary of HTTP headers.
        """
        return {
            "Referer": "https://www.reuters.com/business/finance/",
            "User-Agent": generate_user_agent(device_type="desktop", os=("mac", "linux")),
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-US,en;q=0.9",
        }

    @retry(wait=wait_fixed(5), stop=stop_after_attempt(3))
    async def fetch(self, session: ClientSession, url: str) -> Dict[str, Any]:
        """Fetch JSON data from a given URL using an aiohttp session.

        Args:
            session (ClientSession): The aiohttp session to use for the request.
            url (str): The URL to fetch data from.

        Returns:
            Dict[str, Any]: The JSON response as a dictionary.

        Raises:
            aiohttp.ClientError: If the request fails.
        """
        async with session.get(url, headers=self.get_headers()) as response:
            response.raise_for_status()
            return await response.json()

    async def fetch_articles(self, section_ids: List[str], max_offset: int = 200, page_size: int = 9) -> List[Dict[str, Any]]:
        """Fetch articles from multiple sections asynchronously.

        Args:
            section_ids (List[str]): List of section IDs to fetch articles from.
            max_offset (int, optional): Maximum offset for pagination. Defaults to 200.
            page_size (int, optional): Number of articles per page. Defaults to 9.

        Returns:
            List[Dict[str, Any]]: A list of JSON responses containing article data.
        """
        urls = []
        for section_id in section_ids:
            for offset in range(1, max_offset + 1, page_size):
                params_dict = {
                    "arc-site": "reuters",
                    "fetch_type": "collection",
                    "offset": offset,
                    "section_id": section_id,
                    "size": page_size,
                    "website": "reuters"
                }
                json_query = json.dumps(params_dict)
                encoded_query = urllib.parse.quote(json_query)
                url = f"https://www.reuters.com/pf/api/v3/content/fetch/articles-by-section-alias-or-id-v1?query={encoded_query}&mxId=00000000&_website=reuters"
                urls.append(url)

        async with ClientSession() as session:
            tasks = [self.fetch(session, url) for url in urls]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            return [res for res in responses if not isinstance(res, Exception)]

    def extract_article_data(self, articles: List[Dict[str, Any]], section_id: str) -> List[Dict[str, Any]]:
        """Extract and transform article data into a pandas-friendly format.

        Args:
            articles (List[Dict[str, Any]]): List of articles in JSON format.
            section_id (str): The section ID associated with the articles.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing transformed article data.
        """
        data = []
        for article in articles:
            entry = {
                'id': article.get('id'),
                'published_time': pd.to_datetime(article.get('published_time')), 
                'title': article.get('title'),
                'description': article.get('description'),
                'company_rics': article.get('company_rics', []),
                'kicker': article.get('kicker', {}).get('name'),
                'word_count': pd.to_numeric(article.get('word_count')), 
                'source': article.get('source', {}).get('name'),
                'ad_topics': article.get('ad_topics', []),
                'authors': '; '.join([a.get('name') for a in article.get('authors', [])]),
                'article_type': article.get('article_type'),
                'distributor': article.get('distributor'),
                'update_time': pd.to_datetime(article.get('updated_time')),
                'section_id': section_id 
            }
            data.append(entry)
        return data
