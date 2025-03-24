import requests
import pandas as pd
from user_agent import generate_user_agent
import asyncio
import aiohttp
from aiohttp import ClientSession
from aiohttp.client_exceptions import ClientError
from tenacity import retry, wait_fixed, stop_after_attempt
import json
import urllib.parse

cookie = 'cleared-onetrust-cookies=Thu, 17 Feb 2022 19:17:07 GMT; usprivacy=1---; _lc2_fpi=f511229f0ef8--01jnaz0dpabwth3wedprhhfbe0; _gcl_au=1.1.1373562176.1740903561; permutive-id=40c79d01-2af0-4a43-b1f1-e036981076ca; _fbp=fb.1.1740903561134.887444212517492666; _cb=B3zXvQy-DFICM604F; _scor_uid=e4a1050ae4204e3abceeaf0cf391eb94; _cc_id=88a58455010c9407b15c4748bc701da7; panoramaId_expiry=1741508375669; panoramaId=202dc570fd68d0372e7e1194b94616d539386696998ee50367a5a8137b6c81e2; panoramaIdType=panoIndiv; ajs_anonymous_id=513fc99a-13db-43e2-b228-57ca1ba1b522; _ga=GA1.2.1955516653.1740903562; OneTrustWPCCPAGoogleOptOut=false; _li_dcdm_c=.reuters.com; _lc2_fpi_js=f511229f0ef8--01jnaz0dpabwth3wedprhhfbe0; dicbo_id=%7B%22dicbo_fetch%22%3A1741113130671%7D; _cb_svref=https%3A%2F%2Fwww.bing.com%2F; _parsely_session={%22sid%22:2%2C%22surl%22:%22https://www.reuters.com/business/energy/bailed-out-uniper-repay-27-bln-state-aid-first-quarter-2025-02-25/%22%2C%22sref%22:%22https://www.bing.com/%22%2C%22sts%22:1741113131045%2C%22slts%22:1740903561947}; _parsely_visitor={%22id%22:%22pid=b114016d-25c9-4698-9e64-3377f5ac5b13%22%2C%22session_count%22:2%2C%22last_session_ts%22:1741113131045}; _li_ss=CmgKBgj5ARCZGgoFCAoQmRoKBgikARCbGgoGCN0BEJkaCgUICRCbGgoGCOEBEJsaCgYIgQEQmRoKBgiiARCZGgoJCP____8HEKUaCgYIiQEQmxoKBgilARCZGgoGCNIBEJkaCgUIfhCbGg; __gads=ID=bda0fcb33fa49620:T=1740903576:RT=1741113148:S=ALNI_MahBEHD-x4l5l3EQWrmLM8qMQkIWA; __gpi=UID=000010510b2554c9:T=1740903576:RT=1741113148:S=ALNI_MYlV1Vg016hSsy4J-1SZM56IDFWMA; __eoi=ID=c2d587b42258c9bd:T=1740903576:RT=1741113148:S=AA-AfjYXhxCIaKAYIkAK25_yPzKR; _gid=GA1.2.1594627787.1741113132; RT="z=1&dm=reuters.com&si=wdpf1ww4jpq&ss=m7rd13xf&sl=0&tt=0"; OptanonConsent=isGpcEnabled=0&datestamp=Wed+Mar+05+2025+01%3A32%3A59+GMT%2B0700+(Indochina+Time)&version=202501.1.0&browserGpcFlag=0&isIABGlobal=false&hosts=&consentId=18ce69e9-d5e5-4a7d-8cfe-96cbdaff40ec&interactionCount=0&isAnonUser=1&landingPath=NotLandingPage&groups=1%3A1%2C3%3A1%2CSSPD_BG%3A1%2C4%3A1%2C2%3A1&AwaitingReconsent=false; ABTasty=uid=s325nzw9ks8eb798; ABTastySession=mrasn=&lp=https%253A%252F%252Fwww.reuters.com%252Fbusiness%252Fenergy%252Fbailed-out-uniper-repay-27-bln-state-aid-first-quarter-2025-02-25%252F; _awl=2.1741113196.5-bdf006192533ebc6b882ca5771492231-6763652d617369612d6561737431-0; _chartbeat2=.1740903561905.1741113180397.1001.-2AIDBDkquTC0c7UoCOhZeECSOrSf.2; cto_bundle=lpL_3l9pSndpdHBPcUdPT2cxenZWaG5uWllYaGgwdWozV0NPeUFpU1Z0Y1VkV091SmVZJTJGNlpXRjN5S1RQVmxkdFBkZnRwYUw5V2xBeGZTTjZreEt3QkdUJTJCZDJRMUpDczJNTVFjV0JKMVhKaWJTeFNWZ1JxMVVlRDNOSHV3TUJZSUpkJTJGZUliM2FmR2dwJTJCNGpabE1mT0pCVzdBdyUzRCUzRA; datadome=fBMWjZ0mH1YQEDAFW2P0ax1Ft2lyD01UHw2gqBBOriQkj5xYSv3wPIIqCoqUJm7GK2~7agfrTh1I~jTeJlwzKx7Lii_WaSP7rcdkPVpPCBuDjXdT0uCWZX2X2LsZW1N4; reuters-geo={"country":"-", "region":"-"}; _gat=1; _dd_s=rum=0&expire=1741114236279'
headers = {
    "Referer": "https://www.reuters.com/business/finance/",
    "User-Agent": generate_user_agent(device_type="desktop", os=("mac", "linux")),
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
}

@retry(wait=wait_fixed(5), stop=stop_after_attempt(3))
async def fetch(session: ClientSession, url: str):
    async with session.get(url, headers=headers) as response:
        response.raise_for_status()
        return await response.json()

async def fetch_articles(section_ids, max_offset=200, page_size=9):
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
        tasks = [fetch(session, url) for url in urls]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        return [res for res in responses if not isinstance(res, Exception)]

def extract_article_data(articles, section_id):
    """Extract and transform article data with pandas-friendly formatting"""
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

async def main():
    section_ids = ["/markets/us/", "/business/finance/"] 
    max_offset = 200
    page_size = 9

    articles = []
    for section_id in section_ids:
        responses = await fetch_articles([section_id], max_offset, page_size)
        for response in responses:
            articles.extend(extract_article_data(response.get('result', {}).get('articles', []), section_id))

    df = pd.DataFrame(articles).pipe(lambda d: d.astype({
        'word_count': 'float32',
        'company_rics': 'string',
        'ad_topics': 'string'
    }))

    # Explode list columns
    list_columns = ['company_rics', 'ad_topics']
    for col in list_columns:
        df[col] = df[col].str.join(',')

    output_file = "financial_news_data.csv"

    df.to_csv(
        output_file,
        index=False,
        encoding='utf-8', 
        date_format='%Y-%m-%dT%H:%M:%S.%fZ' 
    )

if __name__ == "__main__":
    asyncio.run(main())
