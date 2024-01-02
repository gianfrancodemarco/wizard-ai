from collections import defaultdict
from urllib.parse import quote
from httpx import Client
from parsel import Selector

# 1. Create HTTP client with headers that look like a real web browser
client = Client(
    headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9,lt;q=0.8,et;q=0.7,de;q=0.6",
    },
    follow_redirects=True,
    http2=True,  # use HTTP/2 
)


def parse_search_results(selector: Selector):
    """parse search results from google search page"""
    results = []
    for box in selector.xpath("//h1[contains(text(),'Search Results')]/following-sibling::div[1]/div"):
        title = box.xpath(".//h3/text()").get()
        url = box.xpath(".//h3/../@href").get()
        text = "".join(box.xpath(".//div[@data-sncf]//text()").getall())
        if not title or not url:
            continue
        url = url.split("://")[1].replace("www.", "")
        results.append(title, url, text)
    return results


def scrape_search(query: str, page=1):
    """scrape search results for a given keyword"""
    # retrieve the SERP
    url = f"https://www.google.com/search?hl=en&q={quote(query)}" + (f"&start={10*(page-1)}" if page > 1 else "")
    print(f"scraping {query=} {page=}")
    results = defaultdict(list)
    response = client.get(url)
    assert response.status_code == 200, f"failed status_code={response.status_code}"
    # parse SERP for search result data
    selector = Selector(response.text)
    results["search"].extend(parse_search_results(selector))
    return dict(results)

# example use: scrape 3 pages: 1,2,3
for page in [1, 2, 3]:
    results = scrape_search("scrapfly blog", page=page)
    for result in results["search"]:
        print(result)
