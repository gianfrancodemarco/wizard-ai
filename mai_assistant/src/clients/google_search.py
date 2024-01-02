from typing import Dict, List

import requests
from bs4 import BeautifulSoup
from parsel import Selector
from pydantic import BaseModel
from readability import Document


class GoogleSearchClientPayload(BaseModel):
    query: str
    #page: int = 1
    num_expanded_results: int = 1


class GoogleSearchClient:
    def __init__(self):
        self.headers = {
            'authority': 'www.google.com',
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'accept-language': 'en-US,en;q=0.9,lt;q=0.8,et;q=0.7,de;q=0.6',
            'cache-control': 'no-cache',
            'dnt': '1',
            'pragma': 'no-cache',
            'sec-ch-ua': '"Not_A Brand";v="99", "Microsoft Edge";v="109", "Chromium";v="109"',
            'sec-ch-ua-arch': '"x86"',
            'sec-ch-ua-bitness': '"64"',
            'sec-ch-ua-full-version': '"109.0.1518.78"',
            'sec-ch-ua-full-version-list': '"Not_A Brand";v="99.0.0.0", "Microsoft Edge";v="109.0.1518.78", "Chromium";v="109.0.5414.120"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-model': '""',
            'sec-ch-ua-platform': '"Windows"',
            'sec-ch-ua-platform-version': '"10.0.0"',
            'sec-ch-ua-wow64': '?0',
            'sec-fetch-dest': 'document',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-site': 'none',
            'sec-fetch-user': '?1',
            'upgrade-insecure-requests': '1',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36 Edg/109.0.1518.78',
        }

    def make_request(self, url, params=None):
        params = params or {}
        try:
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            print(f"Error making request to {url}: {e}")
            return None

    def parse_search_results(self, selector: Selector) -> List[Dict[str, str]]:
        # Add parser for Financial data
        # Add parser for other data...

        parsed = []
        results = selector.xpath("//div[@id='rso']/*")
        for result in results:
            try:
                result_element = result.xpath(".//a[1]")
                result_url = result_element[0].xpath("@href").extract_first()
                result_title = result_element[0].xpath(
                    ".//h3[1]//text()").extract()

                parsed.append({
                    "url": result_url,
                    "title": result_title
                })
            except:
                pass
        return parsed

    def get_main_content_from_url(
        self,
        url: str
    ):
        try:
            response = self.make_request(url)
            if response:
                doc = Document(response.text)
                main_content = doc.summary()
                soup = BeautifulSoup(main_content, 'html.parser')
                for script in soup(['script', 'style']):
                    script.decompose()
                text_content = soup.get_text(separator='\n', strip=True)
                return text_content
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None

    def search(
        self,
        payload: GoogleSearchClientPayload
    ) -> List[str]:
        params = {'q': payload.query}

        try:
            response = self.make_request(
                'https://www.google.com/search', params=params)

            if response:
                selector = Selector(response.text)
                results = self.parse_search_results(
                    selector)
                
                texts = []
                for result in results:
                    if len(texts) < payload.num_expanded_results:                        
                        text = self.get_main_content_from_url(result["url"])
                        if text:
                            texts.append(text)

                return "\n\n".join(texts)
        except Exception as e:
            print(f"Error during search: {e}")
