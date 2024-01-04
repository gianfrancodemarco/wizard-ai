from bs4 import BeautifulSoup
from readability import Document


class HtmlProcessor:

    @staticmethod
    def clear_html(html):
        doc = Document(html)
        main_content = doc.summary()
        soup = BeautifulSoup(main_content, 'html.parser')
        for script in soup(['script', 'style']):
            script.decompose()
        return soup.get_text(separator='\n', strip=True)
