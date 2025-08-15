import requests
import trafilatura
import google.generativeai as genai
from dotenv import load_dotenv
import os
import time
from unidecode import unidecode
import requests
from bs4 import BeautifulSoup
import urllib.parse
import sys
from config import Config

class WebSraper:
    def __init__(self, num_results=5):
        self.num_results = num_results
        self.config = Config()

    def search_serper(self, query, max_retries=1):
        url = "https://google.serper.dev/search"
        headers = self.config.get_serper_api_headers()
        data = {"q": query}
        
        for attempt in range(max_retries):
            try:
                res = requests.post(url, headers=headers, json=data)
                res.raise_for_status()
                results = res.json()
                if "organic" in results and results["organic"]:
                    return [{"url": r["link"], "title": r["title"]} for r in results["organic"][:num_results]]
                else:
                    print(f"⚠️ No organic results on attempt {attempt+1}. Retrying...")
            except Exception as e:
                print(f"❌ Error during search (attempt {attempt+1}): {e}")
            time.sleep(1)
        
        return []
    
    def search_duckduckgo(self, query):
        params = {'q': query}
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get('https://html.duckduckgo.com/html/', params=params, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.find_all('a', class_='result__a')
        
        entries = []
        cnt = 0

        for link in results:
            if cnt >= 5:
                break
            cnt += 1
            raw_url = link['href']
            parsed = urllib.parse.urlparse(raw_url)
            query_params = urllib.parse.parse_qs(parsed.query)
            decoded_url = query_params.get("uddg", [raw_url])[0]
            entry = {
                "title": link.text.strip(),
                "url": decoded_url
            }
            entries.append(entry)

        return entries
    
    def search_openai(self, query):
        prompt = f"""
        Bạn là một trợ lý ảo chăm sóc khách hàng.

        Dưới đây là câu hỏi từ khách hàng:
        {query}
        Yêu cầu của bạn là tìm kiếm thông tin trên Internet và trả lời câu hỏi một cách chi tiết, đúng trọng tâm.
        Hãy cung cấp cả liên kết đến các nguồn thông tin bạn sử dụng để trả lời.
        Nếu không tìm thấy thông tin liên quan, hãy trả lời: "Tôi xin lỗi, hiện tại tôi chưa tìm thấy thông tin liên quan trong tài liệu. Bạn vui lòng liên hệ chuyên viên để được hỗ trợ thêm."

        """
        client = self.config.get_openai_api_client()
        completion = client.chat.completions.create(
            model="gpt-4o-search-preview",
            web_search_options={
                "user_location": {
                    "type": "approximate",
                    "approximate": {
                        "country": "VN",
                        "city": "Ho Chi Minh City",      
                        "region": "Ho Chi Minh City",    
                    }
                },
            },
            messages=[{
                "role": "user",
                "content": prompt
            }],
        )
        
        return completion.choices[0].message.content

class ArticleExtractor:
    def extract_article_with_source(self, entry):
        url = entry["url"]
        title = entry["title"]
        downloaded = trafilatura.fetch_url(url)
        content = trafilatura.extract(downloaded)
        if content:
            return {
                "title": title,
                "url": url,
                "content": content
            }
        return None

class Summarizer:
    def __init__(self):
        self.config = Config()
        self.model = self.config.get_gemini_api_model()

    def summarize_articles(self, articles, query):
        combined = ""
        for art in articles:
            combined += f"""Tiêu đề: {art['title']}
    URL: {art['url']}

    {art['content']}

    """

        prompt = f"""
    Bạn là một trợ lý ảo chăm sóc khách hàng, chuyên trả lời các câu hỏi dựa trên tài liệu kỹ thuật hoặc quy định pháp luật được cung cấp.

    Dưới đây là một số tài liệu tham khảo được thu thập từ các nguồn uy tín trên Internet:

    {combined}

    Câu hỏi của khách hàng:
    {query}

    Yêu cầu đối với câu trả lời:
    1. Trả lời trực tiếp và chính xác câu hỏi của khách hàng bằng tiếng Việt dễ hiểu.
    2. Giải thích lý do vì sao bạn đưa ra câu trả lời đó, dựa trên nội dung đã được cung cấp. Không sử dụng cách đánh số như "Đoạn 1", "Đoạn 2" v.v.
    3. Trích dẫn lại thông tin có liên quan (bao gồm nội dung và URL) để chứng minh cho câu trả lời.
    4. Nếu không tìm thấy thông tin phù hợp, hãy trả lời: "Tôi xin lỗi, hiện tại tôi chưa tìm thấy thông tin liên quan trong tài liệu. Bạn vui lòng liên hệ chuyên viên để được hỗ trợ thêm."
    5. Nếu khách hàng yêu cầu gặp chuyên viên (ví dụ: "Tôi muốn gặp chuyên viên"), hãy trả lời: "Tôi sẽ gửi yêu cầu của bạn đến chuyên viên chăm sóc khách hàng. Họ sẽ liên hệ với bạn trong thời gian sớm nhất."

    Luôn giữ thái độ lịch sự, chuyên nghiệp và trả lời rõ ràng, mạch lạc.
    """
        
        response = self.model.generate_content(prompt)
        return response.text

class QueryProcessor:
    def __init__(self, searcher=WebSraper(), extractor=ArticleExtractor(), summarizer=Summarizer()):
        self.searcher = searcher
        self.extractor = extractor
        self.summarizer = summarizer

    def remove_accents(self, text):
        return unidecode(text)

    def search_and_answer(self, query, search_engine="openai"):
        print("🔍 Searching:", query)
        results = None

        if search_engine == "openai":
            print("Search with OpenAI...")
            results = self.searcher.search_openai(query)
            return results
        elif search_engine == "serper":
            print("Search with Serper...")
            results = self.searcher.search_serper(query)

            if not results:
                query_no_accents = self.remove_accents(query)
                print("🔍 Retry with non-accented query:", query_no_accents)
                results = self.searcher.search_serper(query_no_accents)
        elif search_engine == "duckduckgo":
            print("Search with DuckDuckGo...")
            results = self.searcher.search_duckduckgo(query)

        if not results:
            return "⚠️ Không tìm thấy kết quả tìm kiếm phù hợp."

        print("🔗 Found URLs:")
        for r in results:
            print(f"{r['title']} - {r['url']}")

        articles = [self.extractor.extract_article_with_source(r) for r in results]
        articles = [a for a in articles if a and a["content"]]

        if not articles:
            return "⚠️ Không thể trích xuất nội dung từ các bài viết."

        summary = self.summarizer.summarize_articles(articles, query)
        return summary
