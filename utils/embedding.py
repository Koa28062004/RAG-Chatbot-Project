from typing import List
import os
import json
from tqdm import tqdm
import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import string
from underthesea import word_tokenize
from rank_bm25 import BM25Plus
from openai import OpenAI

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = OpenAI()

number = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
chars = ["a", "b", "c", "d", "đ", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "-", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
import string
from typing import List
from rank_bm25 import BM25Plus

from underthesea import word_tokenize as vi_tokenize  # Vietnamese
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
import nltk

en_tokenize = TreebankWordTokenizer().tokenize

# Your custom Vietnamese stopword list
vi_stopwords = number + chars + [
    "của", "và", "các", "có", "được", "theo", "tại", "trong", "về",
    "hoặc", "người", "này", "khoản", "cho", "không", "từ", "phải",
    "ngày", "việc", "sau", "để", "đến", "bộ", "với", "là", "năm",
    "khi", "số", "trên", "khác", "đã", "thì", "thuộc", "điểm", "đồng",
    "do", "một", "bị", "vào", "lại", "ở", "nếu", "làm", "đây",
    "như", "đó", "mà", "nơi", "”", "“"
]

# English stopwords from nltk
en_stopwords = set(stopwords.words('english'))

class BM25EmbeddingFunction(EmbeddingFunction):
    def __init__(self, language="vi"):
        self.top_k_bm25 = 2
        self.bm25_k1 = 0.4
        self.bm25_b = 0.6
        self.language = language.lower()
        
    def __remove_stopword(self, w):
        if self.language == "vi":
            return w not in vi_stopwords
        else:
            return w not in en_stopwords
    
    def __remove_punctuation(self, w):
        return w not in string.punctuation
    
    def __lower_case(self, w):
        return w.lower()
    
    def bm25_tokenizer(self, text):
        # Use appropriate tokenizer
        if self.language == "vi":
            tokens = vi_tokenize(text)
        else:
            tokens = en_tokenize(text)
        # Common cleaning
        tokens = list(map(self.__lower_case, tokens))
        tokens = list(filter(self.__remove_punctuation, tokens))
        tokens = list(filter(self.__remove_stopword, tokens))
        return tokens  
    
    def bm25_plus(self, tokenized_docs: List[List[str]]):
        # print(f"type:", type(documents))
        # tokenized_docs = [self.bm25_tokenizer(doc) for doc in documents]
        bm25 = BM25Plus(tokenized_docs, k1=self.bm25_k1, b=self.bm25_b)
        return bm25

# SentenceTransformer embedding function
class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str = "truro7/vn-law-embedding"):
        if model_name == "dek21-vn-law-embedding":
            self.model = SentenceTransformer("huyydangg/DEk21_hcmute_embedding", truncate_dim=128)
        else:
            self.model = SentenceTransformer("truro7/vn-law-embedding", truncate_dim=128)

    def __call__(self, inputs: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(inputs, convert_to_tensor=True)
        return embeddings.tolist()

# Gemini embedding function
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, inputs: List[str]) -> List[List[float]]:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        genai.configure(api_key=GEMINI_API_KEY)
        model = "models/text-embedding-004"
        response = genai.embed_content(
            model=model,
            content=inputs,
            task_type="retrieval_document",
            title="Document chunk"
        )
        return response["embedding"]
    
class OpenAIEmbeddingFunction(EmbeddingFunction):
    def __call__(self, inputs: List[str]) -> List[List[float]]:
        if not client.api_key:
            raise ValueError("OpenAI API key not set.")
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=inputs
        )
        embeddings = [item.embedding for item in response.data]
        return embeddings