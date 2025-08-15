from dotenv import load_dotenv
import os
import google.generativeai as genai
from openai import OpenAI

class Config:
    def __init__(self):
        load_dotenv()
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        self.SERPER_API_KEY = os.getenv("SERPER_API_KEY")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.MODEL_USED = os.getenv("MODEL_USED", "gemini-2.0-flash")
    
    def get_gemini_api_model(self):
        genai.configure(api_key=self.GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash")
        return model
    
    def get_serper_api_headers(self):
        headers = {"X-API-KEY": self.SERPER_API_KEY}
        return headers
    
    def get_openai_api_client(self):
        client = OpenAI(api_key=self.OPENAI_API_KEY)
        return client
    
    def get_model_used(self):
        if self.MODEL_USED == "gemini":
            return self.get_gemini_api_model()
        elif self.MODEL_USED == "openai":
            return self.get_openai_api_client()
        return self.MODEL_USED