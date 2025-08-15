import os
import json
from typing import List
import chromadb
from utils.embedding import SentenceTransformerEmbeddingFunction, GeminiEmbeddingFunction, OpenAIEmbeddingFunction
from tqdm import tqdm   
import pickle
from utils.embedding import BM25EmbeddingFunction


class BM25DB:
    @staticmethod
    def create_bm25_db(documents: List[dict], path: str = "database/bm25_db.pkl", name: str = "bm25_collection", language: str = "vi"):
        doc_array = []
        full_documents = []
        bm25_embedding_fn = BM25EmbeddingFunction(language=language)
        for i in tqdm(range(0, len(documents)), desc=f"Creating BM25 collection: {name}"):
            doc = documents[i]
            doc_text = doc.get("text", "")
            tokens = bm25_embedding_fn.bm25_tokenizer(doc_text)
            doc_array.append(tokens)
            full_documents.append(doc)

        bm25_plus = bm25_embedding_fn.bm25_plus(doc_array)
        # Save the BM25 index
        with open(path, "wb") as f:
            pickle.dump({
                "bm25": bm25_plus,
                "full_documents": full_documents,
            }, f)

        return bm25_plus, full_documents

    @staticmethod
    def load_bm25_db(path: str = "bm25_db.pkl"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"BM25 database file not found at {path}")

        with open(path, "rb") as f:
            data = pickle.load(f)
            return data["bm25"], data["full_documents"]

# ChromaDB wrapper
class ChromaDB:
    @staticmethod
    def create_chroma_db(documents: List[dict], path: str, name: str, batch_size: int = 500, embedding_fn: str = "dek21-vn-law-embedding"):
        chroma_client = chromadb.PersistentClient(path=path)
        if embedding_fn == "truro7/vn-law-embedding":
            embedding_fn = SentenceTransformerEmbeddingFunction(model_name="truro7/vn-law-embedding")
        elif embedding_fn == "gemini":
            embedding_fn = GeminiEmbeddingFunction()
        elif embedding_fn == "dek21-vn-law-embedding":
            embedding_fn = SentenceTransformerEmbeddingFunction(model_name="dek21-vn-law-embedding")
        elif embedding_fn == "openai":  
            embedding_fn = OpenAIEmbeddingFunction()

        try:
            chroma_client.delete_collection(name)
        except:
            pass

        db = chroma_client.create_collection(name=name, embedding_function=embedding_fn)

        for i in tqdm(range(0, len(documents), batch_size), desc=f"Creating collection: {name}"):
            batch = documents[i:i + batch_size]
            db.add(
                documents=[item["text"] for item in batch],
                metadatas=[item["metadata"] for item in batch],
                ids=[item["id"] for item in batch]
            )
        return db

    @staticmethod
    def load_chroma_collection(path: str, name: str, embedding_fn: str = "vn-law-embedding"):
        if embedding_fn == "truro7/vn-law-embedding":
            embedding_fn = SentenceTransformerEmbeddingFunction(model_name="truro7/vn-law-embedding")
        elif embedding_fn == "gemini":
            embedding_fn = GeminiEmbeddingFunction()
        elif embedding_fn == "dek21-vn-law-embedding":
            embedding_fn = SentenceTransformerEmbeddingFunction(model_name="dek21-vn-law-embedding")
        elif embedding_fn == "openai":
            embedding_fn = OpenAIEmbeddingFunction()

        chroma_client = chromadb.PersistentClient(path=path)
        return chroma_client.get_collection(name=name, embedding_function=embedding_fn)