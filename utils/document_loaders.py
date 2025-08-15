from typing import List
import os
import json
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentLoader:
    def __init__(self, pdf_folder: str, json_path: str):
        self.pdf_folder = pdf_folder
        self.json_path = json_path
        
    def load_documents(self) -> List[dict]:
        text_docs = self.load_text_documents(self.pdf_folder)
        image_docs = self.load_json_documents(self.json_path, "image")
        table_docs = self.load_json_documents(self.json_path, "table")

        return text_docs, image_docs, table_docs

    def load_text_documents(self, pdf_folder: str) -> List[dict]:
        text_docs = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=600,
            separators=["\n\n", "\n- ", "\n1.", "\n", " "],
        )
        
        for fname in os.listdir(pdf_folder):
            if fname.lower().endswith(".pdf"):
                pdf_path = os.path.join(pdf_folder, fname)
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()  # returns list of Langchain Document objects
                
                # documents usually have .page_content and .metadata
                for i, doc in enumerate(documents):
                    chunks = splitter.split_text(doc.page_content)
                    for j, chunk in enumerate(chunks):
                        text_docs.append({
                            "id": f"{fname.replace('.pdf', '')}_{i}_{j}",
                            "text": chunk,
                            "metadata": {
                                "type": "text",
                                "filename": fname,
                                "page": i,
                                "chunk": j
                            }
                        })
        return text_docs

    # Load image metadata JSON
    def load_json_documents(self, json_path: str, doc_type: str) -> List[dict]:
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [{
            "id": entry["id"],
            "text": entry.get("content", "No Caption"),
            "metadata": {
                "type": doc_type,
                "url": entry.get("url", ""),
            }
        } for entry in raw]
