from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from utils.answer_generator_old import AnswerGenerator
from utils.db import ChromaDB, BM25DB
import os
import asyncio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/imgs", StaticFiles(directory="static/imgs"), name="imgs")

answer_generator = AnswerGenerator()


class Config_VN:
    language = "vi"
    embedding_fn = "dek21-vn-law-embedding"
    json_folder = "new-vn-data-json"
    text_db_chroma = ChromaDB.load_chroma_collection(
        "database/new-vn-law-chroma", name="text_docs", embedding_fn=embedding_fn)
    bm25_plus, full_documents_bm25 = BM25DB.load_bm25_db(
        "database/viet_bm25_db.pkl")


class Config_EN:
    language = "en"
    embedding_fn = "openai"
    json_folder = "new-eng-data-json"
    # text_db_chroma = ChromaDB.load_chroma_collection(
    #     "database/new-eng-law-chroma", name="text_docs", embedding_fn=embedding_fn)
    # bm25_plus, full_documents_bm25 = BM25DB.load_bm25_db(
    #     "database/eng_bm25_db.pkl")


class QueryRequest(BaseModel):
    question: str

vn_config = Config_VN()
en_config = Config_EN()

# ✅ Stream chunk by chunk

async def stream_answer(answer: str):
    for i in range(0, len(answer), 20):  # 20 chars per chunk
        yield answer[i:i+20]
        await asyncio.sleep(0.05)  # optional: slow down for demo


@app.post("/ask-viet")
async def ask_question(data: QueryRequest):
    # generate whole answer
    summary_answer, references = answer_generator.combined_answer(
        vn_config.text_db_chroma,
        vn_config.bm25_plus,
        vn_config.full_documents_bm25,
        query=data.question,
        json_folder=vn_config.json_folder,
        language=vn_config.language
    )
    return {"answer": summary_answer, "references": references}
    # return StreamingResponse(stream_answer(answer), media_type="text/plain")


@app.post("/ask-eng")
async def ask_question(data: QueryRequest):
    # generate whole answer
    answer = answer_generator.combined_answer(
        en_config.text_db_chroma,
        en_config.bm25_plus,
        en_config.full_documents_bm25,
        query=data.question,
        json_folder=en_config.json_folder,
        language=en_config.language
    )
    return {"answer": answer}
    # return StreamingResponse(stream_answer(answer), media_type="text/plain")

# ✅ Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# ✅ Serve index.html at root


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "index.html not found", 404
