from utils.scan_pdf_to_md import convert_pdf_to_markdown, extract_images_and_caption_flexible, extract_tables_from_markdown
import os
import json
from utils.db import ChromaDB, BM25DB
from utils.document_loaders import DocumentLoader
from utils.answer_generator_old import AnswerGenerator
from typing import List
import uuid
# from utils.embedding import BM25EmbeddingFunction, SentenceTransformerEmbeddingFunction
import numpy as np
from pyvi import ViTokenizer
import re

def response_data(text_db_chroma, bm25_plus, full_documents_bm25, image_db=None, table_db=None, json_folder=None, language=None):
    """
    Generate responses based on the loaded data.
    """
    # question = "hồ sơ báo cáo nghiên cứu khả thi gồm những gì"
    # question = "hành lang bên là gì"
    question = "quy định cấp điện từ hai nguồn độc lập"
    answerGenerator = AnswerGenerator()
    summary_answer, references = answerGenerator.combined_answer(
        text_db_chroma, bm25_plus, full_documents_bm25, query=question, json_folder=json_folder, language=language)

    print(f"\n\nQuestion: {question}")
    print(f"\n\nAnswer:\n\n {summary_answer}")
    print(f"\n\nReferences:\n\n {references}")


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


def make_id(doc_name, path, extra):
    return f"{doc_name}_{path}_{extra}_{uuid.uuid4().hex[:8]}".replace(" ", "_")


def split_text(text: str, max_len: int = 400, overlap: int = 100) -> List[str]:
    """
    Split text into chunks (approx ~ max_len chars), keeping whole sentences or words,
    and maintaining overlap between chunks without cutting off words.

    :param text: Input text to split
    :param max_len: Maximum length per chunk (in characters)
    :param overlap: Number of characters to overlap between chunks
    :return: List of text chunks
    """
    # Use sentence tokenizer if available
    try:
        sentences = nltk.sent_tokenize(text)
    except:
        # fallback: split by [.?!] with optional whitespace
        sentences = re.split(r'(?<=[.?!])\s+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_len:
            current_chunk += (" " if current_chunk else "") + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Add overlap
    final_chunks = []
    for i in range(len(chunks)):
        chunk = chunks[i]
        if i > 0 and overlap > 0:
            prev = final_chunks[-1]
            overlap_text = prev[-overlap:]
            chunk = overlap_text + " " + chunk
        final_chunks.append(chunk.strip())

    return final_chunks

def process_json_folder_level_4(json_folder: str) -> List[dict]:
    text_docs = []

    def walk_node(node, path_titles, doc_name, current_section, depth):
        lines = []
        has_subsection = False

        for item in node.get("content", []):
            if "type" in item:
                if item["type"] == "text":
                    lines.append(item["data"])
                elif item["type"] == "image":
                    lines.append(item["image_name"])

            elif "subsection" in item:
                has_subsection = True
                title = item["subsection"]

                # At depth == 1 → we're at level 2 → next is level 3
                if depth == 2:
                    current_section = title  # level-3 section title

                lines.append(title)

                walk_node(
                    item,
                    path_titles + [title],
                    doc_name,
                    current_section=current_section,  # pass down level-3 section
                    depth=depth + 1
                )

        # Save only if it's a leaf node (no deeper subsections)
        if lines and not has_subsection and current_section:
            full_path = "_".join(path_titles)
            joined = "\n".join(lines)
            joined_chunks = split_text(joined)
            heading = "\n".join(path_titles)

            for chunk_idx, chunk in enumerate(joined_chunks):
                uid = make_id(doc_name, full_path, f"joined_chunk_{chunk_idx}")
                text_docs.append({
                    "id": uid,
                    "text": f"{heading}\n{chunk}",
                    "metadata": {
                        "type": "text",
                        "doc_name": doc_name,
                        "section": current_section  # level-3 section only
                    }
                })

    for fname in os.listdir(json_folder):
        if fname.lower().endswith(".json"):
            json_path = os.path.join(json_folder, fname)
            print(f"Processing JSON file: {json_path}")

            with open(json_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            doc_name = raw.get("doc_name", fname)
            print(f"Document Name: {doc_name}")

            for section in raw.get("sections", []):
                section_title = section.get("section", "No Title")
                walk_node(
                    section,
                    [doc_name, section_title],
                    doc_name,
                    current_section=None,
                    depth=0
                )

    return text_docs

def process_json_folder_level_3(json_folder: str) -> List[dict]:
    text_docs = []

    def walk_node(node, path_titles, doc_name, current_section, depth):
        lines = []
        has_subsection = False

        for item in node.get("content", []):
            if "type" in item:
                if item["type"] == "text":
                    lines.append(item["data"])
                elif item["type"] == "image":
                    lines.append(item["image_name"])

            elif "subsection" in item:
                has_subsection = True
                title = item["subsection"]

                # At depth == 1 → we're at level 2 → next is level 3
                if depth == 1:
                    current_section = title  # level-3 section title

                lines.append(title)

                walk_node(
                    item,
                    path_titles + [title],
                    doc_name,
                    current_section=current_section,  # pass down level-3 section
                    depth=depth + 1
                )

        # Save only if it's a leaf node (no deeper subsections)
        if lines and not has_subsection and current_section:
            full_path = "_".join(path_titles)
            joined = "\n".join(lines)
            joined_chunks = split_text(joined)
            heading = "\n".join(path_titles)

            for chunk_idx, chunk in enumerate(joined_chunks):
                uid = make_id(doc_name, full_path, f"joined_chunk_{chunk_idx}")
                text_docs.append({
                    "id": uid,
                    "text": f"{heading}\n{chunk}",
                    "metadata": {
                        "type": "text",
                        "doc_name": doc_name,
                        "section": current_section  # level-3 section only
                    }
                })

    for fname in os.listdir(json_folder):
        if fname.lower().endswith(".json"):
            json_path = os.path.join(json_folder, fname)
            print(f"Processing JSON file: {json_path}")

            with open(json_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            doc_name = raw.get("doc_name", fname)
            print(f"Document Name: {doc_name}")

            for section in raw.get("sections", []):
                section_title = section.get("section", "No Title")
                walk_node(
                    section,
                    [doc_name, section_title],
                    doc_name,
                    current_section=None,
                    depth=0
                )

    return text_docs

# Level 2
def process_json_folder(json_folder: str) -> List[dict]:
    text_docs = []

    def walk_node(node, path_titles, doc_name, current_section, depth):
        lines = []
        has_subsection = False
        next_section = current_section

        for item in node.get("content", []):
            if "type" in item:
                if item["type"] == "text":
                    lines.append(item["data"])
                elif item["type"] == "image":
                    lines.append(item["image_name"])

            elif "subsection" in item:
                has_subsection = True
                title = item["subsection"]
                next_path_titles = path_titles + [title]

                # ✅ If current depth is 0 (section level), the next depth is 1 (level 2)
                # If we are at level 1, the subsection becomes the new current_section
                deeper_section = title if depth == 0 else current_section

                walk_node(
                    item,
                    next_path_titles,
                    doc_name,
                    current_section=deeper_section,
                    depth=depth + 1
                )

        # ✅ Only add content if it's leaf and has no more subsections
        if lines and not has_subsection:
            full_path = "_".join(path_titles)
            joined = "\n".join(lines)
            joined_chunks = split_text(joined)

            # Show heading = full title path after doc_name (skip doc_name)
            heading = "\n".join(path_titles[1:]) if len(path_titles) > 1 else path_titles[0]

            for chunk_idx, chunk in enumerate(joined_chunks):
                uid = make_id(doc_name, full_path, f"joined_chunk_{chunk_idx}")
                text_docs.append({
                    "id": uid,
                    "text": f"{heading}\n{chunk}",
                    "metadata": {
                        "type": "text",
                        "doc_name": doc_name,
                        "section": current_section  # ✅ will be level 2 or fallback to level 1
                    }
                })

    # === Process each file ===
    for fname in os.listdir(json_folder):
        if fname.lower().endswith(".json"):
            json_path = os.path.join(json_folder, fname)
            print(f"Processing JSON file: {json_path}")

            with open(json_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            doc_name = raw.get("doc_name", fname)
            print(f"Document Name: {doc_name}")

            for section in raw.get("sections", []):
                section_title = section.get("section", "No Title")
                # ✅ Start with top-level section title
                walk_node(
                    section,
                    [doc_name, section_title],
                    doc_name,
                    current_section=section_title,
                    depth=0
                )

    return text_docs

if __name__ == "__main__":
    json_folder = "new-vn-data-json"
    embedding_fn = "dek21-vn-law-embedding"
    bm25_path = "database/viet_bm25_db.pkl"
    chroma_db_path = "database/new-vn-law-chroma"
    language = "vi"  # "en" or "vi"

    # text_docs_raw = process_json_folder(json_folder)
    # text_docs_level_3_raw = process_json_folder_level_3(json_folder)
    # text_docs_level_4_raw = process_json_folder_level_4(json_folder)

    # with open("text_docs_raw_level_3.json", "w", encoding="utf-8") as f:
    #     json.dump(text_docs_level_3_raw, f, ensure_ascii=False, indent=4)

    # with open("text_docs_raw_level_4.json", "w", encoding="utf-8") as f:
    #     json.dump(text_docs_level_4_raw, f, ensure_ascii=False, indent=4)

    # with open("text_docs_raw.json", "w", encoding="utf-8") as f:
    #     json.dump(text_docs_raw, f, ensure_ascii=False, indent=4)
    
    # # Combine both levels
    # text_docs_raw.extend(text_docs_level_3_raw)
    # text_docs_raw.extend(text_docs_level_4_raw)

    # text_docs = []
    # for doc in text_docs_raw:
    #     segmented_text = ViTokenizer.tokenize(doc["text"])
    #     doc["text"] = segmented_text
    #     text_docs.append(doc)

    # print(f"Total text documents loaded: {len(text_docs)}")
    # text_db_chroma = ChromaDB.create_chroma_db(
    #     text_docs, chroma_db_path, name="text_docs", embedding_fn=embedding_fn)
    # bm25_plus, full_documents_bm25 = BM25DB.create_bm25_db(
    #     text_docs, bm25_path, name="text_docs_bm25", language=language)
 
    text_db_chroma = ChromaDB.load_chroma_collection(chroma_db_path, name="text_docs", embedding_fn=embedding_fn)
    bm25_plus, full_documents_bm25 = BM25DB.load_bm25_db(bm25_path)

    response_data(text_db_chroma, bm25_plus, full_documents_bm25,
                  json_folder=json_folder, language=language)
