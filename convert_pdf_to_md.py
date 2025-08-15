from utils.scan_pdf_to_md import convert_pdf_to_markdown, extract_images_and_caption_flexible, extract_tables_from_markdown
import os
import json
from utils.db import ChromaDB
from utils.document_loaders import DocumentLoader
from utils.answer_generator_old import AnswerGenerator


def process_data(pdf_folder, md_folder, image_folder, image_doc_json, table_doc_path):
    """
    Convert PDF files to Markdown, extract images and captions, and extract tables.
    """
    # Convert PDF files to Markdown
    convert_pdf_to_markdown(pdf_folder, md_folder)

    # Extract images and captions from PDFs
    all_image_docs = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            image_docs = extract_images_and_caption_flexible(
                pdf_path, md_folder, image_folder)
            if image_docs:
                all_image_docs.extend(image_docs)
                print(f"Image documents saved to {image_doc_json}")
    with open(image_doc_json, "w", encoding="utf-8") as f:
        json.dump(all_image_docs, f, ensure_ascii=False, indent=4)

    # Extract tables from Markdown files
    extract_tables_from_markdown(md_folder, "output-tables", table_doc_path)


def load_data(pdf_folder, md_folder, image_folder, image_doc_json, table_doc_path, chroma_path, embedding_fn):
    """
    Load text documents, image captions, and table captions from the specified folders.
    """
    documentLoader = DocumentLoader(pdf_folder, image_doc_json)
    text_docs = documentLoader.load_text_documents(pdf_folder)
    image_docs = documentLoader.load_json_documents(image_doc_json, "image")
    table_docs = documentLoader.load_json_documents(table_doc_path, "table")

    print("📂 Loaded:")
    print(f" - {len(text_docs)} text documents")
    print(f" - {len(image_docs)} image captions")
    print(f" - {len(table_docs)} table captions")

    # Create three separate collections
    text_db = ChromaDB.create_chroma_db(
        text_docs, chroma_path, name="text_docs", embedding_fn=embedding_fn)
    image_db = ChromaDB.create_chroma_db(
        image_docs, chroma_path, name="image_docs", embedding_fn=embedding_fn)
    table_db = ChromaDB.create_chroma_db(
        table_docs, chroma_path, name="table_docs", embedding_fn=embedding_fn)

    # text_db = ChromaDB.load_chroma_collection(chroma_path, name="text_docs", embedding_fn=embedding_fn)
    # image_db = ChromaDB.load_chroma_collection(chroma_path, name="image_docs", embedding_fn=embedding_fn)
    # table_db = ChromaDB.load_chroma_collection(chroma_path, name="table_docs", embedding_fn=embedding_fn)

    return text_db, image_db, table_db


def response_data(text_db, image_db, table_db):
    """
    Generate responses based on the loaded data.
    """
    question = "Vùng có nguy cơ gây lóa tạm thời đối với đường thoát nạn theo phương ngang"
    answerGenerator = AnswerGenerator()
    query, answer, images_res, tables_res, text_combined, image_combined, table_combined = answerGenerator.generate_answer_with_source(
        text_db, image_db, table_db, query=question)

    print("\n>> Gemini Answer:\n", answer)


if __name__ == "__main__":
    pdf_folder = "CHECK"
    md_folder = "CHECK_MD"
    image_folder = "output-images"
    image_doc_json = "temp_process/image_doc.json"
    table_doc_path = "temp_process/table_doc.json"
    chroma_path = "database/chroma_db_vn_law_v2"
    embedding_fn = "vn-law-embedding"

    # Determine if processing is needed
    already_md = os.path.isdir(md_folder) and os.listdir(md_folder)
    json_already = os.path.isfile(
        image_doc_json) and os.path.getsize(image_doc_json)

    if already_md and json_already:
        print("Data already processesed. Skipping process_data() function.")
    else:
        process_data(pdf_folder, md_folder, image_folder,
                     image_doc_json, table_doc_path)

    # Load data into ChromaDB
    text_db, image_db, table_db = load_data(
        pdf_folder, md_folder, image_folder, image_doc_json, table_doc_path, chroma_path, embedding_fn)

    # Response generation
    response_data(text_db, image_db, table_db)
