import streamlit as st
from utils import *
import pandas as pd
from utils.answer_generator_old import AnswerGenerator
from utils.db import ChromaDB

answer_generator = AnswerGenerator()
# Step 1: Add model selection
# model_option = st.selectbox("Chọn mô hình embedding:", ["VN Law Embedding", "Gemini Embedding"])
model_option = "VN Law Embedding"  # Default option


@st.cache_resource
def load_db(model_choice):
    if model_choice == "Gemini Embedding":
        chroma_path = "database/chroma_db_gemini"
        embedding_fn = "gemini"
    else:
        chroma_path = "chroma_db"
        # chroma_path = "database/chroma_db_vn-law"
        embedding_fn = "vn-law-embedding"

    text_db = ChromaDB.load_chroma_collection(
        chroma_path, name="text_docs", embedding_fn=embedding_fn)
    # image_db = ChromaDB.load_chroma_collection(chroma_path, name="image_docs", embedding_fn=embedding_fn)
    # table_db = ChromaDB.load_chroma_collection(chroma_path, name="table_docs", embedding_fn=embedding_fn)
    # return text_db, image_db, table_db
    return text_db, None, None


# Load the databases based on selected model
text_db, image_db, table_db = load_db(model_option)

st.title("Vietnam Fire Protection Regulation Dictionary")

query = st.text_input("Nhập câu hỏi của bạn:")

if st.button("Gửi câu hỏi") and query:
    with st.spinner("Đang tìm kiếm câu trả lời..."):
        # query, answer, images_res, tables_res, text_combined, image_combined, table_combined = answer_generator.generate_answer_with_source(
        #     text_db, image_db, table_db, query=query
        # )
        answer = answer_generator.generate_answer(text_db, query=query)
    st.markdown("### Câu hỏi sau khi được viết lại:")
    st.write(query.replace("PCCC", " "))
    st.markdown("### Trả lời:")
    st.write(answer)

    # if images_res:
    #     st.markdown("Hình ảnh liên quan:")
    #     for i, image in enumerate(images_res):
    #         st.image(image, caption=f"Hình ảnh {i+1}", use_container_width=True)

    # if tables_res:
    #     st.markdown("Bảng liên quan:")
    #     for i, table in enumerate(tables_res):
    #         try:
    #             if table.endswith(".csv"):
    #                 df = pd.read_csv(table)
    #                 st.dataframe(df)
    #             else:
    #                 with open(table, "r", encoding="utf-8") as f:
    #                     st.markdown(f.read(), unsafe_allow_html=True)
    #         except Exception as e:
    #             st.warning(f"Lỗi khi tải bảng từ {table}: {e}")

    # st.markdown("---")

    # st.markdown("### Đoạn văn liên quan:")
    # for i, (text_doc, text_metadata, text_distances) in enumerate(text_combined):
    #     if text_doc.strip():
    #         source = text_metadata.get("filename", "Không rõ nguồn")
    #         st.markdown(f"**[Text {i+1}] - nguồn:** {source} - sự khác biệt: {text_distances:.4f}")
    #         st.write(text_doc)
    #         st.markdown("---")

    # st.markdown("### Hình ảnh liên quan:")
    # for i, (image_doc, image_metadata, image_distances) in enumerate(image_combined):
    #     if image_doc.strip():
    #         source = image_metadata.get("url", "Không rõ nguồn")
    #         st.markdown(f"**[Image {i+1}] - nguồn:** {source} - chú thích {image_doc} - sự khác biệt: {image_distances:.4f}")
    #         st.image(source, caption=f"Hình ảnh {i+1}", use_container_width=True)
    #         st.markdown("---")

    # st.markdown("### Bảng liên quan:")
    # for i, (table_doc, table_metadata, table_distances) in enumerate(table_combined):
    #     if table_doc.strip():
    #         source = table_metadata.get("url", "Không rõ nguồn")
    #         st.markdown(f"**[Table {i+1}] - nguồn:** {source} - chú thích {table_doc} - sự khác biệt: {table_distances:.4f}")
    #         try:
    #             if table_doc.endswith(".csv"):
    #                 df = pd.read_csv(table_doc)
    #                 st.dataframe(df)
    #             else:
    #                 with open(source, "r", encoding="utf-8") as f:
    #                     st.markdown(f.read(), unsafe_allow_html=True)
    #         except Exception as e:
    #             st.warning(f"Lỗi khi tải bảng từ {source}: {e}")
    #         st.markdown("---")
